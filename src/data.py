import os

from datasets import (
    concatenate_datasets,
    load_dataset,
    load_from_disk,
    interleave_datasets,
    DatasetDict,
    Features,
    Sequence,
    Value,
)
from functools import partial
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import Any, Dict, List, Optional, Set, Union
import yaml


DEFAULT_COMPRESSION_TYPES = {
    "input_ids": Sequence(Value("int32")),
    "attention_mask": Sequence(Value("bool")),
    "token_length": Value("int32"),
    "id": Value("string"),
}


class StringHandlingDataCollator:
    def __init__(self, hf_collator):
        self.hf_collator = hf_collator

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 1. Extract the string IDs so the HF collator doesn't see them
        ids = [feature.pop("id") for feature in features if "id" in feature]

        # 2. Use the standard HF collator for input_ids, attention_mask, etc.
        # This returns a dictionary of PyTorch tensors
        batch = self.hf_collator(features)

        # 3. Add the IDs back into the batch as a list of strings
        batch["id"] = ids
        return batch


def create_dataset_for_pretraining(
    data_config: Dict[str, Any],
    trainer_config: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    cols_to_keep: Optional[Set[str]] = None,
) -> Dict[str, Union[Dataset, List[Dataset]]]:
    """
    Create datasets for pretraining with optional document repetition.

    Repetition parameters (in data_config):
        repetition_budget: float
            Fraction of total training tokens that should come from repeated
            documents. E.g., 0.1 means 10% of tokens come from repeated docs.
            Default: 0.0 (no repetition).
        num_repeats: int
            Number of times to repeat the selected documents. E.g., 100 means
            each repeated document appears 100 times.
            Default: 1 (no repetition).

    Example:
        If repetition_budget=0.1 and num_repeats=100, then 0.1% of unique
        documents will be repeated 100 times, contributing 10% of total tokens.
    """
    if cols_to_keep is None:
        cols_to_keep = {"input_ids", "attention_mask", "token_length", "id"}

    # TODO: Spin this out to a top level function.
    # https://chatgpt.com/share/68f0657f-fab0-800d-8329-a8c8acf18ac8
    def tokenize_truncate_and_count(example):
        # Tokenize.
        # Make certain we end on EOS. See: https://arxiv.org/abs/2403.17031
        tokenized_input = tokenizer(
            example["text"] + tokenizer.eos_token,
            truncation=True,
            max_length=trainer_config["max_length"],
        )
        # Make sure we end on an EOS token ID.
        if tokenized_input["input_ids"][-1] != tokenizer.eos_token_id:
            tokenized_input["input_ids"].append(tokenizer.eos_token_id)
            tokenized_input["attention_mask"].append(1)
        example["input_ids"] = tokenized_input["input_ids"]
        example["attention_mask"] = tokenized_input["attention_mask"]
        # Count the number of tokens.
        example["token_length"] = len(tokenized_input["input_ids"])
        return example

    # Specify where to cache rank-0 tokenized artifacts so other ranks can just load
    hf_cache_root = os.getenv("HF_DATASETS_CACHE") or os.path.join(
        os.getcwd(), ".hf_cache"
    )
    os.makedirs(hf_cache_root, exist_ok=True)
    corpus_train_dataset_subset_cache_dir = os.path.join(
        hf_cache_root, "corpus_subset_tokenized"
    )
    corpus_eval_dataset_cache_dir = os.path.join(hf_cache_root, "corpus_eval_tokenized")

    # claude suggests this change
    # cache_is_ready = os.path.exists(
    #     os.path.join(corpus_train_dataset_subset_cache_dir, "state.json")
    # ) and os.path.exists(
    #     os.path.join(corpus_eval_dataset_cache_dir, "state.json")
    # )

    # if _is_main() and not cache_is_ready:

    if _is_main():
        num_proc = min(64, os.cpu_count())

        num_train_epochs = trainer_config["num_train_epochs"]
        num_training_tokens_per_epoch = trainer_config["num_training_tokens_per_epoch"]
        target_num_training_tokens_total = trainer_config[
            "target_num_training_tokens_total"
        ]

        if data_config["corpus"] == "fineweb-edu-dedup":
            corpus_full_dataset = load_dataset(
                "HuggingFaceTB/smollm-corpus",
                "fineweb-edu-dedup",
                split="train",  # This is the only split that exists.
                cache_dir=os.environ.get("HF_HOME", "/data/hf_home"),
                num_proc=num_proc,
            )
            # The full dataset is 220B tokens in 190,168,005 rows.
            # We want 150M tokens for test.
            corpus_split_dataset = corpus_full_dataset.train_test_split(
                test_size=150e6 / 220e9,
                seed=data_config["train_test_split_seed"],
            )
            print("Split corpus into train and test")
            corpus_train_dataset = corpus_split_dataset["train"]
            corpus_eval_dataset = corpus_split_dataset["test"]
            avg_tokens_per_doc = 794
        else:
            raise ValueError

        # Extract repetition parameters from data_config.
        # repetition_budget: fraction of total tokens from repeated docs (e.g., 0.1 = 10%)
        # num_repeats: how many times each repeated doc appears (e.g., 100)
        # If budget=0.1 and repeats=100, then 0.1% of unique docs are repeated 100x.
        repetition_budget = data_config.get("repetition_budget", 0.0)
        num_repeats = data_config.get("num_repeats", 1)

        # Validate repetition parameters.
        if not (0.0 <= repetition_budget < 1.0):
            raise ValueError(
                f"repetition_budget must be in [0, 1), got {repetition_budget}"
            )
        if num_repeats < 1:
            raise ValueError(f"num_repeats must be >= 1, got {num_repeats}")

        # Calculate unique tokens needed.
        # With repetition: we need (1 - budget) * T unique non-repeated tokens,
        # plus budget * T / num_repeats unique repeated tokens.
        # Total unique tokens = T * (1 - budget + budget / num_repeats)
        #                     = T * (1 - budget * (1 - 1/num_repeats))
        if repetition_budget > 0 and num_repeats > 1:
            unique_token_factor = 1 - repetition_budget * (1 - 1 / num_repeats)
            unique_tokens_needed = int(num_training_tokens_per_epoch * unique_token_factor)
            print(
                f"Repetition enabled: budget={repetition_budget:.2%}, "
                f"num_repeats={num_repeats}, unique_token_factor={unique_token_factor:.4f}"
            )
        else:
            unique_tokens_needed = num_training_tokens_per_epoch

        # Subsample the appropriate number of documents without replacement.
        print("Shuffling, subsampling and tokenizing the pretraining corpus.")
        estimated_docs_needed = int(1.1 * unique_tokens_needed / avg_tokens_per_doc)
        num_total_docs = len(corpus_train_dataset)
        all_indices = np.arange(num_total_docs)
        rng = np.random.default_rng(data_config["shuffle_seed"])
        rng.shuffle(all_indices)
        if data_config["direction"] == "top":
            # Work from the start of the shuffled list forwards
            active_indices = all_indices
        elif data_config["direction"] == "bot":
            # Work from the end of the shuffled list backwards
            active_indices = all_indices[::-1].copy()
        else:
            raise ValueError(
                f"Impermissible value of direction (must be 'top' or 'bot'): {data_config['direction']}"
            )

        corpus_train_dataset_subset = corpus_train_dataset.select(
            active_indices[:estimated_docs_needed]
        ).map(tokenize_truncate_and_count, num_proc=num_proc)

        # Handle document selection with optional repetition.
        token_lengths = np.array(corpus_train_dataset_subset["token_length"])
        cumulative_lengths = np.cumsum(token_lengths)

        if repetition_budget > 0 and num_repeats > 1:
            # Calculate token targets for each portion.
            repeated_tokens_target = int(
                num_training_tokens_per_epoch * repetition_budget / num_repeats
            )
            non_repeated_tokens_target = int(
                num_training_tokens_per_epoch * (1 - repetition_budget)
            )

            # Find cutoff for repeated docs (first portion of the shuffled docs).
            idx_repeated_end = int(
                np.searchsorted(cumulative_lengths, repeated_tokens_target)
            )
            # Ensure we have at least 1 doc to repeat if budget > 0.
            idx_repeated_end = max(1, idx_repeated_end + 1)

            # Calculate actual repeated tokens and remaining target.
            actual_repeated_unique_tokens = int(cumulative_lengths[idx_repeated_end - 1])
            actual_repeated_total_tokens = actual_repeated_unique_tokens * num_repeats

            # Select non-repeated docs from the remaining documents.
            remaining_cumsum = np.cumsum(token_lengths[idx_repeated_end:])
            idx_non_repeated_end = int(
                np.searchsorted(remaining_cumsum, non_repeated_tokens_target)
            )
            idx_non_repeated_end = idx_repeated_end + idx_non_repeated_end + 1

            # Extract the two portions.
            repeated_docs = corpus_train_dataset_subset.select(range(idx_repeated_end))
            non_repeated_docs = corpus_train_dataset_subset.select(
                range(idx_repeated_end, idx_non_repeated_end)
            )

            actual_non_repeated_tokens = int(np.sum(non_repeated_docs["token_length"]))

            print(
                f"Repeated docs: {len(repeated_docs):,} docs × {num_repeats} repeats = "
                f"{actual_repeated_total_tokens:,} tokens "
                f"({actual_repeated_total_tokens / num_training_tokens_per_epoch:.2%} of target)"
            )
            print(
                f"Non-repeated docs: {len(non_repeated_docs):,} docs = "
                f"{actual_non_repeated_tokens:,} tokens "
                f"({actual_non_repeated_tokens / num_training_tokens_per_epoch:.2%} of target)"
            )

            # OLD: concatenate_datasets + shuffle + flatten_indices took 2+ hours for high
            # num_repeats (e.g. 10000) because flatten_indices had to materialize ~1.74M rows.
            # That blocked the torchrun process long enough to time out the c10d rendezvous
            # heartbeat (60s). Commented out in favor of the select() approach below.
            #
            # repeated_copies = [repeated_docs] * num_repeats
            # corpus_train_dataset_subset = concatenate_datasets(
            #     repeated_copies + [non_repeated_docs]
            # )
            # corpus_train_dataset_subset = corpus_train_dataset_subset.shuffle(
            #     seed=data_config["shuffle_seed"]
            # )
            # corpus_train_dataset_subset = corpus_train_dataset_subset.flatten_indices(num_proc=num_proc)

            # Build the final index array in numpy: repeated doc indices appear num_repeats
            # times, non-repeated indices once. Shuffle in numpy, then use .select() which
            # stores a lazy index mapping — no physical copy / flatten needed.
            # NOTE: this uses numpy's RNG with the same seed, so the shuffle ordering will
            # differ from the old HF-based shuffle. Results are not bit-for-bit reproducible
            # with any runs that used the old code path.
            repeated_idx = np.repeat(np.arange(idx_repeated_end), num_repeats)
            non_repeated_idx = np.arange(idx_repeated_end, idx_non_repeated_end)
            all_final_indices = np.concatenate([repeated_idx, non_repeated_idx])
            rng_final = np.random.default_rng(data_config["shuffle_seed"])
            rng_final.shuffle(all_final_indices)
            corpus_train_dataset_subset = corpus_train_dataset_subset.select(all_final_indices)

        else:
            # Standard case: no repetition.
            # Find the index where we exceed the target.
            idx_to_keep = int(
                np.searchsorted(cumulative_lengths, num_training_tokens_per_epoch)
            )
            # Select up to that index (+1 to be safe or inclusive).
            corpus_train_dataset_subset = corpus_train_dataset_subset.select(
                range(idx_to_keep + 1)
            )

        # Cut the Arrow buffers in half by casting dtypes before saving (no semantic change).
        # Remove unnecessary columns to reduce size, then save to disk.
        cols_to_drop = [
            c for c in corpus_train_dataset_subset.column_names if c not in cols_to_keep
        ]
        corpus_train_dataset_subset = corpus_train_dataset_subset.remove_columns(
            cols_to_drop
        )
        corpus_train_dataset_subset = corpus_train_dataset_subset.cast(
            Features(
                {
                    k: v
                    for k, v in DEFAULT_COMPRESSION_TYPES.items()
                    if k in cols_to_keep
                }
            ),
            num_proc=num_proc,
        )
        corpus_train_dataset_subset.save_to_disk(
            corpus_train_dataset_subset_cache_dir,
        )

        # Now, we turn to the eval dataset: tokenize, truncate, count, write to disk, etc.
        corpus_eval_dataset = corpus_eval_dataset.map(
            tokenize_truncate_and_count,
            num_proc=num_proc,
        )
        cols_to_drop_eval = [
            c for c in corpus_eval_dataset.column_names if c not in cols_to_keep
        ]
        corpus_eval_dataset = corpus_eval_dataset.remove_columns(cols_to_drop_eval)
        corpus_eval_dataset = corpus_eval_dataset.cast(
            Features(
                {
                    k: v
                    for k, v in DEFAULT_COMPRESSION_TYPES.items()
                    if k in cols_to_keep
                }
            ),
            num_proc=num_proc,
        )
        corpus_eval_dataset.save_to_disk(
            corpus_eval_dataset_cache_dir,
        )

        total_tokens_per_epoch = np.sum(corpus_train_dataset_subset["token_length"])
        print(
            f"Final dataset created with {total_tokens_per_epoch:,} tokens.\n"
            f"With {num_train_epochs:,} training epochs, total training tokens: {num_train_epochs * total_tokens_per_epoch:,}\n"
            f"Target number of total training tokens: {target_num_training_tokens_total:,}\n"
        )

    if (
        _world_size() > 1
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        torch.distributed.barrier()  # non-zero ranks wait for rank 0 to finish

    # All processes load the datasets from disk.
    corpus_train_dataset_subset = load_from_disk(corpus_train_dataset_subset_cache_dir)
    corpus_eval_dataset = load_from_disk(corpus_eval_dataset_cache_dir)

    datasets_dict = {
        "train": corpus_train_dataset_subset,
        "eval": corpus_eval_dataset,
    }

    return datasets_dict


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _is_main() -> bool:
    return _rank() == 0


def _is_sweep_run() -> bool:
    return os.environ.get("WANDB_SWEEP_ID") is not None
