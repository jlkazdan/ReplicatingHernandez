import os

# Rok asked us to include the following specifications in our code to prevent CPUs from spinning idly:
WORLD_SIZE_ENV = int(os.environ.get("WORLD_SIZE", "1"))
if WORLD_SIZE_ENV <= 1:
    n_threads_str = "32"
else:
    # Under DDP we keep BLAS threads minimal; dataset workers provide the parallelism
    n_threads_str = "1"
os.environ["OMP_NUM_THREADS"] = n_threads_str
os.environ["OPENBLAS_NUM_THREADS"] = n_threads_str
os.environ["MKL_NUM_THREADS"] = n_threads_str
os.environ["VECLIB_MAXIMUM_THREADS"] = n_threads_str
os.environ["NUMEXPR_NUM_THREADS"] = n_threads_str
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# This is needed for deterministic to work.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# 16.48 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large
# try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


import logging
import pprint
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoTokenizer, DataCollatorWithPadding, set_seed
from typing import Any, Dict, List
import wandb

import src.data
import src.globals
import src.models


logging.basicConfig(level=logging.INFO)


def eval_language_model():
    assert torch.cuda.device_count() > 0, "No CUDA devices available."
    run = wandb.init(
        project="scaling-memorization-eval",
        config=src.globals.DEFAULT_EVALUATION_CONFIG,
        entity=wandb.api.default_entity,
    )

    # Convert to a dictionary; otherwise, can't distribute because W&B
    # config is not pickle-able.
    wandb_config: Dict[str, Any] = dict(wandb.config)
    print("CUDA VISIBLE DEVICES: ", os.environ["CUDA_VISIBLE_DEVICES"])
    pprint.pprint(wandb_config)

    set_seed(seed=wandb_config["seed"], deterministic=True)

    # We need to slightly hijack the pretraining logic.
    num_training_tokens_per_epoch = int(
        wandb_config["trainer_config"]["target_num_training_tokens_total"]
        / wandb_config["trainer_config"]["num_train_epochs"]
    )
    wandb_config["trainer_config"][
        "num_training_tokens_per_epoch"
    ] = num_training_tokens_per_epoch

    score_lm_nll_on_datasets(wandb_config=wandb_config)

    wandb.finish()


def score_lm_nll_on_datasets(wandb_config: Dict[str, Any]):
    # Load model and its tokenizer.
    model = src.models.create_causalm_for_pretraining(
        model_config_dict=wandb_config["model_config"],
    )
    model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(
        wandb_config["model_config"]["model_name"],
        use_fast=True,
        trust_remote_code=True,
    )

    # Create the dataset.
    datasets_dict = src.data.create_dataset_for_pretraining(
        data_config=wandb_config["data_config"],
        trainer_config=wandb_config["trainer_config"],
        tokenizer=tokenizer,
        cols_to_keep={"input_ids", "attention_mask", "token_length", "id"},
    )

    # We use CrossEntropyLoss with reduction='none' to get token-level loss
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    data_collator = src.data.StringHandlingDataCollator(
        DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,
            max_length=wandb_config["trainer_config"]["max_length"] + 1,
            return_tensors="pt",
        )
    )

    print("Starting Evaluation Loop...")
    model.eval()
    for split, dataset in datasets_dict.items():
        num_seen_tokens = 0

        dataloader = DataLoader(
            dataset,
            batch_size=wandb_config["trainer_config"]["batch_size"],
            collate_fn=data_collator,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

        max_length = wandb_config["trainer_config"]["max_length"]
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"][:, :max_length].to(model.device)
            attention_mask = batch["attention_mask"][:, :max_length].to(model.device)
            uuids = batch["id"]

            with torch.no_grad():
                logits_BLV = model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits

                # Shift logits_BLV and labels for causal LM loss
                # Logits: [B, Seq, Vocab] -> predict next token
                # Shift: logits_BLV[..., :-1, :] predicts input_ids[..., 1:]
                shift_logits_BLV = logits_BLV[..., :-1, :]
                shift_labels_BL = input_ids[..., 1:]
                shift_mask = attention_mask[..., 1:]

                # Calculate Loss (NLL) per token
                shift_logits_BVL = shift_logits_BLV.swapaxes(1, 2)
                loss_BL = loss_fct(shift_logits_BVL, shift_labels_BL)

                # Mask out padding tokens
                loss_BL = loss_BL * shift_mask

                # Sum NLL per sequence
                nll_mean_B = loss_BL.mean(dim=1)

                # Count valid tokens per sequence (for averaging later if needed)
                valid_tokens_B = shift_mask.sum(dim=1)

                # Store results
                nll_np_B = nll_mean_B.float().cpu().numpy()
                lens_np_B = valid_tokens_B.float().cpu().numpy()

                for uuid, nll, length in zip(uuids, nll_np_B, lens_np_B):
                    results_to_log = {
                        "split": split,
                        "seq_token_length": length,
                        "avg_nll": nll,
                        "id": uuid,
                    }
                    wandb.log(results_to_log)
                    # Be nicer to W&B, even if that takes more time per run.
                    time.sleep(1.0 / 60.0)

                num_seen_tokens += valid_tokens_B.sum()

                # By default, the "test" split has ~150M tokens.
                # We only want target_num_training_tokens_total.
                if split == "eval":
                    if (
                        num_seen_tokens
                        > wandb_config["trainer_config"][
                            "target_num_training_tokens_total"
                        ]
                    ):
                        break


if __name__ == "__main__":
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(i) for i in range(torch.cuda.device_count())]
        )
    eval_language_model()
    logging.info("Finished eval_language_model.py!")
