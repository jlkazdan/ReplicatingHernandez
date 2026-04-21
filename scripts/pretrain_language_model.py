import os
import datasets
print(datasets.config.HF_DATASETS_CACHE)
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

import datetime
import logging
import gc
import math
import numpy as np
import pprint
import shutil
import time
import torch

# Graph break from `Tensor.item()`, consider setting:
#      torch._dynamo.config.capture_scalar_outputs = True
#  or:
#      env TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
#  to include these operations in the captured graph.
#
#  Graph break: from user code at:
#    File "/lfs/skampere2/0/rschaef/KoyejoLab-Scaling-Memorization/scaling_mem_env/lib/python3.12/site-packages/transformers/modeling_flash_attention_utils.py", line 354, in torch_dynamo_res
#
#      max_length_q = max_length_q.item()
#
torch._dynamo.config.capture_scalar_outputs = True

# Compiling seems to be causing problems down the line :/
torch.compiler.disable()
import torch.distributed
import torch.nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithFlattening,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    set_seed,
)
from typing import Any, Dict
import wandb

import src.data
import src.globals
import src.models


logging.basicConfig(level=logging.INFO)


def pretrain():
    # Set the device for the current process
    torch.cuda.set_device(_local_rank())
    assert _world_size() > 0, "No CUDA devices available."

    if _world_size() > 1 and not torch.distributed.is_initialized():
        # We need to increase the timeout for tokenizing the dataset.
        # 10 minutes is default. 150 minutes should be ample.
        torch.distributed.init_process_group(
            backend="nccl", timeout=datetime.timedelta(minutes=480)
        )

    print("CUDA VISIBLE DEVICES: ", os.environ["CUDA_VISIBLE_DEVICES"])
    print(
        f"RANK={_rank()} LOCAL_RANK={_local_rank()} WORLD_SIZE={_world_size()} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}"
    )

    run, run_id, wandb_config, pted_model_hf_name = initialize_wandb()
    pprint.pprint(wandb_config)

    # Create the output directory.
    output_dir = os.path.join("models", "pt_language_model", pted_model_hf_name)
    if _is_main():
        wandb.config.update({"output_dir": output_dir}, allow_val_change=True)
    print("Output Directory: ", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    set_seed(seed=wandb_config["seed"], deterministic=True)

    if wandb_config["model_config"]["model_name"].startswith("Qwen3/Qwen3-"):
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-4B-Base",
            use_fast=True,
            trust_remote_code=True,
        )
        tokenizer.model_max_length = wandb_config["trainer_config"]["max_length"]
        # 1) Ensure a distinct padding token exists and is NOT the EOS.
        if (
            tokenizer.pad_token_id is None
            or tokenizer.pad_token_id == tokenizer.eos_token_id
        ):
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    else:
        raise NotImplementedError

    model: PreTrainedModel = src.models.create_causalm_for_pretraining(
        model_config_dict=wandb_config["model_config"],
    )

    # Resize embeddings so the model knows about the new token.
    model.resize_token_embeddings(len(tokenizer))

    # Set consistent config for training *and* generation.
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    # 4) Right-pad for training; left-pad for batched generation is fine later
    tokenizer.padding_side = "right"

    wandb_config = compute_derived_hyperparameters(
        model=model,
        wandb_config=wandb_config,
    )

    # Lightly modify configs as necessary.
    if wandb_config["model_config"]["torch_dtype"] == "bfloat16":
        wandb_config["trainer_config"]["bf16"] = True
    else:
        wandb_config["trainer_config"]["bf16"] = False
    if wandb_config["model_config"]["torch_dtype"] == "float16":
        wandb_config["trainer_config"]["fp16"] = True
    else:
        wandb_config["trainer_config"]["fp16"] = False

    pretraining_config = TrainingArguments(
        adam_beta1=wandb_config["trainer_config"]["adam_beta1"],
        adam_beta2=wandb_config["trainer_config"]["adam_beta2"],
        bf16=wandb_config["trainer_config"]["bf16"],
        data_seed=wandb_config["trainer_config"]["data_seed"],
        dataloader_drop_last=wandb_config["trainer_config"]["dataloader_drop_last"],
        dataloader_num_workers=wandb_config["trainer_config"]["dataloader_num_workers"],
        dataloader_prefetch_factor=wandb_config["trainer_config"][
            "dataloader_prefetch_factor"
        ],
        save_steps=wandb_config["trainer_config"].get("save_steps", 500),
        ddp_backend="nccl" if _world_size() > 1 else None,
        ddp_find_unused_parameters=False if _world_size() > 1 else None,
        eval_on_start=wandb_config["trainer_config"]["eval_on_start"],
        eval_strategy=wandb_config["trainer_config"]["eval_strategy"],
        eval_steps=wandb_config["trainer_config"]["eval_steps"],
        fp16=wandb_config["trainer_config"]["fp16"],
        gradient_accumulation_steps=wandb_config["trainer_config"][
            "gradient_accumulation_steps"
        ],
        gradient_checkpointing=wandb_config["trainer_config"]["gradient_checkpointing"],
        hub_model_id=f"jkazdan/{pted_model_hf_name}",
        hub_private_repo=True,
        hub_strategy=wandb_config["trainer_config"]["hub_strategy"],
        include_num_input_tokens_seen=True,
        learning_rate=float(wandb_config["trainer_config"]["learning_rate"]),
        logging_steps=wandb_config["trainer_config"]["logging_steps"],
        lr_scheduler_type=wandb_config["trainer_config"]["lr_scheduler_type"],
        max_grad_norm=wandb_config["trainer_config"]["max_grad_norm"],
        max_steps=-1,
        # metric_for_best_model="eval_benchmark_loss",
        # metric_for_best_model="num_input_tokens_seen",
        # greater_is_better=True,
        num_train_epochs=wandb_config["trainer_config"]["num_train_epochs"],
        optim=wandb_config["trainer_config"]["optim"],
        output_dir=output_dir,
        per_device_eval_batch_size=wandb_config["trainer_config"][
            "per_device_eval_batch_size"
        ],
        per_device_train_batch_size=wandb_config["trainer_config"][
            "per_device_train_batch_size"
        ],
        remove_unused_columns=wandb_config["trainer_config"]["remove_unused_columns"],
        run_name=run_id,
        report_to=wandb_config["trainer_config"]["report_to"],
        save_strategy=wandb_config["trainer_config"]["save_strategy"],
        save_total_limit=wandb_config["trainer_config"]["save_total_limit"],
        seed=wandb_config["seed"],
        torch_compile=wandb_config["trainer_config"]["torch_compile"],
        # warmup_steps=wandb_config["trainer_config"]["warmup_steps"],
        warmup_ratio=wandb_config["trainer_config"]["warmup_ratio"],
        weight_decay=wandb_config["trainer_config"]["weight_decay"],
    )

    datasets_dict = src.data.create_dataset_for_pretraining(
        data_config=wandb_config["data_config"],
        trainer_config=wandb_config["trainer_config"],
        tokenizer=tokenizer,
    )
    train_dataset = datasets_dict["train"]
    eval_dataset = datasets_dict["eval"]
    if _is_main():
        wandb.config.data_config.update(
            {
                "train_dataset_num_tokens": np.sum(train_dataset["token_length"]),
                "eval_dataset_num_tokens": np.sum(eval_dataset["token_length"]),
            }
        )

    # Apply the preparation to both the training and evaluation splits.
    train_dataset = prepare_dataset_for_model(train_dataset)
    eval_dataset = prepare_dataset_for_model(eval_dataset)

    data_collator = DataCollatorWithFlattening(
        return_position_ids=True,  # default True; explicit for clarity.
        separator_id=-100,  # ensures no cross-example predictions.
    )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=pretraining_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train.
    if _is_main():
        logging.info("Beginning training...")
    trainer.train()

    # Evaluate after training.
    if _is_main():
        logging.info("Finished training. Beginning final evaluation...")
    eval_metrics_after = trainer.evaluate()
    if _is_main():
        wandb.log({f"eval_after/{k}": v for k, v in eval_metrics_after.items()})
        pprint.pprint(eval_metrics_after)

    # Push to HF Hub.
    if _is_main():
        logging.info(f"Finished final evaluation. Saving to disk...")
        tokenizer.padding_side = "left"  # Otherwise, generate later gets screwed up.
        tokenizer.save_pretrained(pretraining_config.output_dir)
        trainer.save_model(output_dir=pretraining_config.output_dir)
        # logging.info(f"Finished final evaluation. Pushing to HuggingFace...")
        # trainer.push_to_hub()
        # logging.info("Pushed to HuggingFace.")

    # For some reason, the trainer holds onto GPU memory even after finishing.
    # There might be a smarter way of freeing up the memory, but here's my workaround.
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(5)

    if _is_main():
        # Delete the dataset from disk to save disk space.
        cache_dir = os.environ.get("HF_DATASETS_CACHE")
        if cache_dir and os.path.exists(cache_dir):
            logging.info(f"Run finished. Deleting cached dataset at: {cache_dir}")
            try:
                shutil.rmtree(cache_dir)
                logging.info("Successfully deleted cached dataset.")
            except OSError as e:
                logging.error(f"Error deleting cache directory {cache_dir}: {e}")

        wandb.finish()

    cleanup()


def cleanup():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def compute_derived_hyperparameters(
    model: AutoModelForCausalLM, wandb_config: Dict[str, Any]
) -> Dict[str, Any]:
    # 1. Calculate the number of model parameters.
    num_parameters = sum(p.numel() for p in model.parameters())

    # 2. Compute the target number of training tokens.
    target_num_training_tokens_total = int(
        20 * wandb_config["trainer_config"]["overtrain_multiplier"] * num_parameters
    )

    # 3. Compute a reasonable batch size, according to https://arxiv.org/abs/2412.01505.
    num_tokens_per_optimizer_step = int(
        3.24 * np.power(10, 3) * np.power(target_num_training_tokens_total, 0.264)
    )

    # 4. Compute the number of sequences.
    # num_visible_devices = torch.cuda.device_count()
    num_tokens_per_forward_pass = (
        _world_size()
        * wandb_config["trainer_config"]["per_device_train_batch_size"]
        * wandb_config["trainer_config"]["max_length"]
    )
    gradient_accumulation_steps = math.ceil(
        num_tokens_per_optimizer_step / num_tokens_per_forward_pass
    )

    # 4. Compute the number of training tokens per epoch.
    num_training_tokens_per_epoch = int(
        target_num_training_tokens_total
        / wandb_config["trainer_config"]["num_train_epochs"]
    )

    # 5. Calculate the learning rate. It should grow with the square root of the batch size.
    learning_rate = wandb_config["trainer_config"]["base_learning_rate"] * np.sqrt(
        num_tokens_per_optimizer_step
    )

    additional_trainer_config_data = {
        "gradient_accumulation_steps_unrounded": num_tokens_per_optimizer_step
        / num_tokens_per_forward_pass,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "num_visible_devices": torch.cuda.device_count(),  # local (per process)
        "world_size": _world_size(),  # global (all processes)
        "num_tokens_per_forward_pass": num_tokens_per_forward_pass,
        "num_tokens_per_optimizer_step": num_tokens_per_optimizer_step,
        "num_training_tokens_per_epoch": num_training_tokens_per_epoch,
        "target_num_training_tokens_total": target_num_training_tokens_total,
    }

    pprint.pprint(
        additional_trainer_config_data,
    )

    # Write to W&B.
    if _is_main():
        wandb.config.trainer_config.update(additional_trainer_config_data)

    # Add to our W&B config that controls everything.
    wandb_config["trainer_config"].update(additional_trainer_config_data)

    return wandb_config


def create_pretrained_model_huggingface_name(wandb_config: Dict[str, Any]) -> str:
    init_model_name = wandb_config["model_config"]["model_name"].split("/")[-1]
    num_train_epochs = wandb_config["trainer_config"]["num_train_epochs"]
    overtrain_multiplier = wandb_config["trainer_config"]["overtrain_multiplier"]
    seed = wandb_config["seed"]
    direction = wandb_config["data_config"]["direction"]
    shuffle_seed = wandb_config["data_config"]["shuffle_seed"]
    train_test_split_seed = wandb_config["data_config"]["train_test_split_seed"]
    repetition_budget = wandb_config["data_config"].get("repetition_budget", 0.0)
    num_repeats = wandb_config["data_config"].get("num_repeats", 1)
    pted_model_hf_name = f"scale_mem_{init_model_name}_epch_{num_train_epochs}_ot_{overtrain_multiplier}_s_{seed}_dir_{direction}_shfs_{shuffle_seed}_ttss_{train_test_split_seed}_rb_{repetition_budget}_nr_{num_repeats}"
    if len(pted_model_hf_name) > 94:
        raise ValueError(f"pted_model_hf_name is too long: {pted_model_hf_name}")
    return pted_model_hf_name


def prepare_dataset_for_model(dataset: Dataset) -> Dataset:
    """Prepares a dataset for the Trainer by adding labels and removing unneeded columns."""
    # Remove all columns that are not expected by the model.
    columns_to_keep = ["input_ids", "attention_mask"]
    columns_to_remove = [
        col for col in dataset.column_names if col not in columns_to_keep
    ]
    dataset = dataset.remove_columns(columns_to_remove)
    return dataset


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


def initialize_wandb():
    run = None
    run_id = None
    cfg_dict = None

    if _is_main():
        if _is_sweep_run():
            run = wandb.init()
        else:
            run = wandb.init(
                project="scaling-memorization-pt",
                entity="rylan",
                config=src.globals.DEFAULT_PRETRAINING_CONFIG,
            )
        run_id = run.id
        # Get a plain dict so it's pickle and/or broadcast-friendly.
        cfg_dict = dict(wandb.config)
    else:
        # Do not initialize wandb at all on non-zero ranks
        # (safer than disabled mode because it prevents accidental reads)
        pass

    # Make sure the process group is up before broadcast.
    if _world_size() > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    # Broadcast run_id and cfg_dict from rank 0
    if _world_size() > 1:
        obj_list = [run_id, cfg_dict]
        torch.distributed.broadcast_object_list(obj_list, src=0)
        run_id, cfg_dict = obj_list

    pted_model_hf_name = create_pretrained_model_huggingface_name(
        wandb_config=cfg_dict,
    )

    # Use a consistent per-run HF datasets cache across all ranks.
    os.environ[
        "HF_DATASETS_CACHE"
    ] = f"{os.environ.get('HF_DATASETS_CACHE', '/lfs/skampere1/0/shared_hf_cache/datasets')}/cached_datasets/{pted_model_hf_name}"
    
    return run, run_id, cfg_dict, pted_model_hf_name


if __name__ == "__main__":
    pretrain()
    logging.info("Finished pretrain_language_model.py!")
