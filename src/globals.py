DEFAULT_EVALUATION_CONFIG = {
    "data_config": {
        "corpus": "fineweb-edu-dedup",
        "direction": "bot",
        # "direction": "top",
        "shuffle_seed": 0,
        "train_test_split_seed": 0,
    },
    "model_config": {
        "attn_implementation": "flash_attention_2",
        "model_name": "RylanSchaeffer/scale_mem_Qwen3-48M_epch_1_ot_1_s_0_dir_bot_shfs_1_ttss_0",
        "torch_dtype": "bfloat16",
    },
    "trainer_config": {
        "batch_size": 32,
        "max_length": 2048,
        "num_train_epochs": 1,
        "target_num_training_tokens_total": 10_000_000,
    },
    "seed": 0,
}

DEFAULT_PRETRAINING_CONFIG = {
    "data_config": {
        "corpus": "fineweb-edu-dedup",
        "direction": "bot",
        # "direction": "top",
        "shuffle_seed": 0,
        "train_test_split_seed": 0,
        # Repetition parameters for studying repeated data in pretraining.
        # repetition_budget: fraction of tokens from repeated docs (e.g., 0.1 = 10%)
        # num_repeats: how many times each repeated doc appears (e.g., 100)
        # Example: budget=0.1, repeats=100 → 0.1% of docs repeated 100x → 10% of tokens
        "repetition_budget": 0.0,  # Default: no repetition
        "num_repeats": 1,  # Default: no repetition
    },
    "model_config": {
        "attn_implementation": "flash_attention_2",
        # "model_name": "Qwen3/Qwen3-34M",
        "model_name": "Qwen3/Qwen3-660M",
        # "model_name": "Qwen3/Qwen3-48M",
        # "model_name": "Qwen3/Qwen3-63M",
        # "model_name": "Qwen3/Qwen3-93M",
        # "model_name": "Qwen3/Qwen3-344M",
        # "model_name": "Qwen3/Qwen3-806M",
        "torch_dtype": "bfloat16",
    },
    "trainer_config": {
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "data_seed": 0,
        "dataloader_drop_last": True,
        "dataloader_num_workers": 4,
        "dataloader_prefetch_factor": 4,
        "eval_on_start": True,
        "eval_strategy": "steps",
        "eval_steps": 100,
        "gradient_checkpointing": False,
        # "gradient_checkpointing": True,
        "hub_strategy": "end",
        "base_learning_rate": 0.000001,
        "logging_steps": 1,
        # "lr_scheduler_type": "constant_with_warmup",
        "lr_scheduler_type": "cosine",
        "max_grad_norm": 1.0,
        "max_length": 2048,
        "max_steps": -1,
        "num_train_epochs": 1,
        "optim": "adamw_torch",
        # "overtrain_multiplier": 0.1,
        # "overtrain_multiplier": 0.05,
        "overtrain_multiplier": 0.01,
        "per_device_eval_batch_size": 23,
        # "per_device_train_batch_size": 4,
        "per_device_train_batch_size": 23,
        "remove_unused_columns": False,
        # "remove_unused_columns": True,
        "report_to": "wandb",
        "save_strategy": "no",
        "save_total_limit": 1,
        "torch_compile": False,
        "warmup_ratio": 0.2,
        "weight_decay": 0.01,
    },
    "seed": 0,
}


# TODO: Come up with a more elegant solution.
# https://qwenlm.github.io/blog/qwen2.5/
MODEL_NAMES_TO_PARAMETERS_DICT = {
    "34M": 34e6,
    "48M": 48e6,
    "63M": 63e6,
    "93M": 93e6,
    "153M": 153e6,
    "344M": 344e6,
    "1.44B": 1.44e9,
    "Qwen3-34M": 34e6,
    "Qwen3-48M": 48e6,
    "Qwen3-63M": 63e6,
    "Qwen3-93M": 93e6,
    "Qwen3-153M": 153e6,
    "Qwen3-344M": 344e6,
    "Qwen3-1.44B": 1.44e9,
    "Qwen2.5-0.5B": 0.49e9,
    "Qwen2.5-1.5B": 1.5e9,
    "Qwen2.5-3B": 3.1e9,
    "Qwen2.5-7B": 7.6e9,
    "Qwen2.5-14B": 14.7e9,
    "Qwen2.5-32B": 32.5e9,
    "Qwen2.5-72B": 72.7e9,
}
