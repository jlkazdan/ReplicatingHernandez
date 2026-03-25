from accelerate import infer_auto_device_map, dispatch_model
import math
import numpy as np
import os
import pprint
import torch
import torch.utils.data
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from typing import Any, Dict, List, Optional, Tuple, Union

qwen3_parameters_to_depths_and_widths = {
    "2M": (1, 6),
    "16M": (2, 48),
    "34M": (3, 96),
    "48M": (4, 128),
    "63M": (5, 160),
    "93M": (6, 224),
    "111M": (7, 256),
    "153M": (9, 320),
    "191M": (10, 384),
    "262M": (12, 480),
    "344M": (14, 576),
    "499M": (18, 704),
    "660M": (21, 832),
    "806M": (23, 940),
    "934M": (25, 1010),
    "1.08B": (27, 1100),
    "1.26B": (29, 1180),
    "1.44B": (31, 1260),
}


def create_causalm_for_pretraining(
    model_config_dict: Dict[str, Any]
) -> PreTrainedModel:
    model_name = model_config_dict["model_name"]

    if model_name.startswith("Qwen3/Qwen3-"):
        if model_config_dict["torch_dtype"] == "bfloat16":
            torch_dtype = torch.bfloat16
        elif model_config_dict["torch_dtype"] == "float16":
            torch_dtype = torch.float16
        elif model_config_dict["torch_dtype"] == "float32":
            torch_dtype = torch.float32
        else:
            raise NotImplementedError

        from transformers import Qwen3Config, Qwen3ForCausalLM

        num_parameters_str: str = model_config_dict["model_name"].split("-")[1]
        depth, width = qwen3_parameters_to_depths_and_widths[num_parameters_str]
        intermediate_size = 256 * math.floor((255 + math.floor(8 * width / 3)) / 256)

        model_config = Qwen3Config(
            hidden_size=width,
            num_hidden_layers=depth,
            intermediate_size=intermediate_size,
            torch_dtype=torch_dtype,
        )
        model = AutoModelForCausalLM.from_config(
            model_config,
            # dtype=torch_dtype,
            attn_implementation=model_config_dict.get("attn_implementation", "eager"),
        )

    elif model_name.startswith("RylanSchaeffer/scale_mem_Qwen3"):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
        )
    else:
        raise ValueError(model_config_dict["model_name"])

    return model


def load_automodelforcausallm(
    model_config_dict: Dict[str, Any]
) -> AutoModelForCausalLM:
    if model_config_dict["torch_dtype"] == "bfloat16":
        torch_dtype = torch.bfloat16
    elif model_config_dict["torch_dtype"] == "float16":
        torch_dtype = torch.float16
    elif model_config_dict["torch_dtype"] == "float32":
        torch_dtype = torch.float32
    else:
        raise NotImplementedError

    model_kwargs = {
        # Get attn_implementation from your config, defaulting to "eager".
        "attn_implementation": model_config_dict.get("attn_implementation", "eager"),
        "device_map": "auto",
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }

    if "gemma" in model_config_dict["initial_model_name_or_path"]:
        assert model_kwargs["torch_dtype"] == torch.bfloat16
        # assert model_kwargs["attn_implementation"] == "eager"

    model = AutoModelForCausalLM.from_pretrained(
        model_config_dict["initial_model_name_or_path"],
        **model_kwargs,
    )

    return model
