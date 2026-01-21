from __future__ import annotations

import gc
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from . import get_logger

GPU_LIMIT = 4096


logger = get_logger("utils")


def load_model_and_tokenizer(  # noqa: D417
    model_name: str,
    dtype: torch.dtype = torch.float16,
    quantization_config: dict[str, Any] | None = None,
    **kwargs,  # noqa: ANN003, ARG001
) -> tuple[Any, Any]:
    """Load model and tokenizer from model name.

    Args:
        model_name (str): model name to be loaded.
        dtype (torch.dtype): dtype in which model is loaded.
        quantization_config (dict[str, Any]): configuration for quantization.

    Returns:
        Tuple of model and tokenizer.

    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    bnb_config = None
    if quantization_config:
        if quantization_config.get("method") == "bitsandbytes":
            quantization_config["bnb_4bit_compute_type"] = getattr(
                torch, str(quantization_config.get("bnb_4bit_compute_dtype"))
            )
            logger.info(f"Configuring bitsandbytes quantization. {quantization_config}")
            bnb_config = BitsAndBytesConfig(
                **quantization_config,
                bnb_4bit_use_double_quant=True,
            )
        else:
            raise NotImplementedError

    logger.info(f"Model: {model_name}. Quantized={bnb_config is not None}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, quantization_config=bnb_config, device_map="auto"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def prepare_dataset(split: str = "test", length: int = -1) -> str:
    """Load WikiText-2 dataset and concat into unified string."""
    print("Loading WikiText-2 dataset.")
    dataset = load_dataset(path="wikitext", name="wikitext-2-raw-v1", split=split)
    # Concat each text with two line breaks so to evaluate while persist log context
    text = "\n\n".join(dataset["text"])

    if length > 0:
        text = text[:length]

    print(f"{len(text)=}")

    return text


def cleanup_gpu() -> None:
    """Clean up GPU."""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()


def get_transformer_block(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Find transformer blocks from model architecture."""
    # Llama, Gemma, Qwen, Phi
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    else:  # noqa: RET505
        # NOTE: Other structure is currenctly not supported.
        raise NotImplementedError


def get_target_layers(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    """Get all of specified module from transformer block.

    Returns:
        list of (full module name, module instance)

    """
    target_layers = []

    layers = get_transformer_block(model)
    for idx, layer in enumerate(layers):
        for name, module in layer.named_modules():
            if isinstance(module, torch.nn.Linear):
                full_name = f"layers.{idx}.{name}"
                target_layers.append((full_name, module))
    return target_layers
