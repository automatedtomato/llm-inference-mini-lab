from __future__ import annotations

import gc
from typing import Any

import torch
from tqdm import tqdm

from src.quantizer.models import CustomQuantLinear
from src.quantizer.utils import get_transformer_block
from src.utils import get_logger

logger = get_logger("quantizer")


def _replace_submodule(
    root_module: torch.nn.Module, sub_name: str, new_module: torch.nn.Module
) -> None:
    parts = sub_name.split(".")
    parent = root_module
    for name in parts[:-1]:
        parent = getattr(parent, name)
    setattr(parent, parts[-1], new_module)


def quantize_linear(model: torch.nn.Module) -> torch.nn.Module:
    """Replace Linear module to custom quantized linear module."""
    layers = get_transformer_block(model)
    for layer in layers:
        targets: list[tuple[str, torch.nn.Module | Any]] = []
        for name, mod in layer.named_modules():
            if isinstance(mod, torch.nn.Linear):
                targets.append((name, mod))

        for name, org_linear in targets:
            quantized_linear = CustomQuantLinear(org_linear)
            quantized_linear.to(org_linear.weight.device)
            _replace_submodule(layer, name, quantized_linear)
            logger.info(f"Replace `{name}` with CustomQuantLinear")

            del org_linear
        torch.cuda.empty_cache()
        gc.collect()
    return model
