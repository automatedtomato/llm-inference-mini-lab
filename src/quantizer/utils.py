from __future__ import annotations

from typing import Any

import torch


def get_transformer_block(model: torch.nn.Module) -> Any:  # noqa: ANN401
    """Find transformer blocks from model architecture."""
    # Llama, Gemma, Qwen, Phi
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    else:  # noqa: RET505
        # NOTE: Other structure is currenctly not supported.
        raise NotImplementedError


def get_target_layers(model: torch.nn.Module) -> list[tuple[str, Any]]:
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
