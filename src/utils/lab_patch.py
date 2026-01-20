from __future__ import annotations

import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from utils import get_logger

logger = get_logger("lab_patch")


def _get_model_path(model_id: str) -> Path:
    """Resolve model path from HuggingFace hub cache."""
    try:
        path = snapshot_download(
            repo_id=model_id, allow_patterns=["*.safetensors", "*.json"]
        )
        return Path(path)
    except Exception as e:
        logger.error(f"Failed to locate model files for {model_id}: {e}")
        raise


def _find_weight_file(model_path: Path, param_name: str) -> Path:
    """Find which safetensors file contains the specific parameter."""
    index_file = model_path / "model.safetensors.index.json"

    if index_file.exists():
        with open(index_file, encoding="utf-8") as f:
            index = json.load(f)

        weight_map = index.get("weight_map", {})
        if param_name in weight_map:
            return model_path / weight_map[param_name]

        msg = f"Parameter {param_name} not found in weight map."
        raise ValueError(msg)

    single_file = model_path / "model.safetensors"
    if single_file.exists():
        return single_file

    msg = f"Could not find model weights in {model_path}"
    raise FileNotFoundError(msg)


def _load_tensor_from_file(file_path: Path, tensor_name: str) -> torch.Tensor:
    """Load a single tensor from safetensors file via memory mapping."""
    with safe_open(file_path, framework="pt", device="cpu") as f:
        return f.get_tensor(tensor_name)


def _replace_module(
    model: torch.nn.Module, module_name: str, new_module: torch.nn.Module
) -> None:
    """Replace a submodule in the model tree by name."""
    parts = module_name.split(".")
    parent = model

    # Navigate to the parent
    for name in parts[:-1]:
        parent = getattr(parent, name)

    # Replace the child
    setattr(parent, parts[-1], new_module)


def apply_patch(
    model: torch.nn.Module,
    model_id: str,
    target_layers: list[str],
) -> torch.nn.Module:
    """Replace specified 4-bit layers with original FP16 layers.

    Args:
        model: The loaded 4-bit model.
        model_id: HuggingFace model ID.
        target_layers: List of module names (e.g., ["model.layers.4.mlp.down_proj"]).

    """
    if not target_layers:
        logger.info("No target layers specified for surgery.")
        return model

    model_root = _get_model_path(model_id)

    for layer_name in target_layers:
        try:
            # 1. Identify Weight/Bias names
            weight_name = f"{layer_name}.weight"
            bias_name = f"{layer_name}.bias"

            # 2. Locate and Load Weights
            weight_file = _find_weight_file(model_root, weight_name)
            weight_tensor = _load_tensor_from_file(weight_file, weight_name)

            bias_tensor = None
            try:
                with safe_open(weight_file, framework="pt", device="cpu") as f:
                    if bias_name in f.keys():  # noqa: SIM118
                        bias_tensor = f.get_tensor(bias_name)
            except Exception as e:
                logger.error(e)

            out_features, in_features = weight_tensor.shape
            new_layer = torch.nn.Linear(
                in_features,
                out_features,
                bias=(bias_tensor is not None),
                dtype=torch.float16,
            )

            new_layer.weight = torch.nn.Parameter(weight_tensor.to(dtype=torch.float16))
            if bias_tensor is not None:
                new_layer.bias = torch.nn.Parameter(bias_tensor.to(dtype=torch.float16))

            new_layer.to(model.device)

            _replace_module(model, layer_name, new_layer)
            logger.info(f"Applied patch on {layer_name}.")

        except Exception as e:
            logger.warning(f"Surgery failed on {layer_name}: {e}")
            continue

    return model
