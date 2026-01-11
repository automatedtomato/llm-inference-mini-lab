from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import yaml

if TYPE_CHECKING:
    from pathlib import Path

def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load config from config_path."""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"file not found: {config_path}")

    print(f"Config file loaded from {config_path}")
    model_config = config["model"]
    dtype_str = model_config.get("dtype", "float16")
    try:
        model_config["dtype"] = getattr(torch, dtype_str)
    except AttributeError:
        print(f"Unknown dtype: {dtype_str}. Fallback to float16.")
        model_config["dtype"] = torch.float16

    return config
