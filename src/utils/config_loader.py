from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from pathlib import Path


def load_config(config_path: str | Path) -> dict[str, Any] | None:
    """Load config from config_path."""
    config: dict[str, Any] = {}
    if config_path:
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"file not found: {config_path}")

        print(f"QConfig file loaded from {config_path}")

        return config
    return None
