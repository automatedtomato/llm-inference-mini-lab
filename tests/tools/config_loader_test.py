from pathlib import Path

import pytest
import torch
import yaml

from src.utils.config_loader import load_config

dtypes = ["float16", "bfloat16", "float32", "int8"]


@pytest.mark.parametrize("dtype", dtypes)
def test_load_config_dtype_conversion(tmp_path: Path, dtype: str) -> None:
    dummy_config = {
        "experiment_name": "test_exp",
        "model": {"name": "dummy-model", "dtype": dtype},
        "benchmark": {},
        "prompt": {},
        "output": {},
    }

    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(dummy_config, f)

    loaded_config = load_config(str(config_file))

    assert loaded_config["model"]["dtype"] == getattr(torch, dtype)
