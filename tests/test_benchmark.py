import subprocess

import pytest
import torch

config_paths = ["src/configs/benchmark_config.yaml"]


@pytest.mark.parametrize("config_path", config_paths)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
def benchmark_test_gpu(config_path: str) -> None:
    """Test that bechmark script is excutable."""
    cmd = [
        "python3",
        "-m",
        "src.benchmark",
        "-c",
        config_path,
    ]
    print(f"{cmd=}")
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if process.stdout is not None:
        for line in process.stdout:
            print(line, end="")
        process.wait()

    assert process.returncode == 0
