import os
import subprocess
import time

import pytest
import torch

seeds = [int(os.getenv("SEED", time.time()))]
config_paths = ["src/configs/benchmark_config.yaml"]


@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("config_path", config_paths)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
def test_benchmark_gpu(seed: int, config_path: str) -> None:
    torch.manual_seed(seed)
    print(f"{seed=}")
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
