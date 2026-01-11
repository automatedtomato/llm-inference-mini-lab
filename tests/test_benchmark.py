import subprocess

import pytest

config_paths = ["src/configs/benchmark_config.yaml"]


@pytest.mark.parametrize("config_path", config_paths)
def test_benchmark(config_path: str) -> None:
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
