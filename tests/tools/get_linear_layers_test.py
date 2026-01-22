import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from src.quantizer.utils import get_target_layers

archs = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "google/gemma-2-2b-it",
    "Qwen/Qwen2.5-3B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
]


@pytest.mark.parametrize("arch", archs, ids=["llama", "gemma", "qwen", "phi"])
def test_get_linear_module_from_model(
    arch: str,
) -> None:
    config = AutoConfig.from_pretrained(arch)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    targets = get_target_layers(model)

    linears = []
    for _, mod in targets:
        linears.append(type(mod).__name__)

    assert len(targets) > 0
    assert linears[0] == "Linear"
    assert linears[1:] == linears[:-1]
