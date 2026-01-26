from __future__ import annotations

import os
import time

import pytest
import torch

from src.quantizer.models import CustomQuantLinear
from src.quantizer.passes import quantize_linear

BATCH_SIZE = 1
COS_SIM_THRESHOLD = 0.99

seeds = [int(os.getenv("SEED", time.time_ns()))]

in_feats = [
    512,
    1024,
]

out_feats = [
    200,
    64,
]

bias = [
    True,
    False,
]

model_archs = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "google/gemma-2-2b-it",
    "Qwen/Qwen2.5-3B-Instruct",
    # "microsoft/Phi-3.5-mini-instruct",
]


class DummyAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "self_attn": torch.nn.ModuleDict(
                            {"proj": torch.nn.Linear(1024, 64)}
                        ),
                        "mlp": torch.nn.Linear(64, 16),
                    }
                )
                for _ in range(3)
            ]
        )


@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("in_features", in_feats)
@pytest.mark.parametrize("out_features", out_feats)
@pytest.mark.parametrize("bias", bias)
def test_quant_unit_linear(
    seed: int,
    in_features: int,
    out_features: int,
    bias: bool,  # noqa: FBT001
) -> None:
    torch.manual_seed(seed)

    linear_float = torch.nn.Linear(in_features, out_features, bias=bias)
    linear_quant = CustomQuantLinear(linear_float)

    assert linear_quant.weight.dtype == torch.uint8
    assert linear_quant.scale.dtype == torch.float16
    assert linear_quant.zp.dtype == torch.float16

    x = torch.randn(BATCH_SIZE, in_features)
    out_float = linear_float(x)
    out_quant = linear_quant(x)
    out_float = out_float.flatten()
    out_quant = out_quant.flatten()
    cos_sim = torch.nn.functional.cosine_similarity(out_float, out_quant, dim=0)
    assert cos_sim > COS_SIM_THRESHOLD


def test_quant_model_linear() -> None:
    model = DummyAttention()
    model_quant = quantize_linear(model)
    target = model_quant.model.layers[0].self_attn.proj
    assert isinstance(target, CustomQuantLinear)
    assert target.weight.dtype == torch.uint8
