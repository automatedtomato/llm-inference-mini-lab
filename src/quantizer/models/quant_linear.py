from __future__ import annotations

import torch


class CustomQuantLinear(torch.nn.Module):
    """Custom quantized linear module (int8)."""

    def __init__(self, org_linear: torch.nn.Module) -> None:
        super().__init__()

        assert isinstance(org_linear, torch.nn.Linear)

        self.in_features = org_linear.in_features
        self.out_features = org_linear.out_features
        org_weight = org_linear.weight.data

        w_q, scale, zp = self.quantize(org_weight)
        self.weight = torch.nn.Parameter(w_q, requires_grad=False)
        self.scale = torch.nn.Parameter(scale, requires_grad=False)
        self.zp = torch.nn.Parameter(zp, requires_grad=False)

        if org_linear.bias is not None:
            self.bias = torch.nn.Parameter(org_linear.bias.data.clone())
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        dtype = x.dtype
        w_rec = (self.weight.to(dtype) - self.zp.to(dtype)) * self.scale.to(dtype)
        return torch.nn.functional.linear(x, w_rec, self.bias)

    def quantize(
        self, org_weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantization pass."""
        w_min = org_weight.min(dim=1, keepdim=True).values
        w_max = org_weight.max(dim=1, keepdim=True).values

        scale = (w_max - w_min).clamp(min=1e-5) / 255.0
        scale = scale.to(torch.float16)

        zp = -128 - (w_min / scale)
        zp = torch.round(zp).clamp(-128, 127).to(torch.int8)

        w_q = torch.round(org_weight / scale + zp)
        w_q = torch.clamp(w_q, -128, 127).to(torch.int8)
        return w_q, scale, zp
