from __future__ import annotations

import torch

from src.utils import get_logger

logger = get_logger("quant_linear")

UINT4_MIN, UINT4_MAX = 0, 15


class CustomQuantLinear(torch.nn.Module):
    """Custom quantized linear module (int8)."""

    def __init__(self, org_linear: torch.nn.Module, block_size: int = 128) -> None:
        super().__init__()

        assert isinstance(org_linear, torch.nn.Linear)

        self.in_features = org_linear.in_features
        self.out_features = org_linear.out_features
        self.block_size = block_size
        if self.in_features % self.block_size != 0:
            msg = (
                f"in_features ({self.in_features}) must be devided by "
                f"block_size ({self.block_size})."
            )
            raise ValueError(
                msg
            )

        org_weight = org_linear.weight.data

        w_packed, scale, zp = self.quantize(org_weight)
        self.weight = torch.nn.Parameter(w_packed, requires_grad=False)
        self.scale = torch.nn.Parameter(scale, requires_grad=False)
        self.zp = torch.nn.Parameter(zp, requires_grad=False)

        if org_linear.bias is not None:
            self.bias = torch.nn.Parameter(org_linear.bias.data.clone())
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with unpacking and block dequantization."""
        dtype = x.dtype

        w_h, w_l = self.unpack_uint8_to_uint4()

        w_unpacked = torch.stack((w_h, w_l), dim=-1).reshape(
            self.out_features, self.in_features
        )

        w_blocked = w_unpacked.view(self.out_features, -1, self.block_size)

        w_rec_blocked = (
            w_blocked.to(dtype) - self.zp.to(dtype)
        ) * self.scale.to(dtype)

        w_rec = w_rec_blocked.view(self.out_features, self.in_features)
        return torch.nn.functional.linear(x, w_rec, self.bias)

    def quantize(
        self, org_weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensors]:
        """Block-wise quantization pass."""
        org_weight = org_weight.view(self.out_features, -1, self.block_size)

        w_min = org_weight.min(dim=1, keepdim=True).values
        w_max = org_weight.max(dim=1, keepdim=True).values

        scale = (w_max - w_min).clamp(min=1e-5) / 15.0
        scale = scale.to(torch.float16)

        zp = -(w_min / scale)
        if (zp > UINT4_MAX).any() or (zp < UINT4_MIN).any():
            logger.warning(
                "zero_points before clamp have overflowed values. "
                f"There might be unexpected result: {zp.shape=} | {zp=}"
            )
        zp = torch.round(zp).clamp(UINT4_MIN, UINT4_MAX).to(torch.float16)

        w_q = torch.round(org_weight / scale + zp)
        w_q = torch.clamp(w_q, UINT4_MIN, UINT4_MAX).to(torch.uint8)
        w_q = w_q.view(self.out_features, self.in_features)

        h_bits = w_q[:, 0::2]
        l_bits = w_q[:, 1::2]
        w_packed = (h_bits << 4) | l_bits

        return w_packed, scale, zp

    def unpack_uint8_to_uint4(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Unpack uint4 packed weight to uint8."""
        mask = 0x0F
        w_h = (self.weight >> 4) & mask
        w_l = self.weight & mask

        return w_h, w_l
