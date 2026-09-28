from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

GROUP_SIZE = 128

_KERNELS = None
_GEMM = None

def _kernels():
    global _KERNELS
    if _KERNELS is None:
        import codealign_runtime_kernels
        _KERNELS = codealign_runtime_kernels
    return _KERNELS

def _gemm():
    global _GEMM
    if _GEMM is None:
        import codealign_runtime_gemm
        _GEMM = codealign_runtime_gemm
    return _GEMM

def quantize_to_int4(weight: torch.Tensor, group_size: int = GROUP_SIZE) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = weight.shape
    if cols % group_size != 0:
        raise ValueError(f"cols ({cols}) must be a multiple of group_size ({group_size})")
    
    w_groups = weight.detach().to(torch.float32).contiguous().view(-1, group_size)
    max_vals = w_groups.abs().max(dim=1, keepdim=True)[0]
    scales = torch.clamp(max_vals, min=1e-9) / 7.0

    w_groups = torch.round(w_groups / scales).clamp(-8, 7).to(torch.int32)
    w_pack = w_groups.view(-1, 8)

    packed = torch.zeros(w_pack.shape[0], dtype=torch.int32, device=weight.device)
    for i in range(8):
        packed = packed | ((w_pack[:, i] & 0xF) << (4 * i))

    packed = packed.view(rows, cols // 8)
    scales = scales.view(rows, cols // group_size)

    return packed, scales

def dequantize_int4(packed: torch.Tensor, scales: torch.Tensor, group_size: int = GROUP_SIZE) -> torch.Tensor:
    rows = packed.shape[0]
    shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int32)
    q = (packed.unsqueeze(-1) >> shifts) & 0xF                  # [rows, cols/8, 8]
    q = torch.where(q > 7, q - 16, q).reshape(rows, -1).to(torch.float32)
    return q * scales.to(torch.float32).repeat_interleave(group_size, dim=1)

class QuantizedLinearINT4(nn.Module):
    def __init__(self, q_weight: torch.Tensor, scales: torch.Tensor, bias: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("q_weight", q_weight.contiguous())
        self.register_buffer("scales", scales.to(torch.float32).contiguous())
        self.register_buffer("bias", None if bias is None else bias.detach().to(torch.float32).contiguous())
        self.out_features = q_weight.shape[0]
        self.in_features = q_weight.shape[1] * 8

    @classmethod
    def from_weight(cls, weight: torch.Tensor, bias: torch.Tensor | None = None) -> "QuantizedLinearINT4":
        q_weight, scales = quantize_to_int4(weight)
        return cls(q_weight, scales, bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "QuantizedLinearINT4":
        return cls.from_weight(linear.weight, linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected last dim {self.in_features}, got {tuple(x.shape)}")
        x_2d = x.reshape(-1, self.in_features).to(torch.float32).contiguous()
        if x_2d.shape[0] == 1:
            out = _kernels().gemv_int4_forward(self.q_weight, self.scales, x_2d[0]).unsqueeze(0)
        else:
            out = _gemm().gemm_int4_forward(x_2d, self.q_weight, self.scales)
        if self.bias is not None:
            out = out + self.bias
        return out.to(x.dtype).view(*x.shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, int4 g={GROUP_SIZE}"

def replace_linear_layers(module: nn.Module, skip: Iterable[str] = (), _prefix: str = "") -> int:
    skip = set(skip)
    replaced = 0
    for name, child in module.named_children():
        qualified = f"{_prefix}{name}"
        if isinstance(child, nn.Linear):
            if qualified in skip:
                continue
            setattr(module, name, QuantizedLinearINT4.from_linear(child))
            replaced += 1
        else:
            replaced += replace_linear_layers(child, skip, f"{qualified}.")
    return replaced