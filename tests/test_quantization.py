from __future__ import annotations

import torch
import torch.nn as nn

import scripts.quantization as quantization
from scripts.quantization import QuantizedLinearINT4, dequantize_int4, quantize_to_int4, replace_linear_layers
from scripts.roofline import QWEN25_CODER_05B, block_bytes_int4, ceiling_ms, model_bytes_int4

def test_packing_layout_is_the_documented_one():
    w = torch.zeros(1, 128)
    w[0, :8] = torch.tensor([7, -7, 1, -1, 0, 3, -3, 7], dtype=torch.float32)
    packed, scales = quantize_to_int4(w)
    assert packed.shape == (1, 16) and scales.shape == (1, 1)
    assert scales.item() == 1.0                         
    word = packed[0, 0].item() & 0xFFFFFFFF
    nibbles = [(word >> (4 * k)) & 0xF for k in range(8)]
    assert nibbles == [7, 9, 1, 15, 0, 3, 13, 7]       

def test_roundtrip_error_is_at_most_half_a_step():
    torch.manual_seed(0)
    w = torch.randn(64, 896)
    packed, scales = quantize_to_int4(w)
    deq = dequantize_int4(packed, scales)
    step = scales.repeat_interleave(128, dim=1)
    assert ((w - deq).abs() <= step / 2 + 1e-6).all()
    p2, s2 = quantize_to_int4(deq)
    assert torch.equal(p2, packed) and torch.allclose(s2, scales)

def test_scale_is_computed_in_fp32_for_bf16_weights():
    torch.manual_seed(0)
    w = torch.randn(8, 256).to(torch.bfloat16)
    _, scales = quantize_to_int4(w)
    expected = w.float().view(-1, 128).abs().max(dim=1).values.view(8, 2) / 7.0
    assert scales.dtype == torch.float32
    assert torch.equal(scales, expected)

class _FakeKernels:
    def __init__(self):
        self.calls = []

    def gemv_int4_forward(self, q, s, vec):
        self.calls.append("gemv")
        assert vec.dim() == 1 and vec.dtype == torch.float32
        return dequantize_int4(q, s) @ vec

    def gemm_int4_forward(self, a, q, s):
        self.calls.append("gemm")
        assert a.dim() == 2 and a.dtype == torch.float32
        return a @ dequantize_int4(q, s).T

def test_quantized_linear_dispatch_and_shapes(monkeypatch):
    fake = _FakeKernels()
    monkeypatch.setattr(quantization, "_KERNELS", fake)
    monkeypatch.setattr(quantization, "_GEMM", fake)
    torch.manual_seed(0)
    linear = nn.Linear(256, 64, bias=True)
    qlinear = QuantizedLinearINT4.from_linear(linear)
    w = dequantize_int4(qlinear.q_weight, qlinear.scales)

    x1 = torch.randn(1, 1, 256, dtype=torch.bfloat16)      # decode
    x5 = torch.randn(2, 5, 256, dtype=torch.bfloat16)      # prefill / batch
    out1, out5 = qlinear(x1), qlinear(x5)
    assert fake.calls == ["gemv", "gemm"]
    assert out1.shape == (1, 1, 64) and out5.shape == (2, 5, 64) and out5.dtype == torch.bfloat16
    torch.testing.assert_close(out5.float(), x5.float() @ w.T + linear.bias, rtol=2e-2, atol=2e-2)
    assert {"q_weight", "scales", "bias"} <= set(dict(qlinear.named_buffers()))

def test_replace_linear_layers_respects_skip():
    model = nn.Sequential()
    model.add_module("body", nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 256)))
    model.add_module("lm_head", nn.Linear(256, 512, bias=False))
    assert replace_linear_layers(model, skip=("lm_head",)) == 2
    assert isinstance(model.body[0], QuantizedLinearINT4) and isinstance(model.lm_head, nn.Linear)

def test_roofline_numbers():
    d = QWEN25_CODER_05B
    assert abs(block_bytes_int4(d) / 1e6 - 7.93) < 0.01        
    assert abs(ceiling_ms(block_bytes_int4(d)) * 1e3 - 8.85) < 0.05 
    assert 0.28 < ceiling_ms(model_bytes_int4(d, quantize_lm_head=True)) < 0.31
    assert 0.50 < ceiling_ms(model_bytes_int4(d, quantize_lm_head=False)) < 0.54