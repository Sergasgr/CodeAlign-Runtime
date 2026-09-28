from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from scripts.quantization import QuantizedLinearINT4, dequantize_int4, quantize_to_int4

pytestmark = pytest.mark.gpu

def _int4_case(rows, cols, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(rows, cols, generator=g).to(device)
    q, s = quantize_to_int4(w)
    return q, s, dequantize_int4(q, s)

@pytest.mark.parametrize("rows,cols", [(4864, 896), (896, 4864), (896, 896), (128, 896)])
def test_gemv_int4_matches_dequantized_reference(rows, cols, device):
    import codealign_runtime_kernels as kernels

    q, s, w = _int4_case(rows, cols, device)
    x = torch.randn(cols, device=device)
    out = kernels.gemv_int4_forward(q, s, x)
    ref = (w.double() @ x.double()).float()
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-3)

@pytest.mark.parametrize("m", [2, 11, 37])
def test_gemm_int4_matches_dequantized_reference(m, device):
    import codealign_runtime_gemm as gemm

    q, s, w = _int4_case(4864, 896, device, seed=1)
    a = torch.randn(m, 896, device=device)
    out = gemm.gemm_int4_forward(a, q, s)
    torch.testing.assert_close(out, (a.double() @ w.double().T).float(), rtol=1e-4, atol=1e-3)

@pytest.mark.parametrize("seq_len", [1, 5])
def test_quantized_linear_is_a_drop_in_replacement(seq_len, device):
    torch.manual_seed(0)
    linear = nn.Linear(896, 128, bias=True).to(device)
    qlinear = QuantizedLinearINT4.from_linear(linear).to(device)
    x = torch.randn(1, seq_len, 896, device=device, dtype=torch.bfloat16)
    ref = x.float() @ dequantize_int4(qlinear.q_weight, qlinear.scales).T + linear.bias.float()
    out = qlinear(x)
    assert out.dtype == torch.bfloat16 and out.shape == (1, seq_len, 128)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)

def _attention(Q, K, V):
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(Q.shape[-1])
    return torch.softmax(scores, dim=-1) @ V

@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("seq_len", [1, 255, 257, 3000])
def test_flash_decoding_matches_softmax_attention(head_dim, seq_len, device):
    import codealign_runtime_kernels as kernels

    torch.manual_seed(seq_len)
    Q = torch.randn(1, 1, head_dim, device=device)
    K = torch.randn(1, seq_len, head_dim, device=device)
    V = torch.randn(1, seq_len, head_dim, device=device)
    K[0, -1] = Q[0, 0] * 4.0
    torch.testing.assert_close(kernels.flash_decoding_forward(Q, K, V), _attention(Q, K, V), rtol=1e-4, atol=1e-4)

def test_fast_argmax_matches_torch(device):
    import codealign_runtime_transformer as engine

    torch.manual_seed(0)
    logits = torch.randn(11, 151936, device=device)
    assert torch.equal(engine.fast_argmax(logits).long(), logits.argmax(-1))

def test_find_candidate_draft():
    import codealign_runtime_transformer as engine

    assert engine.find_candidate_draft([1, 2, 3, 9, 8, 7, 1, 2, 3], 3, 5) == [9, 8, 7, 1, 2]
    assert engine.find_candidate_draft([5, 1, 2, 3, 4, 1, 2, 3], 3, 5) == [4, 1, 2, 3]  
    assert engine.find_candidate_draft([1, 2, 3, 4], 3, 5) == []                       
    assert engine.find_candidate_draft([1, 2], 3, 5) == []                           
    assert engine.find_candidate_draft([1, 2, 1, 2], 2, 0) == []