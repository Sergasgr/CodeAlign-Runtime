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

GQA_MAX_SEQ = 2048
GQA_PAST_LENS = [0, 1, 31, 32, 33, 255, 256, 257, 1000, 2037]

def _gqa_reference(q, k_cache, v_cache, past_len):
    T, H, D = q.shape
    group = H // k_cache.shape[0]
    out = torch.empty(T, H, D, dtype=torch.float64, device=q.device)
    for t in range(T):
        S = past_len + t + 1
        k = k_cache[:, :S].double().repeat_interleave(group, dim=0)   # [H, S, D]
        v = v_cache[:, :S].double().repeat_interleave(group, dim=0)
        scores = torch.einsum("hd,hsd->hs", q[t].double(), k) / math.sqrt(D)
        out[t] = torch.einsum("hs,hsd->hd", torch.softmax(scores, dim=-1), v)
    return out

def _gqa_case(num_heads, num_kv_heads, head_dim, num_tokens, past_len, device):
    g = torch.Generator().manual_seed(1000 * num_tokens + past_len)
    q = torch.randn(num_tokens, num_heads, head_dim, generator=g)
    k_cache = torch.randn(num_kv_heads, GQA_MAX_SEQ, head_dim, generator=g)
    v_cache = torch.randn(num_kv_heads, GQA_MAX_SEQ, head_dim, generator=g)
    seq_max = past_len + num_tokens
    group = num_heads // num_kv_heads
    k_cache[:, 0] += 4.0 * q[0, ::group]           
    k_cache[:, seq_max - 1] = 4.0 * q[-1, ::group]   
    k_cache[:, seq_max:] = float("nan")              
    v_cache[:, seq_max:] = float("nan")
    return q.to(device), k_cache.to(device), v_cache.to(device)

@pytest.mark.parametrize("past_len", GQA_PAST_LENS)
@pytest.mark.parametrize("num_tokens", [1, 3, 11])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(14, 2), (1, 1), (8, 1)])
def test_gqa_attention_matches_reference(num_heads, num_kv_heads, head_dim, num_tokens, past_len, device):
    import codealign_runtime_kernels as kernels

    q, k_cache, v_cache = _gqa_case(num_heads, num_kv_heads, head_dim, num_tokens, past_len, device)
    out = kernels.gqa_attention_forward(q, k_cache, v_cache, past_len, poison_scratch=True)
    ref = _gqa_reference(q, k_cache, v_cache, past_len)
    assert not torch.isnan(out).any()
    torch.testing.assert_close(out, ref.float(), rtol=1e-4, atol=1e-4)
    if past_len + num_tokens > 1:
        mean_v = v_cache[:, :past_len + num_tokens].double().mean(dim=1).repeat_interleave(num_heads // num_kv_heads, dim=0)
        assert (ref[-1] - mean_v).abs().max() > 1e-3
    assert torch.equal(out, kernels.gqa_attention_forward(q, k_cache, v_cache, past_len))

@pytest.mark.parametrize("max_chunks", [1, 2, 7])
def test_gqa_attention_respects_max_chunks(max_chunks, device):
    import codealign_runtime_kernels as kernels

    q, k_cache, v_cache = _gqa_case(14, 2, 64, 11, 1000, device)
    out = kernels.gqa_attention_forward(q, k_cache, v_cache, 1000, max_chunks=max_chunks, poison_scratch=True)
    torch.testing.assert_close(out, _gqa_reference(q, k_cache, v_cache, 1000).float(), rtol=1e-4, atol=1e-4)

@pytest.mark.parametrize("num_heads,num_kv_heads,head_dim,past_len,num_tokens", [
    (9, 1, 64, 0, 1),     
    (14, 3, 64, 0, 1),     
    (14, 2, 96, 0, 1),     
    (14, 2, 64, 2040, 9),  
])
def test_gqa_attention_rejects_unsupported_shapes(num_heads, num_kv_heads, head_dim, past_len, num_tokens, device):
    import codealign_runtime_kernels as kernels

    q = torch.randn(num_tokens, num_heads, head_dim, device=device)
    k_cache = torch.randn(num_kv_heads, GQA_MAX_SEQ, head_dim, device=device)
    with pytest.raises(RuntimeError):
        kernels.gqa_attention_forward(q, k_cache, k_cache.clone(), past_len)

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