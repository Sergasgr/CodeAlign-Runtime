from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from scripts.quantization import dequantize_int4, quantize_to_int4

pytestmark = pytest.mark.gpu

PROMPT_LEN = 29    
NOISE_FACTOR = 10.0 
REL_FLOOR = 1e-4      

def int4_reference(hf_model, dtype: torch.dtype, device) -> nn.Module:
    ref = copy.deepcopy(hf_model).to(device=device, dtype=dtype).eval()
    for module in ref.modules():
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                w = dequantize_int4(*quantize_to_int4(child.weight.data.float())).to(dtype)
                if name == "lm_head":
                    child.weight = nn.Parameter(w)  
                else:
                    child.weight.data.copy_(w)
    return ref


def references(hf_model, device):
    return int4_reference(hf_model, torch.float32, device), int4_reference(hf_model, torch.float64, device)

def make_engine(hf_model, device):
    from scripts.engine import CodeAlignEngine

    return CodeAlignEngine(hf_model, max_seq_len=256, quantize_lm_head=True, device=device)

def prompt_ids(vocab: int, n: int = PROMPT_LEN, seed: int = 1) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (n,), generator=g).tolist()

def rel_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    return ((out.double() - ref.double()).abs().max() / ref.double().abs().max()).item()

def assert_within_fp32_noise(name: str, got: torch.Tensor, ref32: torch.Tensor, ref64: torch.Tensor) -> None:
    noise = rel_err(ref32, ref64)
    err = rel_err(got, ref64)
    bound = max(NOISE_FACTOR * noise, REL_FLOOR)
    assert err <= bound, (f"{name}: engine error {err:.2e} vs HF fp32 error {noise:.2e} "
                          f"(both relative to HF float64; bound {bound:.2e})")

def top2_margin(logits: torch.Tensor) -> torch.Tensor:
    top = logits.topk(2, dim=-1).values
    return top[..., 0] - top[..., 1]

def layer0_output(model, ids, device) -> torch.Tensor:
    captured = {}
    hook = model.model.layers[0].register_forward_hook(
        lambda mod, inp, out: captured.setdefault("out", out[0] if isinstance(out, tuple) else out))
    with torch.no_grad():
        model(torch.tensor([ids], device=device))
    hook.remove()
    return captured["out"][0] 

def test_block_matches_hf_decoder_layer(hf_model, device):
    ref32, ref64 = references(hf_model, device)
    engine = make_engine(hf_model, device)
    ids = prompt_ids(hf_model.config.vocab_size)

    block = engine.blocks[0]
    h = engine.embed(ids)
    outs = []
    for start in range(0, len(ids), engine.max_tokens_per_forward):
        chunk = h[start:start + engine.max_tokens_per_forward].contiguous()
        block.forward(chunk)
        outs.append(chunk)
    got = torch.cat(outs)
    assert block.seq_len() == len(ids)
    assert_within_fp32_noise("layer 0 output", got, layer0_output(ref32, ids, device), layer0_output(ref64, ids, device))

def test_model_logits_match_hf(hf_model, device):
    ref32, ref64 = references(hf_model, device)
    engine = make_engine(hf_model, device)
    ids = prompt_ids(hf_model.config.vocab_size)
    with torch.no_grad():
        l32 = ref32(torch.tensor([ids], device=device)).logits[0]
        l64 = ref64(torch.tensor([ids], device=device)).logits[0]

    engine.reset()
    chunks = [engine.forward(ids[s:s + engine.max_tokens_per_forward], logits="all")
              for s in range(0, len(ids), engine.max_tokens_per_forward)]
    got = torch.cat(chunks)
    assert got.shape == l64.shape
    assert_within_fp32_noise("logits", got, l32, l64)
    noise_abs = (l32.double() - l64).abs().max()
    clear = top2_margin(l64) > NOISE_FACTOR * noise_abs
    assert torch.equal(got.argmax(-1)[clear], l64.argmax(-1)[clear])

def test_greedy_generation_matches_hf(hf_model, device):
    ref32, ref64 = references(hf_model, device)
    engine = make_engine(hf_model, device)
    ids = prompt_ids(hf_model.config.vocab_size, n=17, seed=3)
    new = 24

    got = engine.generate(ids, max_new_tokens=new)         
    assert len(got) == new
    seq = torch.tensor([ids + got[:-1]], device=device)
    with torch.no_grad():
        l32 = ref32(seq).logits[0, len(ids) - 1:]             
        l64 = ref64(seq).logits[0, len(ids) - 1:]
    for i, token in enumerate(got):
        best = int(l64[i].argmax())
        if token != best:
            gap = (l64[i, best] - l64[i, token]).item()
            noise_abs = (l32[i].double() - l64[i]).abs().max().item()
            assert gap <= NOISE_FACTOR * noise_abs, (
                f"token {i}: engine picked {token}, HF picks {best} with a margin of {gap:.3e} "
                f"(HF fp32 noise at this position: {noise_abs:.3e})")

def test_speculative_decoding_is_lossless(hf_model, device):
    from scripts.generate_speculative import SpeculativeDecoder

    engine = make_engine(hf_model, device)
    decoder = SpeculativeDecoder(engine, ngram=2, max_draft=5)
    base = prompt_ids(hf_model.config.vocab_size, n=12, seed=5)
    ids = base + base + base[:6]  
    new = 40

    greedy, greedy_stats = decoder.generate(ids, new, speculative=False)
    pld, pld_stats = decoder.generate(ids, new, speculative=True)
    assert greedy_stats.drafted == 0 and greedy_stats.forwards == new
    assert pld_stats.drafted > 0
    assert len(pld) == new
    assert engine.seq_len == len(ids) + len(pld) - 1
    assert pld == greedy or _diverges_at_near_tie(engine, ids, greedy, pld)

def _diverges_at_near_tie(engine, prompt, a, b) -> bool:
    i = next(k for k, (x, y) in enumerate(zip(a, b)) if x != y)
    engine.reset()
    logits = engine.prefill(prompt + a[:i], logits="last")[0]
    return bool(top2_margin(logits) < 1e-3 * logits.abs().max())

def test_block_bounds_are_enforced(tiny_qwen, device):
    engine = make_engine(tiny_qwen, device)
    block = engine.blocks[0]
    hidden = tiny_qwen.config.hidden_size
    with pytest.raises(RuntimeError):
        block.forward(torch.zeros(engine.max_tokens_per_forward + 1, hidden, device=device))
    with pytest.raises(RuntimeError):
        block.forward(torch.zeros(hidden, device=device)) 
    block.set_seq_len(block.max_seq_len() - 1)
    with pytest.raises(RuntimeError):
        block.forward(torch.zeros(2, hidden, device=device)) 
    block.reset_kv_cache()
    with pytest.raises(RuntimeError):
        block.rollback_kv_cache(1)