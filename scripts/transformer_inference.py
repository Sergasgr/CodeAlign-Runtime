from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoTokenizer

from scripts.engine import CodeAlignEngine
from scripts.inference_config import BENCH_ITERATIONS, CONTEXT_LENGTHS, MAX_SEQ_LEN, MODEL, QUANTIZE_LM_HEAD, WARMUP_ITERATIONS
from scripts.results_io import percentiles, save_results
from scripts.roofline import block_bytes_int4, decode_ceiling_ms, kv_cache_bytes, model_bytes_int4

E2E_PROMPT = '''import json
from dataclasses import dataclass


@dataclass
class User:
    id: int
    name: str
    email: str


def load_users(path: str) -> list[User]:
    """Load users from a JSON file."""
'''

def time_steps(step, prepare, iterations: int, warmup: int) -> list[float]:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for i in range(iterations + warmup):
        prepare()
        torch.cuda.synchronize()
        start.record()
        step()
        end.record()
        end.synchronize()
        if i >= warmup:
            samples.append(start.elapsed_time(end))
    return samples

def bench_block(engine: CodeAlignEngine, contexts, iterations, warmup) -> list[dict]:
    d = engine.dims
    block = engine.blocks[0]
    h0 = torch.randn(1, d.hidden, device=engine.device, dtype=torch.float32)
    h = torch.empty_like(h0)
    rows = []
    for S in contexts:
        def prepare():
            h.copy_(h0)
            block.set_seq_len(S - 1)

        stats = percentiles(time_steps(lambda: block.forward(h), prepare, iterations, warmup))
        ceiling = decode_ceiling_ms(d, S, scope="block")
        stats.update(context=S, ceiling_ms=ceiling, pct_of_ceiling_p50=100 * ceiling / stats["p50_ms"], bytes=block_bytes_int4(d) + kv_cache_bytes(d, S))
        rows.append(stats)
        print(f"  block  S={S:5d}  p50 {stats['p50_ms']:.4f} ms  p99 {stats['p99_ms']:.4f}  "
              f"ceiling {ceiling * 1e3:.2f} us  ({stats['pct_of_ceiling_p50']:.2f}% of ceiling)")
    block.reset_kv_cache()
    return rows

def bench_model(engine: CodeAlignEngine, contexts, iterations, warmup) -> list[dict]:
    d = engine.dims
    token = [0]
    rows = []
    for S in contexts:
        def step():
            engine.argmax(engine.forward(token, logits="last"))

        stats = percentiles(time_steps(step, lambda: engine.set_seq_len(S - 1), iterations, warmup))
        ceiling = decode_ceiling_ms(d, S, scope="model", quantize_lm_head=engine.quantize_lm_head)
        stats.update(context=S, ceiling_ms=ceiling, pct_of_ceiling_p50=100 * ceiling / stats["p50_ms"],
                     tokens_per_s_p50=1000.0 / stats["p50_ms"],
                     bytes=model_bytes_int4(d, engine.quantize_lm_head) + d.num_layers * kv_cache_bytes(d, S))
        rows.append(stats)
        print(f"  model  S={S:5d}  p50 {stats['p50_ms']:.3f} ms ({stats['tokens_per_s_p50']:.1f} tok/s)  "
              f"p99 {stats['p99_ms']:.3f}  ceiling {ceiling:.3f} ms  ({stats['pct_of_ceiling_p50']:.2f}%)")
    engine.reset()
    return rows

def bench_e2e(engine: CodeAlignEngine, new_tokens: int) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompt_ids = tokenizer(E2E_PROMPT)["input_ids"]
    eos = {tokenizer.eos_token_id}

    engine.reset()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    token = engine.argmax(engine.prefill(prompt_ids, logits="last"))[0]
    torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - t0) * 1e3

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    generated, tpot = [token], []
    while len(generated) < new_tokens and token not in eos:
        start.record()
        token = engine.argmax(engine.forward([token], logits="last"))[0]
        end.record()
        end.synchronize()
        tpot.append(start.elapsed_time(end))
        generated.append(token)
    engine.reset()

    text = tokenizer.decode(generated)
    stats = percentiles(tpot) if tpot else {}
    print(f"  e2e    prompt {len(prompt_ids)} tokens: TTFT {ttft_ms:.1f} ms (chunked prefill), "
          f"TPOT p50 {stats.get('p50_ms', float('nan')):.3f} ms over {len(tpot)} tokens")
    print("  --- generated continuation ---\n" + text[:600] + "\n  ------------------------------")
    return {"prompt_tokens": len(prompt_ids), "generated_tokens": len(generated), "ttft_ms": ttft_ms,
            "ttft_note": "chunked prefill through the decode/verify path (11 tokens per forward), not optimized",
            "tpot": stats, "sample_text": text}

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--contexts", type=int, nargs="+", default=CONTEXT_LENGTHS)
    parser.add_argument("--iterations", type=int, default=BENCH_ITERATIONS)
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERATIONS)
    parser.add_argument("--e2e-tokens", type=int, default=128)
    parser.add_argument("--skip-e2e", action="store_true")
    args = parser.parse_args()
    if max(args.contexts) > MAX_SEQ_LEN:
        parser.error(f"contexts must be <= MAX_SEQ_LEN ({MAX_SEQ_LEN})")

    print(f"Loading {MODEL} and quantizing to INT4 (lm_head INT4: {QUANTIZE_LM_HEAD})...")
    engine = CodeAlignEngine.from_pretrained(MODEL, max_seq_len=MAX_SEQ_LEN, quantize_lm_head=QUANTIZE_LM_HEAD)
    torch.cuda.synchronize()
    
    cuda_buffers = engine.cuda_buffer_bytes()
    memory = {"torch_tensors_mb": torch.cuda.memory_allocated() / 2**20, "cuda_buffers_mb": cuda_buffers / 2**20,
              "load_peak_torch_mb": torch.cuda.max_memory_allocated() / 2**20}
    memory["resident_mb"] = memory["torch_tensors_mb"] + memory["cuda_buffers_mb"]
    torch.cuda.reset_peak_memory_stats()
    print(f"Engine ready: {len(engine.blocks)} layers, max {engine.max_tokens_per_forward} tokens/forward, "
          f"resident VRAM {memory['resident_mb']:.0f} MB (tensors {memory['torch_tensors_mb']:.0f} + "
          f"C++ buffers {memory['cuda_buffers_mb']:.0f})\n")

    results = {"model": MODEL, "quantize_lm_head": QUANTIZE_LM_HEAD, "iterations": args.iterations,
               "warmup": args.warmup, "max_seq_len": MAX_SEQ_LEN}
    print("=== Single decoder block (layer 0) ===")
    results["block"] = bench_block(engine, args.contexts, args.iterations, args.warmup)
    print(f"\n=== Full model ({len(engine.blocks)} layers + final norm + lm_head + argmax) ===")
    results["model_decode"] = bench_model(engine, args.contexts, args.iterations, args.warmup)
    if not args.skip_e2e:
        print("\n=== End-to-end greedy generation ===")
        results["e2e"] = bench_e2e(engine, args.e2e_tokens)
    memory["decode_peak_mb"] = torch.cuda.max_memory_allocated() / 2**20 + memory["cuda_buffers_mb"]
    results["memory"] = memory
    save_results("level5_engine", results)

if __name__ == "__main__":
    main()