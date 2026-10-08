"""Level 4 — decode attention: numerical validation + benchmark against PyTorch (fp32, batch=1).

1. Single head (kept from v1.0 so the table stays comparable): one query against a KV-cache of S tokens.
2. Qwen2.5-0.5B attention layer (v1.1): 14 query heads over 2 KV heads, T new tokens (1 = decode step,
   6 ~ a PLD verify pass) after S - T cached positions, causal. This is what the engine runs per layer.

Both run the GQA attention kernels (src/ops/gqa_attention.cu). Validation uses standard-normal inputs and
checks that its own data would catch a kernel returning mean(V) (`discriminative`).
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

import codealign_runtime_kernels

from scripts.inference_config import HEAD_DIM, MAX_SEQ_LEN, NUM_HEADS, NUM_KV_HEADS
from scripts.results_io import percentiles, save_results

TOLERANCE = 1e-4
VALIDATION_SEQS = [1, 255, 256, 257, 1000, 1024, 2000, 2048, 4096]
SEEDS = [0, 42, 12345]
BENCH_SEQS = [256, 4096, 16384, 65536, 131072, 262144]

GQA_TOKENS = [1, 6]                    # decode step / verify pass (5.7 tokens per forward on average in v1.0)
GQA_SEQS = [128, 512, 1024, 2048]      # context S after the new tokens, as in Level 5
BANDWIDTH = 896e9                      # RTX 5070 Ti, bytes/s

class HeadAttention:
    def __init__(self, D, S, peaky_at: int | None = None):
        self.D = D
        self.S = S
        self.Q = torch.randn(1, 1, self.D, device='cuda')
        self.K = torch.randn(1, self.S, self.D, device='cuda')
        self.V = torch.randn(1, self.S, self.D, device='cuda')
        if peaky_at is not None:
            self.K[0, peaky_at] = self.Q[0, 0] * 4.0

    def attention_pytorch(self):
        scores = (self.Q @ self.K.transpose(-2, -1)) / math.sqrt(self.D)
        attn_weights = F.softmax(scores, dim=-1)
        return attn_weights @ self.V

    def attention_custom(self):
        return codealign_runtime_kernels.flash_decoding_forward(self.Q, self.K, self.V)

class GqaLayer:
    def __init__(self, D, S, T, heads=NUM_HEADS, kv_heads=NUM_KV_HEADS, max_seq=MAX_SEQ_LEN):
        self.D, self.S, self.T, self.heads, self.kv_heads = D, S, T, heads, kv_heads
        self.past_len = S - T
        self.q = torch.randn(T, heads, D, device='cuda')
        self.k_cache = torch.randn(kv_heads, max_seq, D, device='cuda')
        self.v_cache = torch.randn(kv_heads, max_seq, D, device='cuda')
        self.q_sdpa = self.q.transpose(0, 1).unsqueeze(0)             # [1, H, T, D]
        self.k_sdpa = self.k_cache[:, :S].unsqueeze(0)               # [1, KVH, S, D]
        self.v_sdpa = self.v_cache[:, :S].unsqueeze(0)
        self.mask = None
        if T > 1:
            positions = torch.arange(S, device='cuda')
            limits = self.past_len + torch.arange(T, device='cuda')
            self.mask = positions[None, :] <= limits[:, None]         

    def attention_pytorch(self):
        return F.scaled_dot_product_attention(self.q_sdpa, self.k_sdpa, self.v_sdpa, attn_mask=self.mask,
                                              enable_gqa=True)

    def attention_custom(self):
        return codealign_runtime_kernels.gqa_attention_forward(self.q, self.k_cache, self.v_cache, self.past_len)

    def reference(self):
        group = self.heads // self.kv_heads
        k = self.k_cache[:, :self.S].double().repeat_interleave(group, dim=0)
        v = self.v_cache[:, :self.S].double().repeat_interleave(group, dim=0)
        out = torch.empty(self.T, self.heads, self.D, dtype=torch.float64, device='cuda')
        for t in range(self.T):
            n = self.past_len + t + 1
            scores = torch.einsum("hd,hsd->hs", self.q[t].double(), k[:, :n]) / math.sqrt(self.D)
            out[t] = torch.einsum("hs,hsd->hd", torch.softmax(scores, dim=-1), v[:, :n])
        return out

    def kv_bytes(self):
        return 2 * self.kv_heads * self.S * self.D * 4     

def validate_correctness(D: int) -> tuple[bool, list[dict]]:
    print(f"--- NUMERICAL VALIDATION, single head (D={D}, randn inputs, tolerance {TOLERANCE:g}) ---")
    print(f"  {'case':<14} | {'S':>6} | {'max_err':>10} | {'err if kernel returned mean(V)':>30} | status")
    print("  " + "-" * 82)
    rows, all_passed = [], True

    cases = [(f"seed={seed}", seed, S, None) for seed in SEEDS for S in VALIDATION_SEQS]
    cases += [(f"peaky@{pos}", 7, S, pos) for S, pos in [(1000, 999), (4096, 4000), (4096, 3)]]
    for label, seed, S, peaky_at in cases:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        head = HeadAttention(D, S, peaky_at)
        O_pt = head.attention_pytorch()
        O_custom = head.attention_custom()
        max_err = torch.max(torch.abs(O_pt - O_custom)).item()
        mean_v_err = torch.max(torch.abs(O_pt - head.V.mean(dim=1, keepdim=True))).item()
        discriminative = S == 1 or mean_v_err > 10 * TOLERANCE
        passed = max_err < TOLERANCE and discriminative
        all_passed &= passed
        rows.append({"case": label, "S": S, "max_abs_err": max_err, "mean_v_err": mean_v_err,
                     "discriminative": discriminative, "pass": passed})
        print(f"  {label:<14} | {S:>6} | {max_err:>10.2e} | {mean_v_err:>30.2e} | {'PASS' if passed else 'FAIL'}")
    return all_passed, rows

def validate_gqa(D: int) -> tuple[bool, list[dict]]:
    print(f"\n--- NUMERICAL VALIDATION, Qwen layer ({NUM_HEADS}/{NUM_KV_HEADS} heads, D={D}, float64 reference) ---")
    print(f"  {'T':>3} | {'S':>6} | {'max_err':>10} | {'SDPA max_err':>12} | {'err if mean(V)':>14} | status")
    print("  " + "-" * 66)
    rows, all_passed = [], True
    for T in GQA_TOKENS:
        for S in GQA_SEQS:
            torch.manual_seed(S + T)
            layer = GqaLayer(D, S, T)
            ref = layer.reference()
            max_err = (layer.attention_custom().double() - ref).abs().max().item()
            sdpa_err = (layer.attention_pytorch()[0].transpose(0, 1).double() - ref).abs().max().item()
            mean_v = layer.v_cache[:, :S].double().mean(dim=1).repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=0)
            mean_v_err = (ref[-1] - mean_v).abs().max().item()
            discriminative = mean_v_err > 10 * TOLERANCE
            passed = max_err < TOLERANCE and discriminative
            all_passed &= passed
            rows.append({"T": T, "S": S, "max_abs_err": max_err, "sdpa_max_abs_err": sdpa_err,
                         "mean_v_err": mean_v_err, "discriminative": discriminative, "pass": passed})
            print(f"  {T:>3} | {S:>6} | {max_err:>10.2e} | {sdpa_err:>12.2e} | {mean_v_err:>14.2e} | "
                  f"{'PASS' if passed else 'FAIL'}")
    return all_passed, rows

def make_timer(iterations: int, warmup: int):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    def timed(fn):
        samples = []
        for i in range(iterations + warmup):
            start_event.record()
            fn()
            end_event.record()
            end_event.synchronize()
            if i >= warmup:
                samples.append(start_event.elapsed_time(end_event))
        return percentiles(samples)
    return timed

def benchmark(D: int, iterations: int, warmup: int) -> list[dict]:
    print(f"\n--- BENCHMARK, single head (D={D}, batch=1, KV-cache growth, {iterations} iterations) ---")
    print(f"  {'Seq Length':<12} | {'PyTorch p50 (ms)':<17} | {'Custom p50 (ms)':<16} | {'PyTorch/Custom':<10}")
    print("  " + "-" * 66)
    timed = make_timer(iterations, warmup)
    rows = []
    for S in BENCH_SEQS:
        head = HeadAttention(D, S)
        pt = timed(head.attention_pytorch)
        custom = timed(head.attention_custom)
        ratio = pt["p50_ms"] / custom["p50_ms"] if custom["p50_ms"] > 0 else 0.0
        kv_bytes = 2 * S * D * 4
        rows.append({"S": S, "pytorch": pt, "custom": custom, "ratio_pytorch_over_custom": ratio,
                     "kv_bytes": kv_bytes, "ceiling_ms": kv_bytes / BANDWIDTH * 1e3})
        print(f"  {S:<12} | {pt['p50_ms']:<17.4f} | {custom['p50_ms']:<16.4f} | {ratio:<10.2f}x")
    return rows

def benchmark_gqa(D: int, iterations: int, warmup: int) -> list[dict]:
    print(f"\n--- BENCHMARK, Qwen layer ({NUM_HEADS}/{NUM_KV_HEADS} heads, D={D}, {iterations} iterations) ---")
    print(f"  {'T':>3} | {'S':>6} | {'SDPA p50 (ms)':>13} | {'Custom p50 (ms)':>15} | {'SDPA/Custom':>11} | "
          f"{'KV ceiling (ms)':>15} | {'% of ceiling':>12}")
    print("  " + "-" * 96)
    timed = make_timer(iterations, warmup)
    rows = []
    for T in GQA_TOKENS:
        for S in GQA_SEQS:
            layer = GqaLayer(D, S, T)
            pt = timed(layer.attention_pytorch)
            custom = timed(layer.attention_custom)
            ratio = pt["p50_ms"] / custom["p50_ms"] if custom["p50_ms"] > 0 else 0.0
            ceiling_ms = layer.kv_bytes() / BANDWIDTH * 1e3
            pct = 100 * ceiling_ms / custom["p50_ms"] if custom["p50_ms"] > 0 else 0.0
            rows.append({"T": T, "S": S, "heads": NUM_HEADS, "kv_heads": NUM_KV_HEADS, "pytorch_sdpa": pt,
                         "custom": custom, "ratio_pytorch_over_custom": ratio, "kv_bytes": layer.kv_bytes(),
                         "ceiling_ms": ceiling_ms, "pct_of_ceiling_p50": pct})
            print(f"  {T:>3} | {S:>6} | {pt['p50_ms']:>13.4f} | {custom['p50_ms']:>15.4f} | {ratio:>10.2f}x | "
                  f"{ceiling_ms:>15.5f} | {pct:>11.1f}%")
    return rows

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head-dim", type=int, default=HEAD_DIM, help="64 = Qwen2.5-0.5B head dim (64 or 128)")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    single_passed, validation = validate_correctness(args.head_dim)
    gqa_passed, gqa_validation = validate_gqa(args.head_dim)
    all_passed = single_passed and gqa_passed
    print(f"\nResult: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    if not all_passed:
        print("Benchmark skipped — fix validation errors first.")
        save_results("level4_flash_decoding", {"head_dim": args.head_dim, "validation": validation,
                                               "gqa_validation": gqa_validation, "benchmark": None, "gqa_layer": None})
        raise SystemExit(1)
    rows = benchmark(args.head_dim, args.iterations, args.warmup)
    gqa_rows = benchmark_gqa(args.head_dim, args.iterations, args.warmup)
    save_results("level4_flash_decoding", {"head_dim": args.head_dim, "tolerance": TOLERANCE, "validation": validation,
                                           "gqa_validation": gqa_validation, "benchmark": rows, "gqa_layer": gqa_rows})

if __name__ == "__main__":
    main()