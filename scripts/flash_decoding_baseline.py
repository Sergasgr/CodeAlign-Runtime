from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

import codealign_runtime_kernels

from scripts.inference_config import HEAD_DIM
from scripts.results_io import percentiles, save_results

TOLERANCE = 1e-4                      
VALIDATION_SEQS = [1, 255, 256, 257, 1000, 1024, 2000, 2048, 4096]
SEEDS = [0, 42, 12345]
BENCH_SEQS = [256, 4096, 16384, 65536, 131072, 262144]

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

def validate_correctness(D: int) -> tuple[bool, list[dict]]:
    print(f"--- NUMERICAL VALIDATION (D={D}, randn inputs, tolerance {TOLERANCE:g}) ---")
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

def benchmark(D: int, iterations: int, warmup: int) -> list[dict]:
    print(f"\n--- BENCHMARK (D={D}, batch=1, KV-cache growth, {iterations} iterations) ---")
    print(f"  {'Seq Length':<12} | {'PyTorch p50 (ms)':<17} | {'Custom p50 (ms)':<16} | {'PyTorch/Custom':<10}")
    print("  " + "-" * 66)

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

    rows = []
    for S in BENCH_SEQS:
        head = HeadAttention(D, S)
        pt = timed(head.attention_pytorch)
        custom = timed(head.attention_custom)
        ratio = pt["p50_ms"] / custom["p50_ms"] if custom["p50_ms"] > 0 else 0.0
        kv_bytes = 2 * S * D * 4
        rows.append({"S": S, "pytorch": pt, "custom": custom, "ratio_pytorch_over_custom": ratio,
                     "kv_bytes": kv_bytes, "ceiling_ms": kv_bytes / 896e9 * 1e3})
        print(f"  {S:<12} | {pt['p50_ms']:<17.4f} | {custom['p50_ms']:<16.4f} | {ratio:<10.2f}x")
    return rows

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head-dim", type=int, default=HEAD_DIM, help="64 = Qwen2.5-0.5B head dim")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    all_passed, validation = validate_correctness(args.head_dim)
    print(f"\nResult: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    if not all_passed:
        print("Benchmark skipped — fix validation errors first.")
        save_results("level4_flash_decoding", {"head_dim": args.head_dim, "validation": validation, "benchmark": None})
        raise SystemExit(1)
    rows = benchmark(args.head_dim, args.iterations, args.warmup)
    save_results("level4_flash_decoding", {"head_dim": args.head_dim, "tolerance": TOLERANCE, "validation": validation, "benchmark": rows})

if __name__ == "__main__":
    main()