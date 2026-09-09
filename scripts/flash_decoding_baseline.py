import torch
import torch.nn.functional as F
import math

import codealign_runtime_kernels

D = 128   # Head dimension (Qwen2.5-0.5B uses 128-dim heads)
S = 1024  # Default sequence length (KV-cache tokens)

class HeadAttention:
    def __init__(self, D, S):
        self.D = D
        self.S = S
        self.Q = torch.rand(1, 1, self.D, device='cuda')
        self.K = torch.rand(1, self.S, self.D, device='cuda')
        self.V = torch.rand(1, self.S, self.D, device='cuda')

    def attention_pytorch(self):
        scores = (self.Q @ self.K.transpose(-2, -1)) / math.sqrt(self.D)
        attn_weights = F.softmax(scores, dim=-1)
        return attn_weights @ self.V

    def attention_custom(self):
        return codealign_runtime_kernels.flash_decoding_forward(self.Q, self.K, self.V)

def validate_correctness():
    print("--- NUMERICAL VALIDATION (multi-seed, multi-sequence) ---")
    print(f"  {'seed':<6} | {'S':>6} | {'max_err':>10} | {'status'}")
    print("  " + "-" * 42)

    sequences = [256, 1000, 1024, 2000, 2048, 4096]
    seeds = [0, 42, 12345]

    all_passed = True
    for seed in seeds:
        for S in sequences:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            head = HeadAttention(D, S)
            O_pt = head.attention_pytorch()
            O_custom = head.attention_custom()
            max_err = torch.max(torch.abs(O_pt - O_custom)).item()
            passed = max_err < 1e-2
            if not passed:
                all_passed = False
            print(f"  {seed:<6} | {S:>6} | {max_err:>10.6f} | {'PASS' if passed else 'FAIL'}")

    return all_passed

def benchmark():
    print("\n--- BENCHMARK (Batch=1, KV-Cache Growth) ---")
    print(f"  {'Seq Length':<12} | {'PyTorch (ms)':<15} | {'Custom (ms)':<15} | {'Speedup':<10}")
    print("  " + "-" * 58)

    sequences = [256, 4096, 16384, 65536, 131072, 262144]
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for S in sequences:
        head = HeadAttention(D, S)

        # Warmup
        for _ in range(10):
            head.attention_pytorch()
            head.attention_custom()
        torch.cuda.synchronize()

        # PyTorch
        start_event.record()
        for _ in range(100):
            head.attention_pytorch()
        end_event.record()
        torch.cuda.synchronize()
        pt_ms = start_event.elapsed_time(end_event) / 100.0

        # Custom Flash-Decoding
        start_event.record()
        for _ in range(100):
            head.attention_custom()
        end_event.record()
        torch.cuda.synchronize()
        custom_ms = start_event.elapsed_time(end_event) / 100.0

        speedup = pt_ms / custom_ms if custom_ms > 0 else 0
        print(f"  {S:<12} | {pt_ms:<15.4f} | {custom_ms:<15.4f} | {speedup:<10.2f}x")

if __name__ == "__main__":
    all_passed = validate_correctness()
    print(f"\nResult: {'ALL PASSED' if all_passed else 'SOME FAILED'}")

    if all_passed:
        benchmark()
    else:
        print("Benchmark skipped — fix validation errors first.")