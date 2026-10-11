# CodeAlign-Runtime

**A C++/CUDA inference engine for small code models, measured against hardware ceilings.**

Hand-written CUDA kernels serve Qwen2.5-Coder-0.5B, a code completion model, at batch=1, which is the IDE autocompletion case. The project is built in levels, from a naive GEMV up to a full 24-layer INT4 engine with speculative decoding. Every latency is reported next to a **theoretical ceiling computed from the bytes that must cross the memory bus**, and compared against PyTorch and llama.cpp on the same GPU.

> **Starting hypothesis:** at batch=1, autoregressive decoding is dominated by matrix-vector products (GEMV). GEMV is memory-bound, so moving fewer bytes per token (quantization) should matter more than raw FLOPs. The measurements below test this kernel by kernel and end to end. The answer is more nuanced than the hypothesis, and the numbers show why.

> **Status:** this README describes **v1.1** (branch `perf/v1.1`): a profiling-driven performance pass over v1.0 plus quality controls for the INT4 model. The `v1.1.0` tag is pending an ONNX Runtime reference and the final re-measurement on the release commit. The v1.0 code and numbers live at the `v1.0.0` tag.

---

## Results at a Glance

NVIDIA RTX 5070 Ti (896 GB/s), Qwen2.5-Coder-0.5B, batch=1, greedy decoding. All numbers come from `results/*.json`; profiling figures come from the Nsight summaries in `results/profiles/`.

| System | TPOT p50 | tok/s | % of bandwidth ceiling | Peak VRAM | HumanEval pass@1 |
|--------|---:|---:|---:|---:|---:|
| PyTorch eager, bf16 | 14.61 ms | 68 | 7.5% | 997 MB | 24.4% |
| **This engine v1.1, INT4 (S=512)** | **1.73 ms** | **578** | **17.8%** | **609 MB** | **15.9%** |
| This engine v1.0, INT4 (S=512) | 56.09 ms | 17.8 | 0.55% | 599 MB | 14.0% |
| llama.cpp F16 (depth 512) | 2.20 ms | 454 | 50% | — | — |
| llama.cpp Q4_0 (depth 512) | 1.33 ms | 754 | 29% | — | — |
| llama.cpp Q4_K_M (depth 512) | 1.39 ms | 719 | 31% | — | — |

HumanEval for the INT4 rows is the same INT4 weights and kernels running inside Hugging Face's `generate()` ([Quantization vs. Quality](#quantization-vs-quality)).

What the data shows:

1. **The engine is correct wherever it is tested.** Every kernel matches a CPU/PyTorch reference on random data. The full INT4 engine matches Hugging Face running the same quantized weights within its own fp32 noise (tested against float64), and it generates coherent code. Speculative decoding produces exactly the greedy output on every prompt.
2. **v1.1 is 32× faster than v1.0 and 8.4× faster than PyTorch eager; llama.cpp Q4_0 is still 1.30× faster.** Three changes, each driven by the v1.0 profile and measured on its own ([What Changed in v1.1](#what-changed-in-v11)):
   - no device synchronization after every launch: 55.9 → 49.7 ms per token;
   - batched GQA attention (two launches per layer for all heads and tokens, each K/V row read once per KV head): 49.7 → 4.1 ms;
   - a new INT4 GEMV that stages the activations in shared memory: 4.1 → 1.73 ms.
3. **INT4 now turns its byte reduction into speed.** On the 151,936×896 lm_head the new INT4 GEMV reads its 72 MB at **74% of DRAM bandwidth** (L2 cold), 6.4× faster than the v1.0 INT4 kernel and **6.2× faster than the fp32 GEMV** for 7.5× fewer bytes. In v1.0 INT4 was only 1.09× faster than fp32, because the kernel was L1-bound on scattered activation reads.
4. **INT4 is not free in quality, and the control runs locate the loss.** Round-to-nearest INT4 (g=128) drops HumanEval pass@1 from **24.4% to 15.9%** (40 → 26 of 164 problems).
   - The same INT4 weights run with PyTorch matmul instead of the custom kernels score 14.6% (24/164); the v1.0 kernels scored 14.0% (23/164). Greedy pass@1 moves by ±3 problems with the fp32 summation order alone, so the kernels are not the cause.
   - Keeping the lm_head in bf16 scores 15.2% (25/164) while reading 200 MB more per token. The loss is in the 168 quantized projections of the decoder blocks, which makes calibrated quantization the first quality lever.
5. **Where the remaining 1.73 ms go.** The GPU kernels take 1.62 ms per token: the INT4 GEMVs 44%, RoPE 26% (double-precision sin/cos, one block per token), RMSNorm 8%, attention 9% and the argmax 7%. The host needs about as long to launch the 418 kernels of a step, so the next levers are fewer launches (CUDA Graphs) and the small kernels, not the big GEMVs.
6. **Speculative decoding depends on the cost of a multi-token forward.** Prompt Lookup Decoding is lossless. After the attention fix it reached **1.94×** overall, because a 5.3-token verify cost only 1.49× a single-token step. After the GEMV fix the single-token step got 2.4× faster but multi-token forwards (Split-K GEMM) did not: a verify now costs 3.8× a step and PLD gives **1.13×**.
7. **Even llama.cpp is far from its ceiling at 0.5B.** Q4_0 moves 2.85× fewer bytes than F16 but is only 1.66× faster, at 29% of its bandwidth ceiling. For a model this small, fixed per-token costs are a large share of latency; bandwidth is only part of the story.

---

## Table of Contents

- [Why This Project Exists](#why-this-project-exists)
- [Architecture](#architecture)
- [How Correctness Is Verified](#how-correctness-is-verified)
- [What Changed in v1.1](#what-changed-in-v11)
- [Implemented Levels](#implemented-levels)
  - [Level 0 — PyTorch Baseline](#level-0--pytorch-baseline)
  - [Levels 1–3 — GEMV Kernels (naive, optimized, INT4)](#levels-13--gemv-kernels-naive-optimized-int4)
  - [Level 4 — Decode Attention](#level-4--decode-attention)
  - [Level 5 — C++ Transformer Engine](#level-5--c-transformer-engine)
  - [Level 6 — Prompt Lookup Decoding](#level-6--prompt-lookup-decoding)
- [Reference: llama.cpp](#reference-llamacpp)
- [Quantization vs. Quality](#quantization-vs-quality)
- [Benchmarking Methodology](#benchmarking-methodology)
- [Scope](#scope)
- [Roadmap](#roadmap)
- [Hardware and Software](#hardware-and-software)
- [Setup & Usage](#setup--usage)
- [Portfolio Context](#portfolio-context)

---

## Why This Project Exists

Most "optimized inference" portfolio projects look alike. They show an isolated `tok/s` number with no hardware context, treat quantization as free in quality, and apply generic optimizations with no link to a product. CodeAlign-Runtime differs on three axes:

1. **Every speed number is anchored to a ceiling:** bytes read per token / memory bandwidth = minimum possible latency ([`scripts/roofline.py`](scripts/roofline.py)).
2. **Quantization is measured as a trade-off.** Speed is reported *and* quality is measured: HumanEval pass@1 of bf16 vs. INT4, with the same evaluation harness used in [CodeAlign](https://github.com/Sergasgr/CodeAlign), plus control runs that separate the kernels from the quantization.
3. **Correctness is tested, not assumed.** Every kernel is checked against a CPU/PyTorch reference, and the whole engine against Hugging Face's Qwen2 implementation.

The target use case is IDE code completion: short prompts, batch=1, and the latency per token is what the user feels.

---

## Architecture

```
CodeAlign-Runtime/
├── src/
│   ├── include/codealign_cuda.h  # error checking + stream-ordered kernel launches (CODEALIGN_SYNC_CHECK=1 debug mode)
│   ├── gemv/                     # GEMV kernels: naive, optimized (fp32), INT4 naive/optimized (v1.0) and v2 (v1.1)
│   ├── gemm/                     # GEMM kernels: naive, tiled (fp32), INT4 naive/tiled/Split-K
│   ├── ops/                      # batched GQA attention, RMSNorm, RoPE, SwiGLU, residual/bias, KV-cache, argmax,
│   │                             # and the LEGACY v1.0 Flash-Decoding (Level 4 baseline only)
│   ├── transformer/              # Level 5: QwenBlock (structs, memory, forward, pybind11 binding)
│   └── speculative/              # Level 6: n-gram oracle for Prompt Lookup Decoding
├── benchmarks/
│   ├── gemv_benchmark.cpp        # C++ harness: validation + L2-hot/L2-cold timing of the GEMV kernels (incl. lm_head shape)
│   ├── gemm_benchmark.cpp        # C++ harness: same for the GEMM kernels (incl. Split-K)
│   ├── benchmark_utils.{h,cpp}   # CUDA_CHECK, percentiles, L2 flush, INT4 quantization, CPU references, JSON
│   ├── gemv_binding.cpp          # PyTorch binding: INT4 GEMV v2 / v1.0, GQA attention, v1.0 Flash-Decoding
│   └── gemm_binding.cpp          # PyTorch binding: INT4 GEMM
├── scripts/
│   ├── baseline.py               # Level 0: PyTorch eager TTFT/TPOT/VRAM
│   ├── flash_decoding_baseline.py# Level 4: validation + benchmark vs PyTorch (single head and Qwen attention layer)
│   ├── engine.py                 # Level 5: CodeAlignEngine (24 QwenBlocks + embedding/norm/lm_head glue)
│   ├── transformer_inference.py  # Level 5: block / full-model / end-to-end decode benchmark (--profile: Nsight)
│   ├── generate_speculative.py   # Level 6: Prompt Lookup Decoding + lossless check
│   ├── quantization.py           # INT4 quantization + QuantizedLinearINT4 (kernels) / DequantizedLinearINT4 (torch control)
│   ├── evaluate_quality.py       # HumanEval pass@1: bf16, INT4 kernels, INT4 dequantized, lm_head ablation (bigcode)
│   ├── analyze_humaneval.py      # re-runs the saved generations: per-problem status, overlap, error types
│   ├── llama_cpp_benchmark.py    # llama.cpp reference (F16, Q4_0, Q4_K_M) via llama-bench
│   ├── roofline.py               # bandwidth ceilings (bytes / 896 GB/s)
│   ├── report.py                 # results/*.json -> the markdown tables of this README
│   ├── results_io.py             # JSON results + environment (GPU, driver, versions, git commit)
│   ├── inference_config.py       # model id, dimensions, benchmark constants
│   └── baseline_config.py        # Level 0 prompt and iteration counts
├── tests/                        # pytest: CPU tests + GPU tests (kernels, engine vs Hugging Face)
├── results/                      # JSON output of every benchmark + Nsight summaries in profiles/ (committed)
├── notes/v1.1-log.md             # step-by-step v1.1 log: what changed, measurements, what they say
├── build.sh                      # builds the CMake harness + the 3 PyTorch extensions
├── CMakeLists.txt, *_setup.py    # native build / PyTorch extension builds
├── pyproject.toml, uv.lock       # Python dependencies (uv)
└── Dockerfile                    # CUDA 13.0 environment matching the locked torch build
```

---

## How Correctness Is Verified

Speed numbers mean nothing if the kernel computes something else. Every level has a check that would fail on a real bug:

| What | Reference | Where |
|------|-----------|-------|
| GEMV/GEMM kernels (fp32 and INT4, incl. Split-K and the GEMV v2 bias epilogue), the four decode shapes and the 151,936×896 lm_head | CPU with double accumulation on **random data**. INT4 kernels are compared against the *dequantized* weights, so a kernel bug cannot hide inside the quantization error. Outputs are pre-filled with NaN, so an output the kernel never writes fails. | `benchmarks/*_benchmark.cpp`, `tests/test_kernels_cuda.py` |
| Batched GQA attention | float64 causal attention with K/V repeated per group: (14, 2), (8, 1), (1, 1) heads, head_dim 64/128, 1/3/11 tokens, cached lengths around every chunk boundary up to 2,048. The cache beyond the valid length **and the scratch buffers are filled with NaN**, so reading past the causal limit or a partial the kernel never wrote fails. Dominant keys in the first and last chunks; repeated calls must be bit-identical; unsupported shapes must raise. | `tests/test_kernels_cuda.py`, `scripts/flash_decoding_baseline.py` |
| Single-head decode attention (v1.1 kernel and the LEGACY v1.0 kernel) | PyTorch softmax attention with standard-normal inputs, S from 1 to 4,096 (incl. non-multiples of the chunk), plus dominant keys. The script checks that its data would **fail a kernel that ignored the scores** (one returning mean(V)); with uniform [0,1) inputs the softmax is nearly flat, and that shortcut would pass a 1e-2 threshold. | `scripts/flash_decoding_baseline.py`, `tests/test_kernels_cuda.py` |
| One decoder block | Hugging Face `Qwen2DecoderLayer` with the same INT4 weights dequantized, in float64 | `tests/test_engine_cuda.py` |
| Full engine | Hugging Face logits at every position, and every greedy token (teacher forcing), in float64 | `tests/test_engine_cuda.py` |
| Prompt Lookup Decoding | Must produce exactly the greedy output. The KV-cache length after rollbacks is checked. | `tests/test_engine_cuda.py`, `scripts/generate_speculative.py` |
| INT4 packing, `nn.Linear` drop-ins, roofline, kernel index math | Bit-level layout, round-trip error ≤ half a step; numpy ports of the GEMV v2 (staging, persistent grid, nibble decoding) and of the GQA attention (chunk heuristic, warp split, LSE merges) against float64 references | `tests/test_quantization.py`, `tests/test_cpu_emulation.py` (no GPU needed) |
| HumanEval analysis | Re-running the saved generations must reproduce bigcode's pass@1 for every run (`matches_bigcode` in `results/humaneval_analysis.json`) | `scripts/analyze_humaneval.py`, `tests/test_analyze_humaneval.py` |

The engine tolerance is calibrated, not hard-coded: the engine's error against Hugging Face in float64 must stay within 10× of Hugging Face's own fp32 error. Real Qwen2.5 weights are ill-conditioned, so a fixed threshold would be either too loose or too tight. The engine tests run on three models:

- a small random-weight Qwen2 (same architecture, smaller dimensions);
- the same small model with large q/k biases, which reproduces Qwen2.5's conditioning;
- the real Qwen2.5-Coder-0.5B (`-m "not real_model"` skips the download).

All tests also pass with `CODEALIGN_SYNC_CHECK=1`, which synchronizes after every launch so an asynchronous error is reported by the kernel that caused it.

---

## What Changed in v1.1

v1.0 was correct but slow (56 ms per token). Profiling one decode step with Nsight Systems and the key kernels with Nsight Compute showed three causes, and each v1.1 step removes one of them. Every step was measured on a clean commit with the GPU free of other clients; the full log, with the profiles and the intermediate tables, is in [`notes/v1.1-log.md`](notes/v1.1-log.md).

| Step | Change | v1.0 evidence | Model p50, S=512 | Block p50, S=512 | Launches per token |
|---|---|---|---:|---:|---:|
| v1.0 | — | — | 55.89 ms | 2.08 ms | 1,114 |
| 1 | Stream-ordered launches, no `cudaDeviceSynchronize()` per kernel | GPU idle 22% of the step; 1,106 device syncs per token | 49.65 ms | 1.82 ms | 1,114 |
| 2 | Batched GQA attention: 2 launches per layer for every head and token | Attention 72.5% of the step; each KV head read 7 times; 2 blocks per launch | 4.11 ms | 0.135 ms | 490 |
| 3 | INT4 GEMV v2 (activations in shared memory, persistent grid, bias in the epilogue) | INT4 GEMV L1-bound, 9–14% of DRAM bandwidth | **1.73 ms** | **0.057 ms** | 418 |
| 4 | Quality controls: INT4 on PyTorch matmul, lm_head in bf16, per-problem analysis | INT4 pass@1 14.0% with no control run | — | — | — |

v1.0 here is the re-measurement in the same clean environment (within 1% of the published 56.09 ms). End to end, TTFT went from 552.6 to 28.1 ms and TPOT p50 from 31.5 to 1.71 ms.

**Legacy kernels.** `src/ops/flash_decoding_*.cu` (v1.0 attention) and `gemv_int4_optimized_kernel` (v1.0 INT4 GEMV) are no longer used by the engine. They stay in the repo, compiled, tested and benchmarked, as the v1.0 rows of the Level 4 and Levels 1–3 tables, so every before/after in this README is measured in the same run on the same data. Their headers record what each design got wrong: the v1.0 attention launched two kernels per token and head and walked each 256-token chunk one token at a time; the v1.0 GEMV read the activation vector with scattered per-lane loads and was L1-bound.

---

## Implemented Levels

### Level 0 — PyTorch Baseline

[`baseline.py`](scripts/baseline.py) loads Qwen2.5-Coder-0.5B in bf16 with Hugging Face `transformers` (eager mode). It measures the two metrics serving systems report separately, with CUDA events, warmup discarded, over 100 runs of a 66-token code prompt plus up to 50 generated tokens:

- **TTFT (Time-To-First-Token):** prefill of the whole prompt.
- **TPOT (Time-Per-Output-Token):** one decode step with the KV-cache. This is the metric the rest of the project attacks.

| Metric | p50 | p90 | p99 |
|--------|---:|---:|---:|
| TTFT (66-token prompt) | 15.90 ms | 16.99 ms | 18.89 ms |
| TPOT | 14.61 ms | 15.68 ms | 17.62 ms |

The ceiling is computed from the loaded model: every parameter byte is read once per token (494 M params × 2 bytes = 988 MB). The tied lm_head/embedding matrix counts once, because the lm_head reads it whole and the embedding lookup reads one row. That gives **1.10 ms/token**. TPOT p50 reaches **7.5%** of it, with a peak of 997 MB of VRAM.

7.5% is typical of eager PyTorch at batch=1. Each decode step dispatches hundreds of small kernels from Python, and the GPU waits between them (not profiled here). The CUDA-event span includes that wait, because it is part of the latency a user sees.

---

### Levels 1–3 — GEMV Kernels (naive, optimized, INT4)

**Files:** [`gemv_naive.cu`](src/gemv/gemv_naive.cu), [`gemv_optimized.cu`](src/gemv/gemv_optimized.cu), [`gemv_quantized.cu`](src/gemv/gemv_quantized.cu), [`quantization.py`](scripts/quantization.py)

- **Level 1, naive:** one thread per output row. Adjacent threads read different rows, so each 32-thread load touches 32 cache lines.
- **Level 2, optimized:** one warp per row (coalesced), `float4` loads (16 B per instruction), and a warp-shuffle reduction (`__shfl_down_sync`).
- **Level 3, INT4:** weights quantized ahead of time and dequantized inside the kernel, so fp32 weights are never materialized.
  - **Scheme:** symmetric per-group INT4 with group size 128 and one fp32 scale = `max(|w|) / 7` per group. Values are rounded to nearest (RTN, no calibration) in [-7, 7] and packed 8 per `uint32_t`; the Python quantizer, the C++ harness and every kernel share this bit layout, and a test checks it.
  - **Size:** 0.53 bytes per weight, i.e. **3.76× fewer bytes than bf16 and 7.5× fewer than fp32**.
  - **v1.0 variants:** `gemv_int4_naive_kernel` (one thread per row) and `gemv_int4_optimized_kernel` (one warp per row, `uint4` loads, warp shuffle).
  - **v2 (v1.1, the engine's kernel):** `gemv_int4_v2_kernel`, designed from the v1.0 Nsight Compute diagnosis below:
    - the activation vector is staged in shared memory once per block, split into two `float4` arrays so that the lane holding word *w* reads entries *w* of both and consecutive lanes hit consecutive banks (no conflicts, no scattered global loads);
    - a persistent grid (4 blocks × 8 warps per SM), so the vector is staged ~280 times in total instead of once per few rows;
    - each warp takes two 896-wide rows as one contiguous span of 224 words (7 full warp loads, no idle lanes; one row at a time for rows ≥ 2,048 wide), and each lane issues 8 word loads before any math;
    - nibbles become floats without the quarter-rate integer-to-float instruction: `word ^ 0x88888888` turns each two's-complement nibble *v* into *v* + 8, and `0x4B000000 | n` is the float 2²³ + *n*, so subtracting 2²³ + 8 gives *v* exactly;
    - the projection bias is added in the epilogue, which removes the separate bias launches of q, k and v.

Every kernel is timed twice. **L2-hot** means repeated runs over the same data; the 17.4 MB fp32 matrix fits in the 48 MB L2, so reads can exceed DRAM bandwidth. **L2-cold** means a 96 MB buffer is written before each timed run, so operands come from DRAM. Only L2-cold numbers are compared with the 896 GB/s ceiling. Timings include each wrapper's launch; since v1.1 the wrappers no longer synchronize the device, so the same v1.0 kernels time lower than in the v1.0 tables (fp32 optimized, L2-cold: 33.5 → 28.4 µs).

**MLP up/gate projection (4864×896)**, the v1.0 benchmark shape:

| Kernel | L2-hot p50 | L2-hot GB/s | L2-cold p50 | L2-cold GB/s | % of 896 GB/s (cold) |
|--------|---:|---:|---:|---:|---:|
| Level 1 — fp32 naive | 89.2 µs | 195.7 | 91.4 µs | 191.1 | 21.3% |
| Level 2 — fp32 optimized | 9.3 µs | 1868.1 | 28.4 µs | **614.3** | **68.6%** |
| Level 3 — INT4 naive | 27.7 µs | 84.3 | 51.0 µs | 45.8 | 5.1% |
| Level 3 — INT4 optimized (v1.0) | 25.7 µs | 91.1 | 26.4 µs | 88.7 | 9.9% |
| Level 3 — **INT4 v2 (v1.1)** | **5.1 µs** | 456.7 | **7.9 µs** | 295.2 | 33.0% |

**lm_head (151,936×896)**, the matrix that measures DRAM bandwidth (72.9 MB in INT4, 545 MB in fp32, both larger than L2):

| Kernel | L2-hot p50 | L2-hot GB/s | L2-cold p50 | L2-cold GB/s | % of 896 GB/s (cold) |
|--------|---:|---:|---:|---:|---:|
| fp32 optimized | 643.1 µs | 847.7 | 682.8 µs | 798.4 | 89.1% |
| INT4 optimized (v1.0) | 702.6 µs | 103.8 | 702.1 µs | 103.9 | 11.6% |
| **INT4 v2 (v1.1)** | **95.0 µs** | 767.9 | **110.2 µs** | **661.9** | **73.9%** |

Validation: 30/30 kernel × shape checks (5 shapes × 6 kernels, including v2 with its bias).

What this shows:

- **Coalescing + vectorization work.** Level 1 → Level 2 is 3.2× from DRAM, and the optimized fp32 kernel reaches 89% of the spec bandwidth on the 545 MB lm_head.
- **v1.0's INT4 kernel was not limited by memory.** It moved 7.5× fewer bytes than fp32 and was barely faster, with the same time L2 hot or cold. Nsight Compute pinned it on the activation-vector access pattern (v1.0 kernels profiled alone at locked clocks with caches flushed, `results/profiles/v1.0_gemv_*.txt`; v2 profiled inside the engine, `results/profiles/v1.1-gemv_lmhead.txt`):

| Nsight Compute | fp32 optimized, 4864×896 | INT4 v1.0, 4864×896 | INT4 v1.0, lm_head | **INT4 v2, lm_head** |
|---|---:|---:|---:|---:|
| Kernel duration | 27.3 µs | 31.0 µs | 867 µs | **105 µs** |
| DRAM throughput | 82.8% | 9.1% | 13.9% | **80.0%** |
| L1/TEX throughput | 15.1% | 97.2% | 99.7% | 42.1% |
| Useful bytes per 32-byte sector (global loads) | not flagged | 4.5 | 4.5 | 27.3 |
| Top warp stall | waiting on memory (96%) | LG queue full (42%) | LG queue full (47%) | waiting on memory (62%) |

- **v1.0, L1-bound:** lane *L* read `vec[32·L … 32·L+31]` one float at a time, so each load instruction touched a different 32-byte sector per lane and used 4.5 of its 32 bytes. L1 saturated while DRAM idled, and on 896-wide rows 4 of the 32 lanes sat idle.
- **v2, DRAM-bound:** with the vector in shared memory the kernel reads the lm_head at 80% of peak DRAM bandwidth inside the engine (`results/profiles/v1.1-gemv_lmhead.txt`). Shared-memory accesses are conflict-free (0.2% excess wavefronts). Occupancy is limited by registers (64 per thread, 4 blocks per SM), which leaves ~10% of headroom.
- **INT4 now pays off:** on the lm_head INT4 v2 is 6.2× faster than fp32 for 7.5× fewer bytes. On the small 2.3 MB matrices the kernel is launch- and latency-bound (7.9 µs cold, 33% of the bandwidth), which is why the engine's per-token GEMV time is dominated by the 168 small projections, not the lm_head.

**GEMM kernels** (16 tokens × 4864×896) are used for multi-token forwards: speculative verification and chunked prefill. Every one of them validates on three shapes, including 11 and 3 tokens.

| Kernel | L2-hot p50 | L2-cold p50 | L2-cold GB/s | % of 896 GB/s (cold) |
|--------|---:|---:|---:|---:|
| fp32 naive | 236.6 µs | 241.2 µs | 73.8 | 8.2% |
| fp32 tiled (shared memory 16×16) | 54.4 µs | 71.6 µs | 248.7 | 27.8% |
| INT4 naive | 48.2 µs | 61.4 µs | 43.7 | 4.9% |
| INT4 tiled | 54.4 µs | 56.9 µs | 47.1 | 5.3% |
| INT4 Split-K (engine path) | 45.0 µs | 47.3 µs | 56.7 | 6.3% |

The INT4 GEMMs are untouched in v1.1 and are now the slow path of the engine. When a tile is loaded, each thread reads a whole 32-bit word to extract a single 4-bit weight, so every packed word is fetched 8 times. That limits speculative verification and prefill ([Level 6](#level-6--prompt-lookup-decoding), [Roadmap](#roadmap)).

---

### Level 4 — Decode Attention

**Files:** [`gqa_attention.h`](src/ops/gqa_attention.h), [`gqa_attention.cu`](src/ops/gqa_attention.cu) (v1.1); [`flash_decoding_partial.cu`](src/ops/flash_decoding_partial.cu), [`flash_decoding_final.cu`](src/ops/flash_decoding_final.cu) (LEGACY v1.0)

During decode, one query attends to a growing KV-cache, so there is nothing to parallelize over in the query dimension. **Flash-Decoding** parallelizes over the cache instead: the cache is split into chunks, each block computes an **online softmax** (running max and sum) over its chunk and outputs a normalized partial vector and its log-sum-exp, and a final kernel merges the chunks with weights `exp(lse_chunk − lse_global)`. The attention matrix is never materialized.

**v1.0 (legacy):** one partial + one final launch per token and per query head, chunks of 256 tokens, one block per chunk that walks its tokens one at a time with two `__syncthreads()` per token, and every K/V row read once per *query* head (7 times per KV head in Qwen2.5-0.5B).

**v1.1, batched GQA attention:** two launches per layer for every token and head of a forward.

- **Partial kernel**, grid (chunks × KV heads × tokens), 4 warps: each block reads its K/V rows **once** and serves the 7 query heads that share them. The warps take different positions of the chunk, each with an online softmax per head in registers; the group size is a template parameter, so the 7 score reductions of a position run as independent shuffles instead of one after another. Each warp loads 4 positions before any math. The 4 warp states are merged once, in shared memory.
- **Final kernel**, grid (heads × tokens): merges only the chunks the token can see (causal: token *t* of a forward sees the cache plus tokens ≤ *t*).
- **Chunk size adapts to the context:** enough blocks for ~2 per SM, at least 32 positions per block, at most 32 chunks (S=512 at T=1: 16 chunks × 2 KV heads = 32 blocks; S=2048: 32 chunks of 64).

**Validation:** 30/30 single-head cases (worst error 1.3e-7, tolerance 1e-4), both kernels; 8/8 Qwen-layer cases against float64 (worst 2.6e-7); 180 pytest cases of the batched kernel (see [How Correctness Is Verified](#how-correctness-is-verified)).

**Single head** (head_dim 64, fp32, against PyTorch's `Q @ Kᵀ → softmax → @ V`; both kernels in the same run). The ceiling is the time to read K and V once.

| Seq length | PyTorch p50 | v1.0 kernel p50 | v1.1 kernel p50 | PyTorch / v1.1 | v1.0 / v1.1 | v1.1 % of DRAM ceiling |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.098 ms | 0.131 ms | **0.023 ms** | 4.26× | 5.69× | — |
| 4,096 | 0.098 ms | 0.135 ms | **0.023 ms** | 4.22× | 5.83× | — |
| 16,384 | 0.101 ms | 0.137 ms | **0.023 ms** | 4.39× | 5.96× | — |
| 65,536 | 0.104 ms | 0.145 ms | **0.035 ms** | 3.02× | 4.21× | — |
| 131,072 | 0.122 ms | 0.252 ms | **0.099 ms** | 1.23× | 2.55× | 75.6% |
| 262,144 | 0.267 ms | 0.285 ms | **0.182 ms** | 1.47× | 1.57× | 82.5% |

**Qwen2.5-0.5B attention layer** (14 query heads over 2 KV heads, T new tokens after S − T cached positions, causal), against `scaled_dot_product_attention(..., enable_gqa=True)`:

| T | S | SDPA p50 | v1.1 kernel p50 | SDPA / v1.1 |
|---:|---:|---:|---:|---:|
| 1 | 128 | 0.166 ms | 0.021 ms | 8.10× |
| 1 | 512 | 0.167 ms | 0.021 ms | 7.98× |
| 1 | 2,048 | 0.167 ms | 0.021 ms | 7.96× |
| 6 | 128 | 0.195 ms | 0.021 ms | 9.36× |
| 6 | 512 | 0.196 ms | 0.021 ms | 9.27× |
| 6 | 2,048 | 0.196 ms | 0.031 ms | 6.40× |

- **Up to ~16k tokens the ~21–23 µs per call is host overhead** of the binding (scratch allocation and two launches), not kernel time. Inside the engine the profile shows the real cost: **5.0 µs (partial) + 1.3 µs (final) per layer at S=512**, 4.4 + 1.1 µs at S=128. In v1.0 the same layer took ~1.9 ms.
- **Up to 65k tokens the KV-cache (≤ 33.5 MB) fits in the 48 MB L2** across the repeated timed runs (at 65k the measured time even beats the DRAM "ceiling"), so those rows are not reported as a % of DRAM bandwidth. From 131k the cache comes from DRAM and the kernel reaches 76–83% of the bandwidth ceiling.
- **The GQA layout matters more than raw speed:** v1.0 read each KV head 7 times per token; v1.1 reads it once, and verifying T tokens costs about as much as one ([Level 6](#level-6--prompt-lookup-decoding)).

---

### Level 5 — C++ Transformer Engine

**Files:** [`transformer.h`](src/transformer/transformer.h), [`transformer.cpp`](src/transformer/transformer.cpp), [`memory.cpp`](src/transformer/memory.cpp), [`transformer_binding.cpp`](src/transformer/transformer_binding.cpp), [`engine.py`](scripts/engine.py), and the ops in [`src/ops/`](src/ops)

Every kernel from the previous levels is assembled into the full Qwen2.5-Coder-0.5B decoder.

**One decoder block in C++ (`QwenBlock`):**

```
RMSNorm → INT4 q/k/v GEMV v2 (bias in the epilogue) → RoPE (θ = 10⁶) → KV-cache append
→ batched GQA attention (14 query heads share 2 KV heads; 2 launches) → INT4 o_proj → residual
→ RMSNorm → INT4 gate/up → SwiGLU → INT4 down → residual
```

- **Qwen2 specifics:** grouped-query attention (k/v project 896 → 128; query head *h* reads KV head *h / 7*) and a bias on q/k/v.
- **Launches:** 17 per block per decode step, all on PyTorch's current stream and in order. There is no device synchronization inside a forward: launch errors raise immediately (`cudaGetLastError`), errors inside a kernel surface at the next synchronization (the argmax `.tolist()`), and `CODEALIGN_SYNC_CHECK=1` synchronizes after every launch for debugging.
- **Memory:** weights are raw GPU pointers owned by the binding, the KV-cache is `[kv_heads, max_seq_len, head_dim]`, and every intermediate buffer (including the attention scratch: one row per token, head and chunk) is allocated once in the constructor. **The C++ blocks never allocate during inference.**
- **Multi-token forwards:** up to 11 tokens per call (1 + 10 draft tokens). Projections switch from the INT4 GEMV to the INT4 Split-K GEMM. Attention is causal inside the call: token *t* sees the cache plus tokens ≤ *t*.
- **Safety:** shapes, dtypes, token count, KV-cache overflow and rollback range are checked, and CUDA errors surface as Python exceptions.
- **Numerics:** the extensions are compiled with `-use_fast_math`, which turns `powf`/`sinf`/`cosf` into approximate hardware instructions (`sin.approx`, `cos.approx`, `lg2.approx`, `ex2.approx` in the PTX). In RoPE that means angle errors of ~1e-5 rad.
  - Qwen2.5's q/k biases are large, so |q|·|k| is large, and that tiny error drifted the engine from Hugging Face by ~1.6% after one layer and ~12% in the logits. The engine tests caught it.
  - RoPE therefore follows Hugging Face's fp32 angle (`fp32(pos × inv_freq)`) and evaluates sin/cos in double precision, which `-use_fast_math` does not affect. That was negligible in v1.0; in v1.1 it is 19% of the step (see below), and a precomputed cos/sin table is the planned fix.

**The full model (`CodeAlignEngine`, [`engine.py`](scripts/engine.py))** is 24 `QwenBlock`s plus small Python glue: the embedding row lookup, the final RMSNorm, the lm_head on the same INT4 GEMV, and a GPU argmax. Prompts longer than 11 tokens are prefilled in chunks through the same causal path. At load time the engine checks that the model's config matches what the kernels hard-code.

```python
from scripts.engine import CodeAlignEngine

engine = CodeAlignEngine.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")
new_tokens = engine.generate(prompt_ids, max_new_tokens=128, eos_token_ids=[tokenizer.eos_token_id])
```

**Benchmark** ([`transformer_inference.py`](scripts/transformer_inference.py)): decode latency at fixed context lengths. Before each timed step the KV-cache length is set to S−1, so every sample is exactly one decode step at context S. Ceilings include the KV-cache bytes read at S.

| Context S | Block p50 | Block % of ceiling | Model p50 | Model tok/s | Model % of ceiling | v1.0 model p50 |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.057 ms | 15.8% | **1.708 ms** | 586 | 17.4% | 33.58 ms |
| 512 | 0.057 ms | 16.6% | **1.730 ms** | 578 | 17.8% | 56.09 ms |
| 1,024 | 0.057 ms | 17.5% | **1.761 ms** | 568 | 18.2% | 56.03 ms |
| 2,048 | 0.059 ms | 18.9% | **1.822 ms** | 549 | 19.2% | 56.02 ms |

Ceilings: one block reads 7.93 MB (9–11 µs); the full model reads 263 MB (0.30–0.35 ms), with the lm_head in INT4. Unlike v1.0, the block now grows with S, because attention is bound by the KV it reads.

**End to end:** a 53-token prompt gives TTFT 28.1 ms (chunked prefill) and TPOT p50 1.714 ms over 128 generated tokens (v1.0: 553.7 and 31.5 ms). The continuation is coherent code, identical to v1.0's:

```python
    with open(path, encoding="utf-8") as file:
        users = json.load(file)
        users = [User(**user) for user in users]
    return users
```

**Memory:** 608 MB resident (511 MB of tensors, including the bf16 embedding table, plus 98 MB of C++ buffers, 9.7 MB more than v1.0 for the attention scratch), and a 609 MB peak during decode. That is 39% less than PyTorch's 997 MB peak. The one-off INT4 quantization at load peaks at 2.27 GB; a pre-quantized checkpoint (v2.0, track B) removes it.

**Where the time goes:** Nsight Systems over 20 decode steps per context (`transformer_inference.py --profile`, summaries in `results/profiles/v1.1-gemv_decode_*`; the v1.0 profile is in `results/profiles/v1.0_decode_*`). The profiler and its NVTX ranges add host overhead, so this table gives shares; the latencies above come from the benchmark, which runs without it.

| Share of one decode step (GPU span) | S=128 | S=512 |
|---|---:|---:|
| INT4 GEMV v2 (169 launches; lm_head ~95 µs) | 32.1% | 32.5% |
| RoPE (48 launches) | 18.5% | 18.7% |
| Attention (48 launches) | 5.9% | 6.9% |
| RMSNorm (48 launches) | 5.9% | 6.0% |
| argmax over 151,936 logits (1 launch) | 5.5% | 4.8% |
| Other (residual, KV append, SwiGLU, PyTorch glue) | 4.2% | 4.2% |
| GPU idle between kernels | 27.8% | 26.9% |
| Step span under the profiler | 2.23 ms | 2.22 ms |

In v1.0 (S=512) the step spanned 63.6 ms under the profiler, attention was 72.5% of it and the GPU sat idle 21.7% of the time between synchronized launches.

- **Kernel time is 1.62 ms per token.** The INT4 GEMVs take 0.72 ms of it: ~95 µs for the lm_head and ~3.7 µs on average for each of the other 168 projections, which are small enough (57 KB to 2.3 MB) to be latency-bound.
- **The small kernels are now visible.** RoPE (13.1 µs per launch on q: one block per token evaluating sin/cos in double precision, which runs at 1/64 of fp32 speed on this GPU), RMSNorm (two passes over 256 threads) and the single-block argmax (107 µs) together cost 0.65 ms per token.
- **The host is now as slow as the GPU.** A step is 418 launches. With NVTX ranges the host needs 2.29 ms per step (74 µs per layer for 17 launches), so under the profiler the GPU waits 27% of the time. Without NVTX the benchmark's 1.73 ms is within 7% of the 1.62 ms of kernel time: launch cost and GPU time are balanced, and further kernel savings will soon hit the launch floor (CUDA Graphs, [v2.0 track B](#roadmap)).
- **TTFT** is 28.1 ms because prefill still goes through the 11-token verify path: attention handles the 11 tokens in one launch, but the projections run the INT4 Split-K GEMM.

---

### Level 6 — Prompt Lookup Decoding

**Files:** [`speculative.cpp`](src/speculative/speculative.cpp), [`generate_speculative.py`](scripts/generate_speculative.py)

Speculative decoding without a draft model:

1. **Oracle (C++):** find the last 3 tokens earlier in the history and propose the ≤5 tokens that followed them.
2. **Batched verify:** `[last token] + draft` go through the engine in one causal forward.
3. **Accept** the longest draft prefix that matches the greedy predictions, plus one bonus token.
4. **Roll back** the rejected tokens from the KV-cache of all 24 layers.

Speeds are decode-only; the prompt prefill is the same in both modes. 256 new tokens per prompt, unless EOS comes first.

| Prompt | Greedy tok/s | PLD tok/s | Speedup | Acceptance | Tokens/forward | Identical output |
|--------|---:|---:|---:|---:|---:|:---:|
| refactor (type hints) | 588.7 | 799.2 | 1.36× | 91.6% | 5.33 | ✅ |
| C++ getters/setters | 589.1 | 502.9 | 0.85× | 36.6% | 1.33 | ✅ |
| unit tests (69 tokens, EOS) | 590.9 | 672.9 | 1.14× | 80.0% | 2.09 | ✅ |
| open-ended prompt | 589.2 | 799.8 | 1.36× | 97.4% | 3.66 | ✅ |

- **Lossless:** the output is identical to plain greedy decoding on every prompt.
- **The speedup is set by the cost of a verify forward relative to a single-token step,** and each v1.1 step moved that ratio:

  | | v1.0 | After batched attention (step 2) | After GEMV v2 (step 3) |
  |---|---:|---:|---:|
  | Cost of a ~5.5-token verify / a 1-token step | 4.75× | 1.49× | 3.8× |
  | Overall PLD speedup | 1.00× | 1.94× | 1.13× |

  Batched attention made verification almost free. GEMV v2 then made the single-token step 2.4× faster, while multi-token forwards still run the INT4 Split-K GEMM, so verification became relatively expensive again. When most drafts are rejected (C++ getters/setters), PLD is slower than greedy. A multi-token INT4 GEMV that reads the weights once for all T tokens is the fix ([Roadmap](#roadmap)).
- **High acceptance is partly degenerate.** Under greedy decoding the 0.5B base model falls into repetition loops in the refactor and open-ended prompts (see `results/level6_speculative.json`), and PLD copies loops very well. The unit-test and C++ prompts are closer to real completions.
- **Overall:** 1.13× across the four prompts (total greedy time / total PLD time).

---

## Reference: llama.cpp

Same model, same GPU, measured with `llama-bench` ([`llama_cpp_benchmark.py`](scripts/llama_cpp_benchmark.py), llama.cpp build `0253fb21f`, default settings with flash attention off, 128 generated tokens after a KV-cache prefill of the given depth, 5 repetitions). Q4_0 (4-bit, one fp16 scale per 32 weights) is the format closest to this project's INT4 (g=128, fp32 scale). Q4_K_M is llama.cpp's usual default.

| GGUF | Size | Depth 128 | Depth 512 | Depth 1,024 | Depth 2,048 | % of ceiling (512) |
|------|---:|---:|---:|---:|---:|---:|
| F16 | 988 MB | 2.15 ms | 2.20 ms | 2.06 ms | 2.13 ms | 50% |
| Q4_0 | 346 MB | 1.31 ms | 1.33 ms | 1.35 ms | 1.43 ms | 29% |
| Q4_K_M | 392 MB | 1.38 ms | 1.39 ms | 1.42 ms | 1.49 ms | 31% |
| *This engine v1.1, INT4* | *263 MB* | *1.71 ms* | *1.73 ms* | *1.76 ms* | *1.82 ms* | *17%* |

The ceiling is the GGUF's weight bytes / 896 GB/s. Q4_0 moves 2.85× fewer bytes than F16 but is only 1.5–1.66× faster. At 0.5B parameters, even a mature engine spends a large share of each token on fixed costs such as launches, small-matrix inefficiency and sampling. Bandwidth explains the dense case well (50% of the ceiling); it explains INT4 much less. The v1.1 engine reads fewer bytes than Q4_0 and is 1.30× slower at depth 512; its profile above shows the same kind of fixed costs.

---

## Quantization vs. Quality

INT4 moves 3.76× fewer bytes than bf16. Does it cost the model correct code?

[`evaluate_quality.py`](scripts/evaluate_quality.py) runs **HumanEval pass@1 (greedy, one sample per problem)** with `bigcode-evaluation-harness`, the same harness used in [CodeAlign](https://github.com/Sergasgr/CodeAlign). Every `nn.Linear` is replaced, and INT4 prefill uses the INT4 GEMM and decode the INT4 GEMV. Four runs separate the possible causes of a loss:

- **bf16:** the reference.
- **INT4 on the kernels:** the engine's packed weights and kernels.
- **INT4 dequantized (control):** exactly the same quantized weights, dequantized once to fp32 and multiplied with PyTorch (`F.linear`, TF32 off). Same model, no custom kernels.
- **INT4 with the lm_head in bf16:** the ablation of the largest matrix (151,936×896).

[`analyze_humaneval.py`](scripts/analyze_humaneval.py) re-runs the saved generations (`results/humaneval_generations_*.json`) with HumanEval's tests and reproduces bigcode's pass@1 for every run. It adds the per-problem view: error types, generations with a run of 50+ whitespace characters, and a "body-only" score that cuts each completion at its first top-level line after the prompt.

| Run | pass@1 | Solved | Body-only | SyntaxError | ≥50-char blank runs | Weights read per token |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-Coder-0.5B bf16 (reference) | **24.4%** | 40 | 40 | 12 | 0 | 988 MB |
| INT4 g=128 RTN, lm_head INT4, **v1.1 kernels** | **15.9%** | 26 | 27 | 46 | 63 | 263 MB |
| INT4, same weights dequantized, PyTorch matmul (control) | 14.6% | 24 | 29 | 51 | 60 | 263 MB* |
| INT4, lm_head kept in bf16, v1.1 kernels | 15.2% | 25 | 28 | 47 | 77 | 463 MB |
| INT4, lm_head INT4, v1.0 kernels (`v1.0.0` tag) | 14.0% | 23 | 26 | 50 | 64 | 263 MB |

\* the INT4 content; the control stores the dequantized weights in fp32.

The absolute value depends on the harness and prompt format (bigcode's HumanEval prompts, greedy decoding, 512 tokens max). The comparisons that matter are between rows run under identical conditions. Activations between layers stay in Hugging Face's bf16 path.

- **The kernels are not the cause.** On the same quantized weights, the v1.1 kernels solve 26 problems, PyTorch matmul 24 and the v1.0 kernels 23. The generations differ only through fp32 summation order (96 of 164 are identical between the v1.1 kernels and the control; 101 between the v1.0 and v1.1 kernels), and the differences go both ways: 3 problems only the kernels solve, 1 only the control. **Greedy pass@1 of this checkpoint moves by ±3 problems (±1.8 points) from numerics alone,** so differences of that size are noise.
- **The INT4 lm_head is not the cause either.** Keeping it in bf16 changes most generations (only 14 of 164 identical) but scores 25 problems, within the noise of 26, while reading 200 MB more per token (weights-only ceiling 0.29 → 0.52 ms).
- **The loss is in the 168 quantized projections of the decoder blocks:** −14 to −17 problems (−35 to −43% relative) in every INT4 run. Per problem, bf16 and the INT4 kernels solve 22 problems in common; 18 only bf16 solves and 4 only INT4.
- **INT4 often does not stop cleanly.** After finishing the function it emits long runs of blank lines and then starts new text, which the 512-token limit cuts off: 60–77 of 164 INT4 generations contain a 50+ character whitespace run, in every INT4 run including the PyTorch control, and none of the bf16 ones do. Syntax errors go from 12 to 46–51.
- **Most of the loss is in the code itself, not in the stopping.** Scoring only the function body gives 26–29 for INT4 and still 40 for bf16.

The scheme is the simplest one: round-to-nearest with no calibration, symmetric (15 of the 16 levels), one fp32 scale per 128 weights. Better quantization, measured as bytes vs. pass@1, is [v2.0, track C](#v20--triton-a-python-free-runtime-and-faster-prefill).

---

## Benchmarking Methodology

1. **CUDA events** around every timed step, and **warmup discarded**.
2. **Distributions, not anecdotes:** p50/p90/p99 over ≥ 90 samples. p50 is the headline number.
3. **A ceiling next to every latency:** (weights + KV-cache bytes that must be read) / 896 GB/s, from [`roofline.py`](scripts/roofline.py).
4. **DRAM vs. L2:** isolated kernels are timed with L2 hot and with L2 flushed. Only L2-cold numbers are compared with DRAM bandwidth, and results that fit in L2 are not reported as a % of it.
5. **Fixed context:** decode latency is measured at a fixed KV-cache length, not averaged over a growing context.
6. **Correctness before speed:** if validation fails, no timings are reported. The C++ harnesses and the attention script write only the failed checks and exit non-zero.
7. **Traceable results:** every run writes `results/<name>.json` with the GPU, driver, CUDA/torch/transformers versions, git commit and a dirty flag. [`report.py`](scripts/report.py) turns them into the tables above.
8. **Profiles explain, benchmarks measure.** Nsight Systems (decode steps at a fixed context, with NVTX ranges per step and per section) and Nsight Compute (single kernels inside the engine) provide shares and per-kernel diagnostics. Their text/CSV summaries are committed in `results/profiles/`. Latencies reported as results come from the benchmarks, which run without a profiler.
9. **An exclusive GPU.** Without host synchronization the GPU never idles, and another client on the same GPU (a desktop compositor, a browser) preempts kernels: with the desktop active, 111 kernels per 20 profiled steps took 1–7 ms and the model p50 nearly doubled. Every v1.1 number is measured with no other GPU clients; v1.0, re-measured the same way, moves by less than 1%.
10. **Baselines in the same run.** The v1.0 kernels stay compiled and are timed next to their replacements, on the same data, in the same process.

---

## Scope

- **One model family:** the kernels hard-code Qwen2.5 choices (RoPE θ = 10⁶, RMSNorm ε = 10⁻⁶, hidden ≤ 1024, head_dim 64/128, ≤ 8 query heads per KV head), and the engine refuses other configs at load time.
- **Batch = 1, fp32 activations and KV-cache.** Only the weights are INT4.
- **Decode first:** prefill reuses the 11-token verify path. TTFT is reported but not optimized.
- **One GPU:** every number comes from a single RTX 5070 Ti (sm_120). The code builds for the local architecture (`CMAKE_CUDA_ARCHITECTURES native`).

---

## Roadmap

### v1.1 — Performance pass

The goal is a before/after comparison with the same harness, tests and scripts; the v1.0 numbers stay at the `v1.0.0` tag. Each item came from a v1.0 measurement.

| # | Change | Evidence in v1.0 | Result in v1.1 |
|---|--------|------------------|----------------|
| 0 | ✅ **Profile first:** Nsight Systems traces of 20 decode steps at S=128 and S=512, Nsight Compute reports for the key kernels, in `results/profiles/` | Attention share and INT4 diagnosis were estimates from timings | Profiles before and after each item |
| 1 | ✅ **Batched GQA attention (C++/CUDA):** two launches per layer over (KV chunks × KV heads × tokens); each block reads a K/V chunk once for the 7 query heads that share it; positions processed in parallel by the warps; adaptive chunk size; causal multi-token path in the same launches | Attention = 72.5% of the step; 2 blocks per launch, 4% occupancy; each KV head read 7 times; a 5.7-token verify costs 4.75× a single token | Model 49.7 → 4.1 ms; attention 1.9 ms → 6.3 µs per layer; verify 1.49× a single token |
| 2 | ✅ **No device synchronization per launch (C++):** stream-ordered launches, errors checked after each launch | 1,114 launches and 1,106 `cudaDeviceSynchronize()` calls per token; GPU idle 22% of the step | Model 55.9 → 49.7 ms; GPU busy 78% → 99% |
| 3 | ✅ **Coalesced INT4 GEMV v2 (CUDA):** activations in shared memory, no idle lanes on 896-wide rows, persistent grid, bias in the epilogue | INT4 only 1.09× faster than fp32 with 7.5× fewer bytes; L1-bound | lm_head 702 → 110 µs (74% of DRAM bandwidth); model 4.1 → 1.73 ms |
| 4 | **Before/after reporting:** `report.py --baseline v1.0.0` renders v1.0 and v1.1 side by side | — | Pending (release) |
| 5 | ✅ **Quality controls (Python):** INT4 on PyTorch matmul (`--precision int4-dequant`), lm_head kept in bf16 (`int4_bf16head`), `analyze_humaneval.py` | INT4 pass@1 14.0% vs 24.4%, with no control | Kernels and lm_head ruled out; loss in the block projections |
| 6 | **ONNX Runtime reference (C++):** Qwen2.5-Coder in ORT (CUDA EP), measured from C++ with the same methodology, as published and with exactly this project's INT4 weights | — | Pending (`feat/ort-reference`) |

**Target:** a decode TPOT below PyTorch eager (14.6 ms) at every measured context, with the same tests passing. ✅ **Met:** 1.71–1.82 ms at S = 128…2048, 8.0–8.6× below PyTorch eager and 32× faster than v1.0 at S=512. HumanEval pass@1 was re-measured on the new kernels: 15.9% (26/164) vs 14.0% (23/164) in v1.0, the same checkpoint within the ±3-problem noise of fp32 summation order.

**Open items found by the v1.1 profile** (feeding v2.0):

- **Multi-token INT4 path:** a GEMV v2 that reads the weights once for all T ≤ 11 tokens of a verify or prefill forward. It is what limits Prompt Lookup Decoding (a verify costs 3.8× a step) and TTFT.
- **Small kernels:** RoPE with a precomputed cos/sin table, a single-pass RMSNorm, a multi-block argmax. Together they cost 0.65 ms of the 1.62 ms of kernels per token.
- **Launch overhead:** 418 launches per token cost about as much host time as the GPU needs; CUDA Graphs (track B).

### v2.0 — Triton, a Python-free runtime, and faster prefill

Three tracks. Each one mixes Python and C++ and uses the same validation and benchmark methodology.

**A. Triton vs. hand-written CUDA**

- Re-implement the hot kernels in [Triton](https://github.com/triton-lang/triton): INT4 GEMV, batched GQA attention, and fused residual + RMSNorm. Each goes through the same pytest references and the same L2-hot/cold harness, so the comparison is CUDA C++ vs Triton vs PyTorch at identical shapes.
- **Bridge into the C++ engine:** compile the Triton kernels ahead of time with `triton.tools.compile`. It generates C sources that embed the cubin and expose a launcher (`CUresult kernel(CUstream, gridX, gridY, gridZ, args...)`). Link them into `QwenBlock` and select the backend per op (`cuda` / `triton`). The comparison then covers full-engine TPOT, not only microbenchmarks.
- **Done when:** every Triton kernel passes the existing tests, and there is a per-kernel and end-to-end table per backend.

**B. Python-free C++ runtime**

- **Checkpoint:** a pre-quantized format (packed INT4 + scales + metadata, written once by Python) loaded directly by C++.
- **Tokenizer in C++:** e.g. [`mlc-ai/tokenizers-cpp`](https://github.com/mlc-ai/tokenizers-cpp), which wraps Hugging Face tokenizers.
- **CUDA Graphs:** capture the decode step as a graph. The KV-cache length has to live in device memory so that the same graph can be replayed. In v1.1 the host needs about as long to launch a step as the GPU needs to run it.
- **`codealign-cli`:** streaming completion with fill-in-the-middle prompts (`<|fim_prefix|>…<|fim_suffix|>…<|fim_middle|>`, supported by Qwen2.5-Coder), which is the actual IDE completion format.
- **Done when:** the CLI's TTFT/TPOT are measured with the same methodology and compared with the Python-driven engine and llama.cpp.

**C. Precision and prefill**

- **bf16 activations and KV-cache** (`__nv_bfloat162` math), halving the KV traffic.
- **A tensor-core INT4 GEMM for prefill** (`mma.sync`, INT4 weights dequantized to bf16 in registers), with a dedicated prefill path. TTFT is 28.1 ms in v1.1 (553.7 ms in v1.0) vs 15.9 ms in PyTorch.
- **Better INT4, measured as bytes vs. pass@1.** INT4 loses 14–16 HumanEval problems in the block projections, so this is the first quality lever. Three steps, each reported as bytes/weight next to pass@1:
  - **Calibration** (AWQ-style activation-aware scale search, or GPTQ) in the same packed format, so the kernels are unchanged.
  - **Smaller groups** (g=64/32).
  - **Asymmetric groups** (a zero-point, all 16 levels), which needs a kernel change.
- **Done when:** TTFT, TPOT and HumanEval are reported against v1.1.

---

## Hardware and Software

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX 5070 Ti (GB203, Blackwell, compute capability 12.0, 70 SMs) |
| VRAM | 16 GB GDDR7, 256-bit |
| Memory bandwidth (spec) | 896 GB/s |
| L2 cache | 48 MB |
| Driver / CUDA | 595.91.07 / CUDA 13.1 toolkit, PyTorch 2.13.0+cu130 |
| Software | transformers 5.15.1, bigcode-evaluation-harness `8fc5bae`, llama.cpp `0253fb21f`, Nsight Systems 2025.5.2, Nsight Compute 2025.4.1 |

---

## Setup & Usage

### Prerequisites

- NVIDIA GPU with compute capability ≥ 7.5 (CUDA 13 minimum) and a driver that supports CUDA 13
- CUDA Toolkit 13.x (for `nvcc`) and CMake ≥ 3.24, or Docker (below)
  - `build.sh` uses `$CUDA_HOME`, or else the first `nvcc` on `PATH`, for both CMake and the PyTorch extensions. It stops if its major version differs from torch's (13.x). With several toolkits installed (e.g. Ubuntu's `nvidia-cuda-toolkit` 12.0 in `/usr/bin`), run `CUDA_HOME=/usr/local/cuda ./build.sh`.
- Python 3.12+ and [uv](https://github.com/astral-sh/uv)

### Option 1: Docker

```bash
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker   # one-time
docker build -t codealign-runtime .
docker run --gpus all -it --rm -v "$(pwd)":/app codealign-runtime
./build.sh   # inside the container
```

### Option 2: Local

```bash
uv sync
./build.sh            # CMake harness + the 3 PyTorch CUDA extensions (built in place)
```

### Tests

```bash
uv run pytest                               # CPU + GPU tests (GPU tests skip without a GPU/extensions)
uv run pytest -m "not real_model"           # skip the tests that download Qwen2.5-Coder-0.5B
CODEALIGN_SYNC_CHECK=1 uv run pytest        # synchronize after every launch (debug asynchronous errors)
uv run python -m tests.test_cpu_emulation   # CPU emulation report (no GPU)
```

### Benchmarks

Run these from the repository root, with no other GPU clients (see [methodology](#benchmarking-methodology)); each writes `results/<name>.json`.

```bash
./build/gemv_benchmark                                  # Levels 1-3 (+ lm_head shape)
./build/gemm_benchmark                                  # GEMM kernels (incl. Split-K)
uv run python -m scripts.baseline                       # Level 0
uv run python -m scripts.flash_decoding_baseline        # Level 4
uv run python -m scripts.transformer_inference          # Level 5
uv run python -m scripts.generate_speculative           # Level 6
uv run python -m scripts.evaluate_quality --precision bf16
uv run python -m scripts.evaluate_quality --precision int4
uv run python -m scripts.evaluate_quality --precision int4-dequant          # control: same weights, PyTorch matmul
uv run python -m scripts.evaluate_quality --precision int4 --keep-lm-head   # saved as int4_bf16head
uv run python -m scripts.analyze_humaneval              # per-problem analysis of the saved generations
LLAMA_CPP_DIR=/path/to/llama.cpp uv run --with gguf --with sentencepiece python -m scripts.llama_cpp_benchmark
uv run python -m scripts.report                         # markdown tables from results/*.json
```

### Profiling

`transformer_inference.py --profile` runs decode steps at one fixed context inside `cudaProfilerStart/Stop`, with NVTX ranges per step and per section (`--profile-no-sections` keeps only the per-step range). It writes no results JSON.

```bash
mkdir -p profiles results/profiles
# Nsight Systems: timeline + per-kernel / NVTX / API summaries
uv run nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true \
    -o profiles/v1.1-gemv_decode_s512 python -m scripts.transformer_inference --profile --profile-context 512
nsys stats --force-export=true --format csv \
    --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum,nvtx_gpu_proj_sum,cuda_gpu_trace \
    --output results/profiles/v1.1-gemv_decode_s512 profiles/v1.1-gemv_decode_s512.nsys-rep
# Nsight Compute: one kernel inside the engine (here the lm_head GEMV: GEMV launch 169 of each step)
uv run ncu --profile-from-start off --set full -f -k regex:gemv_int4_v2_kernel \
    --launch-skip 168 --launch-count 1 -o profiles/v1.1-gemv_lmhead \
    python -m scripts.transformer_inference --profile --profile-context 512 --profile-steps 2
ncu --import profiles/v1.1-gemv_lmhead.ncu-rep --page details > results/profiles/v1.1-gemv_lmhead.txt
```

Nsight Compute needs access to the GPU performance counters (`ERR_NVGPUCTRPERM` otherwise): set the driver option `NVreg_RestrictProfilingToAdminUsers=0` or run it as root.

`HF_TOKEN` is optional: the model and HumanEval are public. If you need one, copy `.env.example` to `.env`; it is passed to Docker with `--env-file .env`.

---

## Portfolio Context

| Project | Demonstrates |
|---------|-------------|
| [**CodeAlign**](https://github.com/Sergasgr/CodeAlign) | Data curation → SFT → DPO → HumanEval evaluation of a coding LLM |
| **CodeAlign-Runtime** | INT4 quantization → CUDA kernels → a full decode engine, measured against hardware ceilings and llama.cpp, and improved by profiling (v1.0 → v1.1: 32× faster) |

Together they cover the path from training data to serving. The engine serves the public Qwen2.5-Coder-0.5B. Serving CodeAlign's own post-trained checkpoint, once distilled to this size, is future work.

---

## License

MIT
