# CodeAlign-Runtime

**A C++/CUDA inference engine for small code models, measured against hardware ceilings.**

Hand-written CUDA kernels serve Qwen2.5-Coder-0.5B, a code completion model, at batch=1, which is the IDE autocompletion case. The project is built in levels, from a naive GEMV up to a full 24-layer INT4 engine with speculative decoding. Every latency is reported next to a **theoretical ceiling computed from the bytes that must cross the memory bus**, and compared against PyTorch and llama.cpp on the same GPU.

> **Starting hypothesis:** at batch=1, autoregressive decoding is dominated by matrix-vector products (GEMV). GEMV is memory-bound, so moving fewer bytes per token (quantization) should matter more than raw FLOPs. The measurements below test this kernel by kernel and end to end. The answer is more nuanced than the hypothesis, and the numbers show why.

---

## Results at a Glance

NVIDIA RTX 5070 Ti (896 GB/s), Qwen2.5-Coder-0.5B, batch=1, greedy decoding. All numbers come from `results/*.json`.

| System | TPOT p50 | tok/s | % of bandwidth ceiling | Peak VRAM | HumanEval pass@1 |
|--------|---:|---:|---:|---:|---:|
| PyTorch eager, bf16 | 14.61 ms | 68 | 7.5% | 997 MB | 24.4% |
| **This engine, INT4 (S=512)** | **56.09 ms** | **17.8** | **0.55%** | **599 MB** | **14.0%** |
| llama.cpp F16 (depth 512) | 2.20 ms | 454 | 50% | — | — |
| llama.cpp Q4_0 (depth 512) | 1.33 ms | 754 | 29% | — | — |
| llama.cpp Q4_K_M (depth 512) | 1.39 ms | 719 | 31% | — | — |

HumanEval for the INT4 row is the same INT4 weights and kernels running inside Hugging Face's `generate()` ([Quantization vs. Quality](#quantization-vs-quality)).

What the data shows:

1. **The engine is correct wherever it is tested.** Every kernel matches a CPU/PyTorch reference on random data. The full INT4 engine matches Hugging Face running the same quantized weights within its own fp32 noise (tested against float64), and it generates coherent code. Speculative decoding produces exactly the greedy output on every prompt.
2. **INT4 is not free in quality.** Plain round-to-nearest INT4 (g=128, lm_head included) drops HumanEval pass@1 from **24.4% to 14.0%**, i.e. from 40 to 23 of 164 problems (−43% relative). That is the trade-off this project set out to measure, and it makes calibrated quantization a v2.0 priority.
3. **The hypothesis holds for the dense fp32 kernel.** The optimized fp32 GEMV reaches **58% of DRAM bandwidth** (521 GB/s). The INT4 GEMV does not yet turn its 7.5× byte reduction into speed: it is only **1.09× faster than fp32**, and it takes the same time with L2 hot or cold, which rules out DRAM as its bottleneck. The likely cause is how it reads the activation vector.
4. **v1.0 is correct but slow.** The engine is 3.8× slower than PyTorch eager and 42× slower than llama.cpp Q4_0. The measurements point to two causes:
   - An estimated **70% of each decoder block** is the Flash-Decoding loop, which walks every 256-token chunk sequentially, head by head.
   - Every one of the 46 kernel launches per block is followed by a `cudaDeviceSynchronize()`.
5. **Even llama.cpp is far from its ceiling at 0.5B.** Q4_0 moves 2.85× fewer bytes than F16 but is only 1.66× faster, at 29% of its bandwidth ceiling. For a model this small, fixed per-token costs are a large share of latency; bandwidth is only part of the story.
6. **Speculative decoding only pays off if a multi-token forward is cheap.** Prompt Lookup Decoding is lossless, but even with 92% draft acceptance it gives only **1.12×**, and **1.00× overall** across the four prompts. In this engine a 5.7-token verify forward costs 4.75× a single-token forward, because attention is processed token by token.

The performance fixes and the quality controls are the [v1.1 roadmap](#v11--performance-pass); the v1.0 numbers above stay as the baseline.

---

## Table of Contents

- [Why This Project Exists](#why-this-project-exists)
- [Architecture](#architecture)
- [How Correctness Is Verified](#how-correctness-is-verified)
- [Implemented Levels](#implemented-levels)
  - [Level 0 — PyTorch Baseline](#level-0--pytorch-baseline)
  - [Levels 1–3 — GEMV Kernels (naive, optimized, INT4)](#levels-13--gemv-kernels-naive-optimized-int4)
  - [Level 4 — Flash-Decoding](#level-4--flash-decoding)
  - [Level 5 — C++ Transformer Engine](#level-5--c-transformer-engine)
  - [Level 6 — Prompt Lookup Decoding](#level-6--prompt-lookup-decoding)
- [Reference: llama.cpp](#reference-llamacpp)
- [Quantization vs. Quality](#quantization-vs-quality)
- [Benchmarking Methodology](#benchmarking-methodology)
- [Scope of v1.0](#scope-of-v10)
- [Roadmap](#roadmap)
- [Hardware and Software](#hardware-and-software)
- [Setup & Usage](#setup--usage)
- [Portfolio Context](#portfolio-context)

---

## Why This Project Exists

Most "optimized inference" portfolio projects look alike. They show an isolated `tok/s` number with no hardware context, treat quantization as free in quality, and apply generic optimizations with no link to a product. CodeAlign-Runtime differs on three axes:

1. **Every speed number is anchored to a ceiling:** bytes read per token / memory bandwidth = minimum possible latency ([`scripts/roofline.py`](scripts/roofline.py)).
2. **Quantization is measured as a trade-off.** Speed is reported *and* quality is measured: HumanEval pass@1 of bf16 vs. INT4, with the same evaluation harness used in [CodeAlign](https://github.com/Sergasgr/CodeAlign).
3. **Correctness is tested, not assumed.** Every kernel is checked against a CPU/PyTorch reference, and the whole engine against Hugging Face's Qwen2 implementation.

The target use case is IDE code completion: short prompts, batch=1, and the latency per token is what the user feels.

---

## Architecture

```
CodeAlign-Runtime/
├── src/
│   ├── gemv/                     # GEMV kernels: naive, optimized (fp32), INT4 naive/optimized
│   ├── gemm/                     # GEMM kernels: naive, tiled (fp32), INT4 naive/tiled/Split-K
│   ├── ops/                      # Flash-Decoding, RMSNorm, RoPE, SwiGLU, residual/bias, KV-cache, argmax
│   ├── transformer/              # Level 5: QwenBlock (structs, memory, forward, pybind11 binding)
│   └── speculative/              # Level 6: n-gram oracle for Prompt Lookup Decoding
├── benchmarks/
│   ├── gemv_benchmark.cpp        # C++ harness: validation + L2-hot/L2-cold timing of the GEMV kernels
│   ├── gemm_benchmark.cpp        # C++ harness: same for the GEMM kernels (incl. Split-K)
│   ├── benchmark_utils.{h,cpp}   # CUDA_CHECK, percentiles, L2 flush, INT4 quantization, CPU references, JSON
│   ├── gemv_binding.cpp          # PyTorch binding: INT4 GEMV + Flash-Decoding
│   └── gemm_binding.cpp          # PyTorch binding: INT4 GEMM
├── scripts/
│   ├── baseline.py               # Level 0: PyTorch eager TTFT/TPOT/VRAM
│   ├── flash_decoding_baseline.py# Level 4: validation + benchmark vs PyTorch attention
│   ├── engine.py                 # Level 5: CodeAlignEngine (24 QwenBlocks + embedding/norm/lm_head glue)
│   ├── transformer_inference.py  # Level 5: block / full-model / end-to-end decode benchmark
│   ├── generate_speculative.py   # Level 6: Prompt Lookup Decoding + lossless check
│   ├── quantization.py           # INT4 quantization + QuantizedLinearINT4 (drop-in nn.Linear)
│   ├── evaluate_quality.py       # HumanEval pass@1, bf16 vs INT4 (bigcode-evaluation-harness)
│   ├── llama_cpp_benchmark.py    # llama.cpp reference (F16, Q4_0, Q4_K_M) via llama-bench
│   ├── roofline.py               # bandwidth ceilings (bytes / 896 GB/s)
│   ├── report.py                 # results/*.json -> the markdown tables of this README
│   ├── results_io.py             # JSON results + environment (GPU, driver, versions, git commit)
│   ├── inference_config.py       # model id, dimensions, benchmark constants
│   └── baseline_config.py        # Level 0 prompt and iteration counts
├── tests/                        # pytest: CPU tests + GPU tests (kernels, engine vs Hugging Face)
├── results/                      # JSON output of every benchmark (committed)
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
| GEMV/GEMM kernels (fp32 and INT4, incl. Split-K), several shapes | CPU with double accumulation on **random data**. INT4 kernels are compared against the *dequantized* weights, so a kernel bug cannot hide inside the quantization error. Outputs are pre-filled with NaN, so an output the kernel never writes fails. | `benchmarks/*_benchmark.cpp`, `tests/test_kernels_cuda.py` |
| Flash-Decoding | PyTorch softmax attention with standard-normal inputs, S from 1 to 4096 (incl. non-multiples of the chunk), plus a dominant key in the first and last chunks. The script checks that its data would **fail a kernel that ignored the scores** (one returning mean(V)). With uniform [0,1) inputs the softmax is nearly flat, and that shortcut would pass a 1e-2 threshold. | `scripts/flash_decoding_baseline.py`, `tests/test_kernels_cuda.py` |
| One decoder block | Hugging Face `Qwen2DecoderLayer` with the same INT4 weights dequantized, in float64 | `tests/test_engine_cuda.py` |
| Full engine | Hugging Face logits at every position, and every greedy token (teacher forcing), in float64 | `tests/test_engine_cuda.py` |
| Prompt Lookup Decoding | Must produce exactly the greedy output. The KV-cache length after rollbacks is checked. | `tests/test_engine_cuda.py`, `scripts/generate_speculative.py` |
| INT4 packing, `nn.Linear` drop-in, roofline | Bit-level layout, round-trip error ≤ half a step, CPU emulation of the kernels' index math | `tests/test_quantization.py`, `tests/test_cpu_emulation.py` (no GPU needed) |

The engine tolerance is calibrated, not hard-coded: the engine's error against Hugging Face in float64 must stay within 10× of Hugging Face's own fp32 error. Real Qwen2.5 weights are ill-conditioned, so a fixed threshold would be either too loose or too tight. The engine tests run on three models:

- a small random-weight Qwen2 (same architecture, smaller dimensions);
- the same small model with large q/k biases, which reproduces Qwen2.5's conditioning;
- the real Qwen2.5-Coder-0.5B (`-m "not real_model"` skips the download).

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

The kernels are benchmarked on the MLP up/gate projection of the model (4864×896):

- **Level 1, naive:** one thread per output row. Adjacent threads read different rows, so each 32-thread load touches 32 cache lines.
- **Level 2, optimized:** one warp per row (coalesced), `float4` loads (16 B per instruction), and a warp-shuffle reduction (`__shfl_down_sync`).
- **Level 3, INT4:** weights quantized ahead of time and dequantized inside the kernel, so fp32 weights are never materialized.
  - **Scheme:** symmetric per-group INT4 with group size 128 and one fp32 scale = `max(|w|) / 7` per group. Values are rounded to nearest (RTN, no calibration) in [-7, 7] and packed 8 per `uint32_t`; the Python quantizer, the C++ harness and every kernel share this bit layout, and a test checks it.
  - **Size:** 0.53 bytes per weight, i.e. **3.76× fewer bytes than bf16 and 7.5× fewer than fp32**.
  - **Variants:** `gemv_int4_naive_kernel` uses one thread per row. `gemv_int4_optimized_kernel` uses one warp per row, `uint4` loads (32 weights per instruction) and a warp shuffle.

Every kernel is timed twice. **L2-hot** means repeated runs over the same data; the 17.4 MB fp32 matrix fits in the 48 MB L2, so reads can exceed DRAM bandwidth. **L2-cold** means a 96 MB buffer is written before each timed run, so operands come from DRAM. Only L2-cold numbers are compared with the 896 GB/s ceiling. Timings include each wrapper's launch and synchronization.

| Kernel | L2-hot p50 | L2-hot GB/s | L2-cold p50 | L2-cold GB/s | % of 896 GB/s (cold) |
|--------|---:|---:|---:|---:|---:|
| Level 1 — fp32 naive | 93.0 µs | 187.7 | 97.9 µs | 178.4 | 19.9% |
| Level 2 — fp32 optimized | 14.6 µs | 1197.6 | 33.5 µs | **521.0** | **58.1%** |
| Level 3 — INT4 naive | 34.8 µs | 67.2 | 57.8 µs | 40.5 | 4.5% |
| Level 3 — INT4 optimized | 31.3 µs | 74.8 | 30.8 µs | 75.8 | 8.5% |

What this shows:

- **Coalescing + vectorization work.** Going from Level 1 to Level 2 is 2.9× from DRAM and 6.4× from L2. The optimized fp32 kernel reaches 58% of the spec bandwidth, and 1.2 TB/s when the matrix is L2-resident.
- **The INT4 kernel is not limited by memory.** It moves 2.3 MB instead of 17.5 MB, yet it is only **1.09× faster** than fp32 from DRAM (30.8 vs 33.5 µs). It takes the same time with L2 hot or cold (31.3 vs 30.8 µs), which rules out DRAM as the bottleneck.
- **The likely culprit is the activation-vector access pattern.** The packed weights are read coalesced (lane *L* loads the *L*-th `uint4` of the row), but lane *L* then reads `vec[32·L … 32·L+31]` one float at a time. Each of those load instructions touches a different cache line per lane. On top of that, an 896-wide row has only 28 `uint4`s, so 4 of the 32 lanes sit idle. Fixing this is item 3 of [v1.1](#v11--performance-pass).

**GEMM kernels** (16 tokens × 4864×896) are used for multi-token forwards: speculative verification and chunked prefill. Every one of them validates on three shapes, including 11 and 3 tokens.

| Kernel | L2-hot p50 | L2-cold p50 | L2-cold GB/s | % of 896 GB/s (cold) |
|--------|---:|---:|---:|---:|
| fp32 naive | 239.7 µs | 244.7 µs | 72.8 | 8.1% |
| fp32 tiled (shared memory 16×16) | 60.7 µs | 75.9 µs | 234.4 | 26.2% |
| INT4 naive | 52.0 µs | 67.3 µs | 39.9 | 4.5% |
| INT4 tiled | 57.5 µs | 62.0 µs | 43.3 | 4.8% |
| INT4 Split-K (engine path) | 59.1 µs | 60.4 µs | 44.5 | 5.0% |

The INT4 GEMMs are only 1.1–1.3× faster than the fp32 tiled GEMM from DRAM, and no faster with L2 hot. When a tile is loaded, each thread reads a whole 32-bit word to extract a single 4-bit weight, so every packed word is fetched 8 times.

---

### Level 4 — Flash-Decoding

**Files:** [`flash_decoding_partial.cu`](src/ops/flash_decoding_partial.cu), [`flash_decoding_final.cu`](src/ops/flash_decoding_final.cu)

During decode, one query attends to a growing KV-cache, so there is nothing to parallelize over in the query dimension. **Flash-Decoding** parallelizes over the cache instead:

1. **Partial kernel:** the cache is split into chunks of 256 tokens, one block per chunk. The block computes Q·Kᵢ with a warp-shuffle reduction and accumulates the value vectors with an **online softmax** (running max and sum), so the attention matrix is never materialized. It outputs a normalized partial vector and its log-sum-exp.
2. **Final kernel:** merges the chunks with weights `exp(lse_chunk − lse_global)`.

**Validation:** 30/30 cases pass (3 seeds × 9 sequence lengths + 3 dominant-key cases), with a worst error of 1.5e-7 (tolerance 1e-4).

**Benchmark:** single head, head_dim 64, fp32, against PyTorch's `Q @ Kᵀ → softmax → @ V`. The ceiling is the time to read K and V once.

| Seq length | PyTorch p50 | Custom p50 | PyTorch / Custom | Custom % of ceiling | PyTorch % of ceiling |
|---:|---:|---:|---:|---:|---:|
| 256 | 0.098 ms | 0.151 ms | 0.65× | 0.1% | 0.1% |
| 4,096 | 0.100 ms | 0.155 ms | 0.64× | 1.5% | 2.4% |
| 16,384 | 0.105 ms | 0.155 ms | 0.68× | 6.0% | 8.9% |
| 65,536 | 0.106 ms | 0.164 ms | 0.65× | 22.8% | 35.2% |
| 131,072 | 0.123 ms | 0.273 ms | 0.45× | 27.4% | 61.0% |
| 262,144 | 0.270 ms | 0.307 ms | 0.88× | 48.8% | 55.5% |

- **Short contexts are pure overhead.** Up to 65k tokens the custom kernel stays at ~0.15–0.16 ms: two launches, two device synchronizations, and a block that walks its 256 tokens one at a time with two `__syncthreads()` per token.
- **At 262k tokens it reaches 49% of the bandwidth ceiling,** within 12% of PyTorch/cuBLAS. It never overtakes PyTorch, and at 131k it drops to 0.45× before recovering.

---

### Level 5 — C++ Transformer Engine

**Files:** [`transformer.h`](src/transformer/transformer.h), [`transformer.cpp`](src/transformer/transformer.cpp), [`memory.cpp`](src/transformer/memory.cpp), [`transformer_binding.cpp`](src/transformer/transformer_binding.cpp), [`engine.py`](scripts/engine.py), and the ops in [`src/ops/`](src/ops)

Every kernel from the previous levels is assembled into the full Qwen2.5-Coder-0.5B decoder.

**One decoder block in C++ (`QwenBlock`):**

```
RMSNorm → INT4 q/k/v projections (+ bias) → RoPE (θ = 10⁶) → KV-cache append
→ GQA Flash-Decoding (14 query heads share 2 KV heads) → INT4 o_proj → residual
→ RMSNorm → INT4 gate/up → SwiGLU → INT4 down → residual
```

- **Qwen2 specifics:** grouped-query attention (k/v project 896 → 128; query head *h* reads KV head *h / 7*) and a bias on q/k/v.
- **Memory:** weights are raw GPU pointers owned by the binding, the KV-cache is `[kv_heads, max_seq_len, head_dim]`, and every intermediate buffer is allocated once in the constructor. **The C++ blocks never allocate during inference.**
- **Multi-token forwards:** up to 11 tokens per call (1 + 10 draft tokens). Projections switch from the INT4 GEMV to the INT4 Split-K GEMM. Attention is causal inside the call: token *t* sees the cache plus tokens ≤ *t*.
- **Safety:** shapes, dtypes, token count, KV-cache overflow and rollback range are checked, and CUDA errors surface as Python exceptions.
- **Numerics:** the extensions are compiled with `-use_fast_math`, which turns `powf`/`sinf`/`cosf` into approximate hardware instructions (`sin.approx`, `cos.approx`, `lg2.approx`, `ex2.approx` in the PTX). In RoPE that means angle errors of ~1e-5 rad.
  - Qwen2.5's q/k biases are large, so |q|·|k| is large, and that tiny error drifted the engine from Hugging Face by ~1.6% after one layer and ~12% in the logits. The engine tests caught it.
  - RoPE therefore follows Hugging Face's fp32 angle (`fp32(pos × inv_freq)`) and evaluates sin/cos in double precision. Double-precision math is not affected by `-use_fast_math`, and the cost is negligible.

**The full model (`CodeAlignEngine`, [`engine.py`](scripts/engine.py))** is 24 `QwenBlock`s plus small Python glue: the embedding row lookup, the final RMSNorm, the lm_head on the same INT4 kernels, and a GPU argmax. Prompts longer than 11 tokens are prefilled in chunks through the same causal path. At load time the engine checks that the model's config matches what the kernels hard-code.

```python
from scripts.engine import CodeAlignEngine

engine = CodeAlignEngine.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")
new_tokens = engine.generate(prompt_ids, max_new_tokens=128, eos_token_ids=[tokenizer.eos_token_id])
```

**Benchmark** ([`transformer_inference.py`](scripts/transformer_inference.py)): decode latency at fixed context lengths. Before each timed step the KV-cache length is set to S−1, so every sample is exactly one decode step at context S. Ceilings include the KV-cache bytes read at S.

| Context S | Block p50 | Block % of ceiling | Model p50 | Model tok/s | Model % of ceiling |
|---:|---:|---:|---:|---:|---:|
| 128 | 1.36 ms | 0.66% | 33.58 ms | 29.8 | 0.88% |
| 512 | 2.09 ms | 0.45% | 56.09 ms | 17.8 | 0.55% |
| 1,024 | 2.09 ms | 0.48% | 56.03 ms | 17.8 | 0.57% |
| 2,048 | 2.09 ms | 0.54% | 56.02 ms | 17.9 | 0.62% |

Ceilings: one block reads 7.93 MB (9–11 µs); the full model reads 263 MB (0.30–0.35 ms), with the lm_head in INT4.

**End to end:** a 53-token prompt gives TTFT 553.7 ms (chunked prefill) and TPOT p50 31.5 ms over 128 generated tokens. The continuation is coherent code:

```python
    with open(path, encoding="utf-8") as file:
        users = json.load(file)
        users = [User(**user) for user in users]
    return users
```

**Memory:** 598 MB resident (511 MB of tensors, including the bf16 embedding table, plus 88 MB of C++ buffers), and a 599 MB peak during decode. That is 40% less than PyTorch's 997 MB peak. The one-off INT4 quantization at load peaks at 2.27 GB; a pre-quantized checkpoint (v2.0, track B) removes it.

**Where the time goes:**

- **Attention (~70% of the block, estimated):** block time is flat from S=512 to S=2048 because chunks run in parallel, but each block walks its 256 tokens sequentially. At S=128 the loop is 128 tokens long, and the rest of the block is identical. The difference (0.73 ms for 128 tokens × 14 heads) puts the loop at ~0.41 µs per (head, cached token), i.e. about **1.46 ms of the 2.09 ms block**.
- **Everything else (~0.63 ms):** the remaining 18 launches (projections, norms, RoPE, KV append, residuals). Like the 28 attention launches, each is followed by `cudaDeviceSynchronize()`.
- **Full model:** 24 blocks account for ~50 ms of the 56 ms per token. The lm_head, final norm, embedding and argmax add the rest.
- **TTFT is slow** because prefill goes through the 11-token decode path: every prompt token runs its own per-head attention launches.

These measurements define [v1.1](#v11--performance-pass).

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
| refactor (type hints) | 21.8 | 24.4 | 1.12× | 91.6% | 5.33 | ✅ |
| C++ getters/setters | 25.3 | 20.0 | 0.79× | 36.6% | 1.33 | ✅ |
| unit tests (69 tokens, EOS) | 29.9 | 30.4 | 1.01× | 80.0% | 2.09 | ✅ |
| open-ended prompt | 27.4 | 32.3 | 1.18× | 97.4% | 3.66 | ✅ |

- **Lossless:** the output is identical to plain greedy decoding on every prompt.
- **The speedup is capped by the verify cost.** A forward with 5.7 tokens costs 4.75× a single-token forward (218 vs 46 ms), and one with 1.9 tokens costs 1.68×. Attention runs token by token, so verifying k tokens costs almost k decode steps. When most drafts are rejected (C++ getters/setters), PLD is slower than greedy. PLD pays off when a multi-token forward costs about the same as a single-token one, which is what v1.1 targets.
- **High acceptance is partly degenerate.** Under greedy decoding the 0.5B base model falls into repetition loops in the refactor and open-ended prompts (see `results/level6_speculative.json`), and PLD copies loops very well. The unit-test and C++ prompts are closer to real completions.
- **Overall:** 1.00× across the four prompts (total greedy time / total PLD time).

---

## Reference: llama.cpp

Same model, same GPU, measured with `llama-bench` ([`llama_cpp_benchmark.py`](scripts/llama_cpp_benchmark.py), llama.cpp build `0253fb21f`, default settings with flash attention off, 128 generated tokens after a KV-cache prefill of the given depth, 5 repetitions). Q4_0 (4-bit, one fp16 scale per 32 weights) is the format closest to this project's INT4 (g=128, fp32 scale). Q4_K_M is llama.cpp's usual default.

| GGUF | Size | Depth 128 | Depth 512 | Depth 1,024 | Depth 2,048 | % of ceiling (512) |
|------|---:|---:|---:|---:|---:|---:|
| F16 | 988 MB | 2.15 ms | 2.20 ms | 2.06 ms | 2.13 ms | 50% |
| Q4_0 | 346 MB | 1.31 ms | 1.33 ms | 1.35 ms | 1.43 ms | 29% |
| Q4_K_M | 392 MB | 1.38 ms | 1.39 ms | 1.42 ms | 1.49 ms | 31% |

The ceiling is the GGUF's weight bytes / 896 GB/s. Q4_0 moves 2.85× fewer bytes than F16 but is only 1.5–1.66× faster. At 0.5B parameters, even a mature engine spends a large share of each token on fixed costs such as launches, small-matrix inefficiency and sampling. Bandwidth explains the dense case well (50% of the ceiling); it explains INT4 much less.

---

## Quantization vs. Quality

INT4 moves 3.76× fewer bytes than bf16. Does it cost the model correct code?

[`evaluate_quality.py`](scripts/evaluate_quality.py) runs **HumanEval pass@1 (greedy, one sample per problem)** with `bigcode-evaluation-harness`, the same harness used in [CodeAlign](https://github.com/Sergasgr/CodeAlign). It runs twice: once on the bf16 model, and once with every `nn.Linear`, lm_head included, running on this project's INT4 kernels. INT4 prefill uses the INT4 GEMM and decode the INT4 GEMV.

| Model | HumanEval pass@1 |
|-------|:---:|
| Qwen2.5-Coder-0.5B bf16 (reference) | 24.4% (40/164) |
| Qwen2.5-Coder-0.5B INT4 g=128, RTN, lm_head INT4 (our kernels) | **14.0% (23/164)** |

The absolute value depends on the harness and prompt format (bigcode's HumanEval prompts, greedy decoding, 512 tokens max). The comparison that matters is bf16 vs INT4 under identical conditions. The INT4 run uses exactly the engine's packed weights, while activations between layers stay in Hugging Face's bf16 path.

**INT4 costs 10.4 points (−43% relative).** Re-running HumanEval's tests on the saved generations (`results/humaneval_generations_*.json`) gives the detail:

- **Per problem:** 20 problems are solved by both, 20 only by bf16 and 3 only by INT4.
- **INT4 often does not stop cleanly.** After finishing the function it emits long runs of blank lines and then starts new text, which the 512-token limit cuts off. 64 of 164 INT4 generations contain a run of 50+ whitespace characters, and none of the bf16 generations do. Syntax errors go from 12 to 50.
- **Most of the loss is in the code itself, not in the stopping.** Scoring only the generated function body (cut at the first top-level line) gives 26/164 for INT4 and still 40/164 for bf16.

**Why this is the quantization and not the kernels:** the engine matches Hugging Face running the dequantized INT4 weights at every position of the real model, and the `nn.Linear` drop-in matches the dequantized weights (see [How Correctness Is Verified](#how-correctness-is-verified)). So 14.0% is what this INT4 checkpoint scores. The scheme is the simplest one: round-to-nearest with no calibration, symmetric (15 of the 16 levels), one fp32 scale per 128 weights, and the 151,936×896 lm_head quantized too. A direct control run and an lm_head ablation are [v1.1](#v11--performance-pass) items; better quantization is [v2.0, track C](#v20--triton-a-python-free-runtime-and-faster-prefill).

---

## Benchmarking Methodology

1. **CUDA events** around every timed step, and **warmup discarded**.
2. **Distributions, not anecdotes:** p50/p90/p99 over ≥ 90 samples. p50 is the headline number.
3. **A ceiling next to every latency:** (weights + KV-cache bytes that must be read) / 896 GB/s, from [`roofline.py`](scripts/roofline.py).
4. **DRAM vs. L2:** isolated kernels are timed with L2 hot and with L2 flushed. Only L2-cold numbers are compared with DRAM bandwidth.
5. **Fixed context:** decode latency is measured at a fixed KV-cache length, not averaged over a growing context.
6. **Correctness before speed:** if validation fails, no timings are reported. The C++ harnesses and the Flash-Decoding script write only the failed checks and exit non-zero.
7. **Traceable results:** every run writes `results/<name>.json` with the GPU, driver, CUDA/torch/transformers versions and git commit. [`report.py`](scripts/report.py) turns them into the tables above.

---

## Scope of v1.0

- **One model family:** the kernels hard-code Qwen2.5 choices (RoPE θ = 10⁶, RMSNorm ε = 10⁻⁶, hidden ≤ 1024), and the engine refuses other configs at load time.
- **Batch = 1, fp32 activations and KV-cache.** Only the weights are INT4.
- **Decode first:** prefill reuses the 11-token verify path. TTFT is reported but not optimized.
- **One GPU:** every number comes from a single RTX 5070 Ti (sm_120). The code builds for the local architecture (`CMAKE_CUDA_ARCHITECTURES native`).

---

## Roadmap

### v1.1 — Performance pass

The goal is a before/after comparison with the same harness, tests and scripts; the v1.0 numbers stay at the `v1.0.0` tag. Each item comes from a measurement above.

| # | Change | Evidence in v1.0 | Judged by |
|---|--------|------------------|-----------|
| 0 | **Profile first:** an Nsight Systems timeline of one decode step and Nsight Compute reports for the INT4 GEMV and the Flash-Decoding kernel, committed as the v1.0 profile | Attention share and INT4 diagnosis are estimates from timings | Profiles before and after each item |
| 1 | **Batched GQA Flash-Decoding (C++/CUDA):** one launch per layer over (KV chunks × KV heads). Each block loads a K/V chunk once and serves the 7 query heads that share it. Tokens inside a chunk are processed in parallel, not with a sequential loop, and the chunk size adapts to short contexts. The multi-token (verify/prefill) path uses a single launch for all tokens, with per-token causal lengths. | Attention ≈ 70% of block time; block flat from S=512 because each block walks 256 tokens sequentially; a 5.7-token verify costs 4.75× a single token | Block and model TPOT at S = 128…2048; Level 4 table; PLD tokens/forward vs speedup |
| 2 | **No device synchronization per launch (C++):** stream-ordered launches, with errors checked once per forward | 46 launches per block per token, each followed by `cudaDeviceSynchronize()` | Block time at S=128; model TPOT |
| 3 | **Coalesced INT4 GEMV (CUDA):** stage the activation vector in shared memory once per block, so each lane reads its 32 activations without scattered global loads. Map lanes so that no lane idles on 896-wide rows (e.g. several rows per warp). | INT4 only 1.09× faster than fp32 with 7.5× fewer bytes; hot ≈ cold (not DRAM-bound) | Level 3 table (L2-cold GB/s); Nsight Compute memory metrics |
| 4 | **Before/after reporting:** `report.py --baseline v1.0.0` renders v1.0 and v1.1 side by side | — | README tables |
| 5 | **Quality controls (Python):** (a) the same HumanEval run with the dequantized weights in plain `nn.Linear`, which separates the kernels from the quantization; (b) lm_head kept in bf16 (`--keep-lm-head`), saved under its own result name | INT4 pass@1 14.0% vs 24.4% | pass@1 of each run; (b) also reports its bytes: a bf16 lm_head is 272 MB vs 72 MB in INT4, which moves the weights-only ceiling from 0.29 to 0.52 ms/token |

**Target:** a decode TPOT below PyTorch eager (14.6 ms) at every measured context, i.e. at least 3.8× faster than v1.0 at S ≥ 512, with the same tests passing and the same INT4 checkpoint (identical HumanEval generations).

### v2.0 — Triton, a Python-free runtime, and faster prefill

Three tracks. Each one mixes Python and C++ and uses the same validation and benchmark methodology.

**A. Triton vs. hand-written CUDA**

- Re-implement the hot kernels in [Triton](https://github.com/triton-lang/triton): INT4 GEMV, batched GQA Flash-Decoding, and fused residual + RMSNorm. Each goes through the same pytest references and the same L2-hot/cold harness, so the comparison is CUDA C++ vs Triton vs PyTorch at identical shapes.
- **Bridge into the C++ engine:** compile the Triton kernels ahead of time with `triton.tools.compile`. It generates C sources that embed the cubin and expose a launcher (`CUresult kernel(CUstream, gridX, gridY, gridZ, args...)`). Link them into `QwenBlock` and select the backend per op (`cuda` / `triton`). The comparison then covers full-engine TPOT, not only microbenchmarks.
- **Done when:** every Triton kernel passes the existing tests, and there is a per-kernel and end-to-end table per backend.

**B. Python-free C++ runtime**

- **Checkpoint:** a pre-quantized format (packed INT4 + scales + metadata, written once by Python) loaded directly by C++.
- **Tokenizer in C++:** e.g. [`mlc-ai/tokenizers-cpp`](https://github.com/mlc-ai/tokenizers-cpp), which wraps Hugging Face tokenizers.
- **CUDA Graphs:** capture the decode step as a graph. The KV-cache length has to live in device memory so that the same graph can be replayed.
- **`codealign-cli`:** streaming completion with fill-in-the-middle prompts (`<|fim_prefix|>…<|fim_suffix|>…<|fim_middle|>`, supported by Qwen2.5-Coder), which is the actual IDE completion format.
- **Done when:** the CLI's TTFT/TPOT are measured with the same methodology and compared with the Python-driven engine and llama.cpp.

**C. Precision and prefill**

- **bf16 activations and KV-cache** (`__nv_bfloat162` math), halving the KV traffic.
- **A tensor-core INT4 GEMM for prefill** (`mma.sync`, INT4 weights dequantized to bf16 in registers), with a dedicated prefill path. TTFT is 553.7 ms today vs 15.9 ms in PyTorch.
- **Better INT4, measured as bytes vs. pass@1.** v1.0 loses 10.4 HumanEval points, so this is the first quality lever. Three steps, each reported as bytes/weight next to pass@1:
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
| Software | transformers 5.15.1, bigcode-evaluation-harness `8fc5bae`, llama.cpp `0253fb21f` |

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
uv run python -m tests.test_cpu_emulation   # CPU emulation report (no GPU)
```

### Benchmarks

Run these from the repository root; each writes `results/<name>.json`.

```bash
./build/gemv_benchmark                                  # Levels 1-3
./build/gemm_benchmark                                  # GEMM kernels (incl. Split-K)
uv run python -m scripts.baseline                       # Level 0
uv run python -m scripts.flash_decoding_baseline        # Level 4
uv run python -m scripts.transformer_inference          # Level 5
uv run python -m scripts.generate_speculative           # Level 6
uv run python -m scripts.evaluate_quality --precision bf16
uv run python -m scripts.evaluate_quality --precision int4
LLAMA_CPP_DIR=/path/to/llama.cpp uv run --with gguf --with sentencepiece python -m scripts.llama_cpp_benchmark
uv run python -m scripts.report                         # markdown tables from results/*.json
```

`HF_TOKEN` is optional: the model and HumanEval are public. If you need one, copy `.env.example` to `.env`; it is passed to Docker with `--env-file .env`.

---

## Portfolio Context

| Project | Demonstrates |
|---------|-------------|
| [**CodeAlign**](https://github.com/Sergasgr/CodeAlign) | Data curation → SFT → DPO → HumanEval evaluation of a coding LLM |
| **CodeAlign-Runtime** | INT4 quantization → CUDA kernels → a full decode engine, measured against hardware ceilings and llama.cpp |

Together they cover the path from training data to serving. v1.0 serves the public Qwen2.5-Coder-0.5B. Serving CodeAlign's own post-trained checkpoint, once distilled to this size, is future work.

---

## License

MIT
