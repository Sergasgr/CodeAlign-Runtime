# CodeAlign-Runtime

**Low-latency C++/CUDA inference engine for small code models.**

Hand-written CUDA kernels serve Qwen2.5-Coder-0.5B, a code completion model, at batch=1, which is the IDE autocompletion case. The project is built in levels. Each level attacks the same bottleneck, memory bandwidth, from a different angle. Every latency is reported next to a **theoretical ceiling computed from the bytes that must cross the memory bus**, never as a percentage in a vacuum.

> **Core thesis:** at batch=1, autoregressive decoding is dominated by matrix-vector products (GEMV). GEMV is **memory-bound**, not compute-bound. Moving fewer bytes per token (quantization) matters more than raw FLOPs. That is why GGUF and llama.cpp exist. This project tests that claim empirically, one kernel at a time, up to a full engine compared against PyTorch and llama.cpp.

> ⏳ **Results pending.** The v1.0 code is complete and tested. The cells marked ⏳ are filled from `results/*.json` after the v1.0 benchmark run (`uv run python -m scripts.report` prints every table below).

---

## Table of Contents

- [Why This Project Exists](#why-this-project-exists)
- [Architecture](#architecture)
- [How Correctness Is Verified](#how-correctness-is-verified)
- [Implemented Levels](#implemented-levels)
  - [Level 0 — PyTorch Baseline](#level-0--pytorch-baseline)
  - [Level 1 — Naive CUDA Kernel](#level-1--naive-cuda-kernel)
  - [Level 2 — Optimized Kernel](#level-2--optimized-kernel-coalescing-float4-warp-shuffle)
  - [Level 3 — INT4 Quantization + Fused Kernel](#level-3--int4-quantization--fused-kernel)
  - [Level 4 — Flash-Decoding](#level-4--flash-decoding)
  - [Level 5 — C++ Transformer Engine](#level-5--c-transformer-engine)
  - [Level 6 — Prompt Lookup Decoding](#level-6--prompt-lookup-decoding)
- [Results](#results)
- [Quantization vs. Quality](#quantization-vs-quality)
- [Benchmarking Methodology](#benchmarking-methodology)
- [Known Limitations](#known-limitations)
- [Hardware](#hardware)
- [Setup & Usage](#setup--usage)
- [Portfolio Context](#portfolio-context)

---

## Why This Project Exists

Most "optimized inference" portfolio projects look alike. They show an isolated `tok/s` number with no hardware context, treat quantization as free in quality, and apply generic optimizations with no link to a product. CodeAlign-Runtime differs on three axes:

1. **Every speed number is anchored to a ceiling:** bytes read per token / memory bandwidth = minimum possible latency ([`scripts/roofline.py`](scripts/roofline.py)).
2. **Quantization is measured as a trade-off.** Speed is reported *and* quality is measured: HumanEval pass@1 of bf16 vs. INT4, with the same evaluation harness used in [CodeAlign](https://github.com/Sergasgr/CodeAlign).
3. **Correctness is tested, not assumed.** Every kernel is checked against a CPU/PyTorch reference, and the whole engine against Hugging Face's Qwen2 implementation ([How Correctness Is Verified](#how-correctness-is-verified)).

The target use case is IDE code completion: short prompts, batch=1, latency per token is what the user feels.

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
| GEMV/GEMM kernels (fp32 and INT4, incl. Split-K), several shapes | CPU with double accumulation, **random data**. INT4 kernels are compared against the *dequantized* weights, so a kernel bug cannot hide inside the quantization error. Outputs are pre-filled with NaN, so an output the kernel never writes fails. | `benchmarks/*_benchmark.cpp`, `tests/test_kernels_cuda.py` |
| Flash-Decoding | PyTorch softmax attention, standard-normal inputs, S from 1 to 4096 (incl. non-multiples of the chunk), plus a dominant key in the last chunk. The script checks that its data would **fail a kernel that ignored the scores** (returning mean(V)). With uniform [0,1) inputs that shortcut stays under 1e-2 for S ≥ 1000. | `scripts/flash_decoding_baseline.py`, `tests/test_kernels_cuda.py` |
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

**Goal:** the reference everything else is measured against.

[`baseline.py`](scripts/baseline.py) loads Qwen2.5-Coder-0.5B in bf16 and measures two metrics that serving systems report separately:

- **TTFT (Time-To-First-Token):** prefill of the whole prompt. Dense GEMM, compute-heavy.
- **TPOT (Time-Per-Output-Token):** one decode step with the KV-cache. GEMV, memory-bound. **This is the metric the rest of the project attacks.**

CUDA events wrap each step. The first iterations are discarded, and p50/p90/p99 are reported. In eager mode at batch=1 the GPU waits on Python dispatch, so the measured TPOT includes that host overhead. That is the latency a user actually sees, and it is exactly what a C++ engine removes.

The ceiling is computed from the loaded model itself: every parameter byte is read once per token. The tied lm_head/embedding matrix counts once, because the lm_head reads it whole and the embedding lookup reads one row:

```
TPOT_min = 494 M params × 2 bytes / 896 GB/s ≈ 1.10 ms/token   (bf16)
```

**Result:** ⏳ (see [Results](#results)).

---

### Level 1 — Naive CUDA Kernel

**File:** [`gemv_naive.cu`](src/gemv/gemv_naive.cu)

One thread per output row. Each thread walks a whole row of the weight matrix:

```cuda
int row = blockIdx.x * blockDim.x + threadIdx.x;
if (row < rows) {
    float sum = 0.0f;
    for (int i = 0; i < cols; i++)
        sum += d_mat[row * cols + i] * d_vec[i];
    d_out[row] = sum;
}
```

Expected result: well below the bandwidth ceiling. This negative result is content, not failure:

- **Uncoalesced accesses:** the threads of a warp read different rows. Thread 0 reads index `0` and thread 1 reads index `896`, so each 32-thread load touches 32 different cache lines.
- **No vectorized loads:** 4 bytes per load instruction instead of 16.
- **No intra-warp parallelism** in the reduction.

Benchmarked on the MLP up/gate projection of Qwen2.5-0.5B (4864×896).

---

### Level 2 — Optimized Kernel (coalescing, float4, warp shuffle)

**File:** [`gemv_optimized.cu`](src/gemv/gemv_optimized.cu)

1. **One warp per row:** adjacent threads read adjacent addresses, so one memory transaction feeds the whole warp.
2. **`float4` loads:** 16 bytes per load instruction.
3. **Warp-shuffle reduction** (`__shfl_down_sync`): register-to-register, no shared memory round trip.

```cuda
for (int i = 0; i < 5; i++) {
    sum += __shfl_down_sync(FULL_MASK, sum, offset);
    offset /= 2;
}
```

**L2 and honest bandwidth numbers:** the fp32 matrix is 17.4 MB. That fits in the RTX 5070 Ti's 48 MB L2, so repeated runs over the same data are served from L2 and can exceed the 896 GB/s DRAM bandwidth. The harness therefore reports every kernel twice. **L2-hot** means repeated runs over the same data. **L2-cold** means a buffer of 2× the L2 size is written before each timed run, so the operands come from DRAM. Only the cold number is compared with 896 GB/s.

---

### Level 3 — INT4 Quantization + Fused Kernel

**Files:** [`gemv_quantized.cu`](src/gemv/gemv_quantized.cu), [`gemm_quantized.cu`](src/gemm/gemm_quantized.cu), [`quantization.py`](scripts/quantization.py), [`gemv_binding.cpp`](benchmarks/gemv_binding.cpp), [`gemm_binding.cpp`](benchmarks/gemm_binding.cpp)

Weights are quantized to INT4 ahead of time. The kernel dequantizes and multiplies in the same pass, without materializing fp16/fp32 weights in memory. Quantization reduces latency by moving fewer bytes per token, not by computing faster.

#### Quantization scheme

- **Symmetric per-group INT4, group size 128:** each group of 128 consecutive weights in a row shares one fp32 scale, `max(|w|) / 7`. Values are rounded to [-7, 7]; -8 is representable but never produced.
- **Packing:** 8 values per `uint32_t`, with value *k* in bits [4k, 4k+4). The same layout is used by the Python quantizer, the C++ harness and every kernel. A test checks they produce identical bits.
- **Bytes:** 0.5 B/weight plus 4 B per 128 weights = 0.53 B/weight. That is **3.76× fewer bytes than bf16** (7.5× fewer than fp32).

#### Kernels

- **`gemv_int4_naive_kernel`:** one thread per row, unpacks 8 values per word.
- **`gemv_int4_optimized_kernel`:** one warp per row, `uint4` loads (32 INT4 values per instruction), warp-shuffle reduction. Used for every single-token projection.
- **`gemm_int4_optimized` / Split-K:** tiled INT4 GEMMs for multi-token inputs (prefill, speculative verification).

#### PyTorch integration

`QuantizedLinearINT4` is a drop-in `nn.Linear` replacement. A single token (decode) goes through the INT4 GEMV. More tokens (prefill) go through the INT4 GEMM. It computes in fp32 and returns the input dtype. `replace_linear_layers(model, skip=...)` swaps every `nn.Linear` of a Hugging Face model. That is how the HumanEval evaluation runs the INT4 model through `generate()`.

#### Note on the llama.cpp comparison

This scheme (symmetric, g=128, fp32 scales) is **not** llama.cpp's Q4_K_M, which uses super-blocks with 6-bit sub-scales. The closest llama.cpp format is **Q4_0**: 4-bit with one fp16 scale per 32 weights. Both are benchmarked: Q4_0 is the like-for-like row, and Q4_K_M is the "industry default" row.

---

### Level 4 — Flash-Decoding

**Files:** [`flash_decoding_partial.cu`](src/ops/flash_decoding_partial.cu), [`flash_decoding_final.cu`](src/ops/flash_decoding_final.cu), [`gemv_binding.cpp`](benchmarks/gemv_binding.cpp)

During decode a single query attends to a growing KV-cache. There is nothing to parallelize over in the query dimension, which is what FlashAttention exploits in prefill. **Flash-Decoding** parallelizes over the KV-cache instead.

1. **Partial kernel:** the cache is split into chunks of 256 tokens, one block per chunk. The block computes Q·Kᵢ with a warp-shuffle reduction and scales by 1/√D. It accumulates the value vectors with an **online softmax**, keeping a running max and sum, so the attention matrix is never materialized. It outputs a normalized partial vector and its log-sum-exp.
2. **Final kernel:** merges the chunks with weights `exp(lse_chunk − lse_global)`.

**Validation:** 3 seeds × 9 sequence lengths (1 to 4096, including 255/256/257), plus 3 dominant-key cases. Inputs are standard-normal. Tolerance is 1e-4 in fp32, and the script verifies that the data is discriminative (see [How Correctness Is Verified](#how-correctness-is-verified)).

**Benchmark:** a single head at the model's head_dim (64), fp32, batch=1, S from 256 to 262,144, against PyTorch's `Q @ Kᵀ → softmax → @ V`.

**Result:** ⏳. The design is intentionally pedagogical, and these factors are expected to keep it behind PyTorch/cuBLAS at short contexts:

1. **cuBLAS underneath PyTorch:** its GEMMs are autotuned and use tensor cores; this kernel uses scalar warp-shuffle dot products.
2. **A sequential loop inside each chunk:** each block walks its 256 tokens one at a time, with two `__syncthreads()` per token. For S ≤ 256 there is a single block on the whole GPU.
3. **Two launches with a device synchronization between them**, instead of a fused or pipelined pair.

---

### Level 5 — C++ Transformer Engine

**Files:** [`transformer.h`](src/transformer/transformer.h), [`transformer.cpp`](src/transformer/transformer.cpp), [`memory.cpp`](src/transformer/memory.cpp), [`transformer_binding.cpp`](src/transformer/transformer_binding.cpp), [`engine.py`](scripts/engine.py), plus the ops in [`src/ops/`](src/ops)

Every kernel from the previous levels assembled into the full Qwen2.5-Coder-0.5B decoder.

#### One decoder block in C++ (`QwenBlock`)

```
RMSNorm → INT4 q/k/v projections (+ bias) → RoPE (θ = 10⁶) → KV-cache append
→ GQA Flash-Decoding (14 query heads share 2 KV heads) → INT4 o_proj → residual
→ RMSNorm → INT4 gate/up → SwiGLU → INT4 down → residual
```

- **Qwen2 specifics:** grouped-query attention. k/v project 896 → 128 (2 KV heads × 64), and query head *h* reads KV head *h / 7*. There is a bias on q/k/v.
- **Structs** ([`transformer.h`](src/transformer/transformer.h)):
  - `TransformerBlockWeights` holds raw GPU pointers to the 7 INT4 projections (+ biases) and the 2 RMSNorm vectors. The binding keeps the owning `torch::Tensor`s alive.
  - `LayerKVCache` holds `[num_kv_heads, max_seq_len, head_dim]` for K and V.
  - `LayerBuffers` holds every intermediate result.
- **Memory:** everything is allocated once in the constructor. **The C++ blocks never allocate during inference.**
- **Multi-token forwards:** up to 11 tokens (1 + 10 draft tokens) per call. Projections switch from the INT4 GEMV to the INT4 Split-K GEMM. Attention is causal inside the call: token *t* sees the cache plus tokens ≤ *t* of the same call.
- **Safety:** shapes, dtypes, token count, KV-cache overflow and rollback range are checked, and CUDA errors surface as Python exceptions.
- **Numerics:** the extensions are compiled with `-use_fast_math`, which turns `powf`/`sinf`/`cosf` into approximate hardware instructions (`sin.approx`, `cos.approx`, `lg2.approx`, `ex2.approx` in the PTX). In RoPE that means angle errors of ~1e-5 rad, and they grow with the position.
  - Qwen2.5's q/k biases are large, so |q|·|k| is large and a tiny angle error becomes a visible change in the attention scores. This drifted the engine from Hugging Face by ~1.6% after one layer and ~12% in the logits.
  - RoPE therefore follows Hugging Face's fp32 angle (`fp32(pos × inv_freq)`) and evaluates sin/cos in double precision. Double-precision math is not affected by `-use_fast_math` and costs nothing at d/2 threads per token.

#### The full model (`CodeAlignEngine`, [`engine.py`](scripts/engine.py))

24 `QwenBlock`s plus a small amount of Python glue: the embedding row lookup, the final RMSNorm, the lm_head on the same INT4 kernels, and the GPU argmax. Prompts longer than 11 tokens are prefilled in chunks through the same causal path. The engine checks at load time that the model's config matches what the kernels hard-code: RoPE θ, RMSNorm ε, SiLU, no RoPE scaling, hidden ≤ 1024.

```python
from scripts.engine import CodeAlignEngine

engine = CodeAlignEngine.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")
new_tokens = engine.generate(prompt_ids, max_new_tokens=128, eos_token_ids=[tokenizer.eos_token_id])
```

#### Benchmark

[`transformer_inference.py`](scripts/transformer_inference.py) measures decode latency at **fixed context lengths**: the KV-cache length is set to S−1 before each timed step, so every sample is exactly one decode step at context S. It does this for one block and for the full model, and ends with an end-to-end greedy generation from a real prompt. Ceilings include the KV-cache bytes read at S:

| Scope | Weights read per token | Ceiling at S=512 |
|-------|-----------------------:|-----------------:|
| 1 block (INT4, GQA) | 7.93 MB | 9.4 µs |
| Full model, INT4 incl. lm_head | 263 MB | 0.307 ms |

**Result:** ⏳.

---

### Level 6 — Prompt Lookup Decoding

**Files:** [`speculative.cpp`](src/speculative/speculative.cpp), [`generate_speculative.py`](scripts/generate_speculative.py)

Speculative decoding without a draft model. It suits code completion, where the continuation often repeats text that is already in the context: refactors, tests, repetitive members.

1. **Oracle (C++):** find the last *n* tokens (n=3) earlier in the history and propose the ≤5 tokens that followed them.
2. **Batched verify:** `[last token] + draft` go through the engine in **one forward**, using the causal multi-token path of Level 5.
3. **Accept** the longest draft prefix that matches the greedy predictions, plus one bonus token.
4. **Roll back** the rejected draft tokens from the KV-cache of all 24 layers.

Greedy PLD is lossless by construction. The script checks this on every prompt by comparing against plain greedy decoding on the same engine. It reports tokens/s with and without drafts, acceptance rate and tokens per forward, on prompts where copying helps and on an open-ended control prompt.

**Result:** ⏳.

---

## Results

All tables are generated by `uv run python -m scripts.report` from `results/*.json`. Each JSON stores the GPU, driver, library versions and git commit of the run.

### Isolated kernels — GEMV (4864×896, MLP up/gate)

| Kernel | L2-hot p50 (ms) | L2-hot GB/s | L2-cold p50 (ms) | L2-cold GB/s | % of 896 GB/s (cold) |
|--------|---:|---:|---:|---:|---:|
| Level 1 — fp32 naive | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| Level 2 — fp32 optimized | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| Level 3 — INT4 naive | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| Level 3 — INT4 optimized | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |

### Isolated kernels — GEMM (16 tokens × 4864×896)

| Kernel | L2-hot p50 (ms) | L2-hot GB/s | L2-cold p50 (ms) | L2-cold GB/s | % of 896 GB/s (cold) |
|--------|---:|---:|---:|---:|---:|
| fp32 naive | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| fp32 tiled | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| INT4 naive | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| INT4 tiled | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| INT4 Split-K (engine path) | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |

### Level 4 — Flash-Decoding (single head, D=64)

| Seq length | PyTorch p50 (ms) | Custom p50 (ms) | PyTorch / Custom | KV-read ceiling (ms) |
|---:|---:|---:|---:|---:|
| 256 → 262,144 | ⏳ | ⏳ | ⏳ | ⏳ |

### Level 5 — Engine decode latency (Qwen2.5-Coder-0.5B, INT4)

| Context S | Block p50 (ms) | Block % of ceiling | Model p50 (ms) | Model tok/s | Model % of ceiling |
|---:|---:|---:|---:|---:|---:|
| 128 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| 512 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| 1024 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| 2048 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |

### Level 6 — Prompt Lookup Decoding

| Prompt | Greedy tok/s | PLD tok/s | Speedup | Acceptance | Tokens/forward | Identical output |
|--------|---:|---:|---:|---:|---:|:---:|
| refactor (type hints) | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| C++ getters/setters | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| unit tests | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| open-ended (control) | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |

### Summary

| System | TTFT p50 (ms) | TPOT p50 (ms) | % BW ceiling | VRAM (MB) | HumanEval pass@1 |
|--------|---:|---:|---:|---:|---:|
| PyTorch eager (bf16) | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| C++/CUDA engine, INT4 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| + Prompt Lookup Decoding | — | ⏳ | — | — | same as engine (lossless) |
| llama.cpp Q4_0 (closest scheme) | — | ⏳ | — | — | — |
| llama.cpp Q4_K_M (default) | — | ⏳ | — | — | — |

---

## Quantization vs. Quality

INT4 moves 3.76× fewer bytes than bf16. Does it cost the model correct code?

[`evaluate_quality.py`](scripts/evaluate_quality.py) runs **HumanEval pass@1 (greedy, one sample per problem)** with `bigcode-evaluation-harness`, the same harness used in [CodeAlign](https://github.com/Sergasgr/CodeAlign). It runs twice: once on the bf16 model, and once with every `nn.Linear`, lm_head included, running on this project's INT4 kernels:

| Model | HumanEval pass@1 |
|-------|:---:|
| Qwen2.5-Coder-0.5B bf16 (reference) | ⏳ |
| Qwen2.5-Coder-0.5B INT4 g=128 (our kernels) | ⏳ |

The INT4 run uses exactly the same packed weights as the engine. Activations between layers stay in Hugging Face's bf16 path, so its numerics are close to the engine's but not bit-identical; the engine keeps fp32 activations.

---

## Benchmarking Methodology

1. **CUDA events** around every timed step, and **warmup discarded**.
2. **Distributions, not anecdotes:** p50/p90/p99 over ≥ 90 samples. p50 is the headline number.
3. **A ceiling next to every latency:** (weights + KV-cache bytes that must be read) / 896 GB/s, from [`roofline.py`](scripts/roofline.py).
4. **DRAM vs. L2:** isolated kernels are timed with L2 hot and with L2 flushed. Only L2-cold numbers are compared with DRAM bandwidth.
5. **Fixed context:** decode latency is measured at a fixed KV-cache length, not averaged over a growing context.
6. **Correctness before speed:** if validation fails, no timings are reported. The C++ harnesses and the Flash-Decoding script write only the failed checks and exit non-zero.
7. **Traceable results:** every run writes `results/<name>.json` with the GPU, driver, CUDA/torch/transformers versions and git commit.

---

## Known Limitations

These are deliberate v1.0 boundaries. They explain where the gap to the ceiling comes from:

- **One device synchronization per kernel launch.** Every `run_*` wrapper calls `cudaDeviceSynchronize()`. That is 43 launches per block per token, over 1,000 per token for the full model.
- **Attention runs head by head.** There are 14 heads × 2 launches per token. Each block walks its 256-token chunk sequentially, and when S ≤ 256 a single block handles the whole cache.
- **INT4 GEMV vector reads are not coalesced.** In the optimized INT4 kernel, lane *L* reads `vec[32·L … 32·L+31]`, so each load instruction touches many cache lines. This is the likely reason the INT4 kernel does not beat fp32 in isolation, even though it moves 7.5× fewer bytes. Nsight Compute reports this as uncoalesced global accesses.
- **Activations and the KV-cache are fp32:** about 1 MB of KV reads per layer per token at S=1024.
- **Prefill is not optimized.** It is chunked through the 11-token decode/verify path. The engine targets TPOT; TTFT is reported for completeness.
- **Model scope:** the kernels hard-code Qwen2.5 choices (RoPE θ = 10⁶, RMSNorm ε = 10⁻⁶, hidden ≤ 1024). The engine refuses other configs.

---

## Hardware

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX 5070 Ti (GB203, Blackwell, compute capability 12.0) |
| VRAM | 16 GB GDDR7, 256-bit |
| Memory bandwidth (spec) | 896 GB/s |
| L2 cache | 48 MB |
| CUDA | 13.0 (matches the locked `torch 2.13.0+cu130`) |

A single consumer GPU is enough for a 0.5B model. No cloud or multi-GPU needed.

---

## Setup & Usage

### Prerequisites

- NVIDIA GPU with compute capability ≥ 7.5 (CUDA 13 minimum) and a driver that supports CUDA 13.0
- CUDA Toolkit 13.x (for `nvcc`) and CMake ≥ 3.24, or Docker (below)
  - `build.sh` uses `$CUDA_HOME`, or else the first `nvcc` on `PATH`, for both CMake and the PyTorch extensions. It stops if its major version differs from torch's (13.x). With several toolkits installed (e.g. Ubuntu's `nvidia-cuda-toolkit` 12.0 in `/usr/bin`), run `CUDA_HOME=/usr/local/cuda ./build.sh`.
- Python 3.12+ and [uv](https://github.com/astral-sh/uv)

### Option 1: Docker (recommended for reproducibility)

```bash
# One-time: NVIDIA runtime for Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

docker build -t codealign-runtime .
docker run --gpus all -it --rm -v "$(pwd)":/app codealign-runtime
# inside the container:
./build.sh
```

### Option 2: Local

```bash
uv sync
./build.sh            # CMake harness + the 3 PyTorch CUDA extensions (built in place)
```

### Tests

```bash
uv run pytest                        # everything (GPU tests are skipped without a GPU/extensions)
uv run pytest -m "not real_model"    # skip the tests that download Qwen2.5-Coder-0.5B
uv run python -m tests.test_cpu_emulation   # CPU emulation report (no GPU)
```

### Benchmarks (run from the repository root; each one writes `results/<name>.json`)

```bash
./build/gemv_benchmark                                  # Levels 1-3: GEMV kernels
./build/gemm_benchmark                                  # GEMM kernels (incl. Split-K)
uv run python -m scripts.baseline                       # Level 0
uv run python -m scripts.flash_decoding_baseline        # Level 4
uv run python -m scripts.transformer_inference          # Level 5
uv run python -m scripts.generate_speculative           # Level 6
uv run python -m scripts.evaluate_quality --precision bf16
uv run python -m scripts.evaluate_quality --precision int4

# llama.cpp reference (llama.cpp built with -DGGML_CUDA=ON)
LLAMA_CPP_DIR=/path/to/llama.cpp uv run --with gguf --with sentencepiece python -m scripts.llama_cpp_benchmark

uv run python -m scripts.report                         # markdown tables for this README
```

`HF_TOKEN` is optional: Qwen2.5-Coder-0.5B is public. If you need one, copy `.env.example` to `.env`; it is passed to Docker with `--env-file .env`.

---

## Portfolio Context

| Project | Demonstrates |
|---------|-------------|
| [**CodeAlign**](https://github.com/Sergasgr/CodeAlign) | Data curation → SFT → DPO → HumanEval evaluation of a coding LLM |
| **CodeAlign-Runtime** | INT4 quantization → CUDA kernels → a full low-latency decode engine, measured against hardware ceilings |

Together they cover the path from training data to serving. v1.0 serves the public Qwen2.5-Coder-0.5B. Serving CodeAlign's own post-trained checkpoint, once distilled to this size, is future work.

---

## License

MIT
