# CodeAlign-Runtime

**Low-latency C++/CUDA inference engine for small code models.**

A progressive system of hand-written CUDA kernels to serve a code language model (Qwen2.5-Coder-0.5B) with minimal token-to-token latency for autocompletion. Each project level attacks the same bottleneck — memory bandwidth — from a different angle, and every improvement is measured against a theoretical ceiling computed from real hardware specs, not reported as an arbitrary percentage in a vacuum.

> **Core thesis:** during autoregressive generation at batch=1 (the exact case of IDE code completion), the dominant operation is a matrix-vector multiply (GEMV), **memory-bound** — not compute-bound like training. Quantization (moving fewer bytes per token) matters more than raw compute power. This is why GGUF and llama.cpp exist, and this project verifies it empirically, step by step.

---

## Table of Contents

- [Project Status](#project-status)
- [Why This Project Exists](#why-this-project-exists)
- [Architecture](#architecture)
- [Implemented Levels](#implemented-levels)
  - [Level 0 — PyTorch Baseline](#level-0--pytorch-baseline)
  - [Level 1 — Naive CUDA Kernel](#level-1--naive-cuda-kernel)
  - [Level 2 — Optimized Kernel](#level-2--optimized-kernel-coalescing-float4-warp-shuffle)
  - [Level 3 — INT4 Quantization + Fused Kernel](#level-3--int4-quantization--fused-kernel)
- [Results](#results)
- [Benchmarking Methodology](#benchmarking-methodology)
- [Quantization vs. Quality](#quantization-vs-quality)
- [Hardware](#hardware)
- [Setup & Usage](#setup--usage)
- [Roadmap](#roadmap)
- [Portfolio Context](#portfolio-context)

---

## Project Status

| Level | Status | Description |
|-------|--------|-------------|
| 0 — PyTorch Baseline | ✅ Complete | TTFT/TPOT with p50/p90/p99, theoretical ceiling, VRAM |
| 1 — Naive kernel | ✅ Complete | One thread per row, uncoalesced accesses |
| 2 — Optimized kernel | ✅ Complete | float4, warp shuffle, coalesced accesses |
| 3 — INT4 fused | ✅ Complete | Per-group quantization (g=128), dequant+matmul in-kernel |
| 4 — Flash-Decoding | 🔲 Next | Decode-phase attention (parallelization over KV-cache) |
| 5 — C++ Integration | 🔲 Planned | Full inference loop, pybind11, end-to-end benchmark |
| 6 — Prompt Lookup Decoding | 🔲 Planned | Algorithmic speculative decoding for code |

---

## Why This Project Exists

Most "optimized inference" portfolio projects look alike: an isolated `tok/s` number with no hardware context, quantization treated as if it were free in quality, and generic optimizations that would work equally well for any LLM with no connection to a real product. CodeAlign-Runtime differentiates on three axes:

1. **Every speed number is anchored to a theoretical ceiling** (model bytes / memory bandwidth = minimum possible TPOT). Nothing is presented as "good" in a vacuum.
2. **Quantization is treated as a trade-off, not a free trick**, and is measured on both sides: speed *and* quality (HumanEval pass@1, using the same evaluation harness from [CodeAlign](https://github.com/Sergasgr/CodeAlign)).
3. **The project is part of a complete portfolio chain**: data → SFT → DPO (CodeAlign) → quantization → low-latency serving, with the same person behind every link.

The target use case is IDE code completion — exactly the problem JetBrains describes: *"highly optimized, context-aware coding LLMs"*.

---

## Architecture

```
CodeAlign-Runtime/
├── src/
│   ├── gemv.h                    # Unified header — all kernel declarations
│   ├── gemv_naive.cu             # Level 1: naive GEMV kernel (one thread per row)
│   ├── gemv_optimized.cu         # Level 2: float4 + warp shuffle + coalescing
│   ├── gemv_quantized.cu         # Level 3: INT4 GEMV (naive + optimized)
│   ├── main.cpp                  # C++ benchmark harness with CUDA events
│   ├── binding.cpp               # PyTorch ↔ CUDA binding via torch::Extension
│   └── quantization.py           # Per-group INT4 quantization + nn.Linear replacement
├── scripts/
│   ├── baseline.py               # Level 0: PyTorch benchmark (TTFT/TPOT/VRAM/roofline)
│   ├── baseline_config.py        # Model configuration and hardware constants
│   └── evaluate_quality.py       # HumanEval pass@1 evaluation of quantized model
├── CMakeLists.txt                # Native C++/CUDA benchmark build
├── setup.py                      # PyTorch extension build (torch.utils.cpp_extension)
├── pyproject.toml                # Python dependencies (uv)
└── Dockerfile                    # Reproducible environment with CUDA 12.1
```

---

## Implemented Levels

### Level 0 — PyTorch Baseline

**Goal:** establish the "zero" reference against which everything else is measured.

The [`baseline.py`](scripts/baseline.py) script loads Qwen2.5-0.5B-Instruct in bf16/fp16 and measures two separate metrics that any real serving system reports independently:

- **TTFT (Time-To-First-Token):** latency to process the full prompt (prefill). Dense GEMM, compute-bound.
- **TPOT (Time-Per-Output-Token):** latency of each subsequent token in autoregressive decode. GEMV, memory-bound — **this is the metric the rest of the project attacks**.

All measurements use CUDA events (not Python timers, which include host dispatch overhead), discard the first iterations as warmup, and report p50/p90/p99 over valid iterations.

The theoretical TPOT ceiling is computed in [`baseline_config.py`](scripts/baseline_config.py):

```
TPOT_min = model_bytes / GPU_bandwidth
         = (0.5B × 2 bytes) / 896 GB/s
         ≈ 1.12 ms/token
```

Everything measured from here on is reported as **% of this ceiling**, not as an arbitrary improvement percentage with no hardware context.

**Deliverable:** a reusable benchmark harness reporting TTFT, TPOT (p50/p90/p99), VRAM footprint, and % of theoretical ceiling reached.

---

### Level 1 — Naive CUDA Kernel

**File:** [`gemv_naive.cu`](src/gemv_naive.cu)

One thread per output row. Each thread traverses an entire row of the weight matrix and accumulates the dot product with the input vector:

```cuda
int row = blockIdx.x * blockDim.x + threadIdx.x;
if (row < rows) {
    float sum = 0.0f;
    for (int i = 0; i < cols; i++)
        sum += d_mat[row * cols + i] * d_vec[i];
    d_out[row] = sum;
}
```

**Expected (and observed) result:** performance below cuBLAS/PyTorch. This negative result is first-class content, not a failure — it demonstrates understanding of *why* a naive kernel loses to a mature library:

- **Uncoalesced accesses:** adjacent threads within a warp read different rows of the matrix. Thread 0 reads index `0`, thread 1 reads index `896` — positions far apart in memory that force the memory controller to fetch a full cache line for each thread, wasting most of the bandwidth.
- **No vectorized loads:** each read instruction moves 4 bytes when it could move 16.
- **Inefficient reduction:** partial sums accumulate in a single thread with no intra-warp parallelism.

The benchmark with real model dimensions (4864×896 for Qwen2.5-0.5B's MLP Up projection) measured **~190 GB/s** effective bandwidth.

---

### Level 2 — Optimized Kernel (coalescing, float4, warp shuffle)

**File:** [`gemv_optimized.cu`](src/gemv_optimized.cu)

Three techniques applied to approach the bandwidth ceiling:

1. **Thread reassignment for coalescing:** instead of one thread per row, a full warp (32 threads) collaborates on the same row. Adjacent threads read adjacent memory positions → coalesced access → a single memory transaction feeds 32 threads.

2. **Vectorized `float4` loads:** each thread reads 16 bytes per instruction instead of 4, multiplying read throughput. The matrix and vector are reinterpreted as `const float4*`.

3. **Warp shuffle for final reduction:** partial sums from the 32 threads in a warp are combined with `__shfl_down_sync` — direct register-to-register communication within the warp, no round-trip to shared memory:

```cuda
for (int i = 0; i < 5; i++) {
    sum += __shfl_down_sync(FULL_MASK, sum, offset);
    offset /= 2;
}
```

**Result:** ~1360 GB/s measured bandwidth on the same 4864×896 matrix.

**Technical honesty note on L2 cache:** the 17.4 MB matrix fits within the RTX 5070 Ti's L2 cache (~48-64 MB). When running the kernel 100 times over the same data, from the second iteration onward the data is served from the L2 cache rather than VRAM. The ~1360 GB/s figure reflects L2 cache throughput, not main memory bus bandwidth (GDDR7, 896 GB/s theoretical). This is a detail worth reporting: it demonstrates understanding of the real GPU memory hierarchy, not just the kernels themselves.

---

### Level 3 — INT4 Quantization + Fused Kernel

**Files:** [`gemv_quantized.cu`](src/gemv_quantized.cu), [`quantization.py`](src/quantization.py), [`binding.cpp`](src/binding.cpp)

This is the piece with the most real business value. Weights are quantized to INT4 and the kernel dequantizes and multiplies in a single pass, **without materializing the weights in fp16 in memory** — quantization reduces latency not through faster computation, but by moving 4× fewer bytes through the memory bus on every token.

#### Quantization Scheme

- **Symmetric INT4 per-group quantization (group_size=128):** every group of 128 consecutive weights shares a single fp32 scale factor.
- **Packing:** 8 INT4 values are packed into one `uint32_t` (4 bits × 8 = 32 bits).
- **Range:** [-8, 7] with `scale = max(|group|) / 7`.
- Weight quantization is performed **ahead-of-time** (outside the inference loop), not on-the-fly.

#### Dual Implementation

The file contains two variants following the same pedagogical progression as the previous levels:

- **`gemv_int4_naive_kernel`:** one thread per row, sequentially unpacks the 8 INT4 values from each `uint32_t`, applies scale, and accumulates.
- **`gemv_int4_optimized_kernel`:** one warp per row, vectorized `uint4` loads (128 bits = 4 × `uint32_t` = 32 INT4 values per instruction), warp shuffle for reduction.

#### PyTorch Integration

[`quantization.py`](src/quantization.py) exposes a `QuantizedLinearINT4` that serves as a drop-in replacement for `nn.Linear`:

```python
class QuantizedLinearINT4(nn.Module):
    def forward(self, x):
        # Autoregressive decode: batch=1, seq_len=1 → our CUDA kernel
        if batch_size == 1 and seq_len == 1:
            out = codealign_runtime_kernels.gemv_int4_forward(...)
            return out
        raise NotImplementedError("Prefill not yet implemented")
```

The `replace_linear_layers()` function recursively traverses the model and replaces all `nn.Linear` layers with their quantized equivalent. The C++ ↔ Python binding is done via `torch.utils.cpp_extension` in [`setup.py`](setup.py).

#### Important Note on llama.cpp Comparison

This project's INT4 scheme (per-group symmetric, g=128) **is not identical** to llama.cpp's Q4_K_M format, which uses a custom mixed packing scheme with super-blocks and multiple scale levels. The speed comparison in Level 5 will be valid as "my code vs. the industry standard", but it is not an apples-to-apples comparison in terms of quantization scheme — and this is stated explicitly.

---

## Results

### Isolated Kernel Benchmarks (GEMV on 4864×896 matrix, fp32)

| Kernel | Avg Time (ms) | Bandwidth (GB/s) | Note |
|--------|---:|---:|------|
| Level 1 — naive | 0.092 | ~190 | Limited by uncoalesced accesses |
| Level 2 — optimized | 0.013 | ~1360 | **~7× faster** — L2 cache throughput¹ |
| Level 3 — INT4 naive | — | — | Formal measurement pending |
| Level 3 — INT4 optimized | — | — | Formal measurement pending |

> ¹ The dataset (17.4 MB) fits in the RTX 5070 Ti's L2 cache. In real inference with the full model (~500 MB in fp16, ~125 MB in INT4), the bottleneck returns to VRAM bandwidth, not cache. See [Benchmarking Methodology](#benchmarking-methodology).

### End-to-End Model Benchmark (Level 0 — Qwen2.5-0.5B, PyTorch eager)

| Metric | Value | Note |
|--------|------:|------|
| TTFT p50 | — | Formal run pending |
| TPOT p50 | — | Formal run pending |
| % theoretical BW ceiling | — | Ceiling: ~1.12 ms/token |
| Peak VRAM | — | — |

> Cells marked with "—" will be filled in during the next formal benchmark run. Scripts are ready and validated.

### Summary Table (to be completed in Level 5)

| Level | TTFT p50 (ms) | TPOT p50 (ms/token) | % BW ceiling | VRAM | HumanEval pass@1 | Note |
|-------|:---:|:---:|:---:|:---:|:---:|------|
| PyTorch eager (bf16) | — | — | — | — | (reference) | |
| Level 1 — naive | — | — | — | — | — | |
| Level 2 — optimized | — | — | — | — | — | |
| Level 3 — INT4 fused | — | — | — | — | — | vs. bf16 |
| Level 4 — Flash-Decoding | — | — | — | — | — | |
| Level 5 — C++ integration | — | — | — | — | — | |
| + Prompt Lookup Decoding | — | — | — | — | — | acceptance rate: —% |
| llama.cpp (Q4_K_M) | — | — | — | — | — | different scheme² |

> ² Q4_K_M uses a custom mixed packing scheme with super-blocks — the speed comparison is valid as "my code vs. industry standard", but the quantization schemes are not identical.

---

## Benchmarking Methodology

All benchmarks follow the same rules:

1. **CUDA events, not Python timers.** `cudaEventRecord` / `cudaEventElapsedTime` measure GPU time without host dispatch overhead.
2. **Warmup discarded.** The first 10 iterations are excluded from statistics — the GPU clock and caches need to stabilize.
3. **Statistics, not anecdotes.** p50/p90/p99 are reported over at least 90 valid iterations. A single number without a distribution is not empirical verification.
4. **Theoretical ceiling always present.** Every TPOT result is accompanied by its % of the physically minimum possible (model_bytes / GPU_bandwidth).
5. **Correctness accompanies speed.** The benchmark harness in [`main.cpp`](src/main.cpp) validates each kernel's output against the naive reference with configurable tolerance (`1e-4` for fp32→fp32, `0.05` for INT4 vs. fp32).

---

## Quantization vs. Quality

INT4 quantization reduces bytes moved per token by 4× — but is it free in quality? Most projects of this kind don't verify. This one does.

[`evaluate_quality.py`](scripts/evaluate_quality.py) loads the model, applies `replace_linear_layers()` to quantize all weights to INT4 in-memory, and runs **HumanEval pass@1** using the same `bigcode-evaluation-harness` used in [CodeAlign](https://github.com/Sergasgr/CodeAlign). The goal is to report both numbers side by side:

| Model | HumanEval pass@1 |
|-------|:---:|
| Qwen2.5-0.5B bf16 (reference) | — |
| Qwen2.5-0.5B INT4 (our kernel) | — |

The question answered with data: is the 4× reduction in bytes moved costing the model correct code? Answering this with empirical evidence is the kind of rigor that separates someone who optimizes kernels from someone who understands the full trade-off of serving a compressed model.

---

## Hardware

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX 5070 Ti |
| VRAM | 16 GB GDDR7 (256-bit) |
| Memory BW (theoretical) | 896 GB/s |
| L2 Cache | ~48-64 MB |
| CUDA Toolkit | 12.x |

> A single consumer GPU with 8-16 GB is more than enough for a 0.5-1.5B model. No cloud or multi-GPU needed — it's cheaper and faster to iterate than training.

---

## Setup & Usage

### Prerequisites

- NVIDIA GPU with CUDA support (Compute Capability ≥ 7.0)
- CUDA Toolkit 12.x
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (package manager)
- Docker + nvidia-container-toolkit (optional, for reproducibility)

### Option 1: Docker (recommended for reproducibility)

```bash
# Configure NVIDIA runtime for Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
sudo systemctl restart docker

# Build and run
docker build -t codealign-runtime .
docker run --gpus all -it --rm --env-file .env -v $(pwd):/app codealign-runtime
```

### Option 2: Local

```bash
# Install Python dependencies
uv sync

# Level 0 — PyTorch Baseline
uv run python -m scripts.baseline

# Build C++/CUDA benchmark (Levels 1-3)
cmake -B build && cmake --build build
./build/gemv_benchmark

# Build PyTorch extension (INT4 kernel for Level 3)
uv run python setup.py build_ext --inplace

# Post-quantization quality evaluation
uv run python -m scripts.evaluate_quality
```

### Environment Variables

Copy `.env.example` to `.env` and set your Hugging Face token:

```bash
cp .env.example .env
# Edit .env with your HF_TOKEN
```

---

## Roadmap

### Next: Level 4 — Flash-Decoding

During the decode phase (batch=1, a single new query against a growing KV-cache), the compute pattern is not the same as the prefill that original FlashAttention targets. With a single new query there is nothing to parallelize over in the Q dimension — most GPU SMs would sit idle.

The correct technique for this case is **Flash-Decoding**: parallelize over the KV-cache dimension (split it into chunks, compute partial softmax/output per chunk in parallel, and combine with a final reduction step using online-softmax). This is the right design for batch=1 decode, not prefill-style FlashAttention.

**Scoped and defensible:** single head, short context, numerical validation against PyTorch with explicit tolerance (max absolute error < 1e-2 in fp16, tested with multiple seeds).

### Level 5 — Final Integration

Minimal C++ inference loop (tokenizer → embeddings → transformer layers using our kernels → sampling), exposed via `pybind11` or `torch.utils.cpp_extension`. The final table with separate columns for TTFT and TPOT, real VRAM, and HumanEval pass@1 alongside latency.

Comparison: PyTorch eager → Level 1 → Level 2+3 → Level 4 → llama.cpp. Reaching 50-70% of llama.cpp's performance with your own code is a success, not a partial failure.

### Level 6 — Prompt Lookup Decoding

The piece with the best impact/effort ratio, and the only one that is **specific to the use case** — code completion. Source code has extremely high textual redundancy: repeated variable names, recurring structural patterns, edits that literally reuse fragments from the context. This is arguably the best possible case for algorithmic speculative decoding in all of NLP.

**Mechanism:** after generating each token, search the existing context for the longest match with the last N generated tokens. If there's a match, take the following K tokens as a speculative "draft" and verify them in a single forward pass (compute-bound and parallel). Accept the longest prefix that matches what the model would have generated.

No additional model or training required — it's control flow plus a batched forward pass, with no significant extra VRAM.

### Optional Polish

- **Roofline model:** a figure placing Levels 1, 2, and 3 on the arithmetic intensity vs. GFLOPs/s plot. The fp32 kernels fall on the memory-bound diagonal; INT4 shifts to the right — that's the complete visual explanation of why quantization accelerates inference.
- **CUDA Graphs:** capture the kernel launch sequence for a decode step and replay it without CPU launch overhead.
- **Nsight Systems:** visualize gaps between kernels on the GPU timeline.
- **Paged KV-cache:** vLLM-style memory management.

---

## Portfolio Context

This project is the second half of a complete portfolio narrative:

| Project | Demonstrates |
|---------|-------------|
| [**CodeAlign**](https://github.com/Sergasgr/CodeAlign) | Data curation → SFT → DPO → HumanEval evaluation |
| **CodeAlign-Runtime** | Quantization → CUDA kernels → low-latency serving |

The complete chain — data → model → serving in production — with the same person behind every link, is what closes the portfolio narrative end to end.

---

## License

MIT