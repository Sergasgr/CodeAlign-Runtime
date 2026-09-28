# Changelog

## v1.0.0

### Fixed
- **Engine projections never ran:** `QwenBlock.load_weights` read `in_features` from the scales tensor (number of groups, e.g. 7), so the INT4 GEMV guard skipped every projection and left garbage. It is now `packed.size(1) * 8`, with shape and dtype checks.
- **Qwen2 architecture:** the block now implements grouped-query attention (2 KV heads; query head *h* reads KV head *h / 7*; k/v are 896→128) and the q/k/v biases. RoPE and the KV-cache use the correct row strides.
- **Engine safety:** checks for tokens per forward, KV-cache overflow and rollback range. CUDA errors surface as Python exceptions. The block owns the weight tensors it points to and cannot be copied.
- **Flash-Decoding race:** the reduced score shared `warp_sums[0]` with warp 0's partial sum (a write-after-read hazard across warps). It now has its own shared variable.
- **RoPE precision:** under `-use_fast_math`, `powf`/`sinf`/`cosf` compiled to approximate instructions.
  - Qwen2.5's large q/k biases amplify the resulting ~1e-5 rad angle error: the engine drifted from Hugging Face by ~1.6% after layer 0 and ~12% in the logits.
  - The angle now follows Hugging Face's fp32 computation, and sin/cos are evaluated in double precision.
- **Kernels:** the SwiGLU literal `1.0` was double precision, forcing FP64 math. The INT4 GEMV guard now requires `cols % 32`.
- **Level 5 benchmark:** a 1-D input was interpreted as 896 tokens, and the synthetic scales had the wrong shape. The benchmark now measures real weights at fixed context lengths.
- **Build:**
  - CMake links `CUDA::cudart`, so the host `.cpp` files find `cuda_runtime.h`, and builds for the native architecture.
  - Docker uses CUDA 13.0, which matches `torch 2.13.0+cu130`; sm_120 needs ≥ 12.8.
  - The project is no longer installed as a package (`[tool.uv] package = false`).
  - `build.sh` uses one CUDA toolkit for CMake and the extensions (`$CUDA_HOME` or the first `nvcc` on PATH) and checks its major version against torch. CMake would otherwise pick an nvcc next to the C++ compiler, such as Ubuntu's CUDA 12.0 in `/usr/bin`.
- **HumanEval evaluation:**
  - The bigcode-evaluation-harness arguments are complete.
  - INT4 prefill now runs on the INT4 GEMM.
  - Unit-test execution is serialized, because filelock 3.32 raises on concurrent forks from threads.
  - The dataset is loaded as `openai/openai_humaneval`. Current `huggingface_hub` rejects bigcode's pre-namespace id `openai_humaneval`, and bigcode swallowed the error.
- **Measurements:**
  - Level 5 reports resident VRAM: torch tensors plus the C++ `cudaMalloc` buffers, which torch statistics do not see. The load-time INT4 quantization peak is reported separately.
  - Level 6 speeds are decode-only: the prompt prefill is identical in both modes and is reported separately.

### Added
- Full 24-layer engine (`scripts/engine.py`): chunked prefill, greedy generation and a config check against the kernels' hard-coded choices.
- Level 6 end to end: Prompt Lookup Decoding on the full model, with rollback in every layer, acceptance and speedup metrics, and a lossless check against greedy decoding.
- C++ harness:
  - Random data and CPU references (double accumulation). INT4 kernels are checked against the dequantized weights.
  - NaN-prefilled outputs, `CUDA_CHECK`, and validation of every kernel and shape, including Split-K.
  - L2-hot and L2-cold timings with p50/p90/p99, and JSON output.
- Engine tests use a float64 Hugging Face reference with a tolerance calibrated to Hugging Face's own fp32 error, plus a small model with large q/k biases that reproduces Qwen2.5's conditioning without a download.
- Tests (`pytest`): CPU tests for packing, the `nn.Linear` drop-in, the roofline and the kernel emulation. GPU tests for the kernels and for the engine against Hugging Face Qwen2 (block, logits, greedy generation, speculative decoding).
- `pytest` prints the GPU and extension status in its header, and GPU tests are skipped with the real import error.
- `scripts/roofline.py` (bandwidth ceilings), `scripts/results_io.py` (results with environment and git commit), `scripts/report.py` (README tables), `scripts/llama_cpp_benchmark.py` (F16 / Q4_0 / Q4_K_M via llama-bench), `build.sh`.

### Changed
- One model everywhere: `Qwen/Qwen2.5-Coder-0.5B` (base). Level 0 uses a raw completion prompt instead of a chat template.
- Flash-Decoding validation:
  - Standard-normal inputs, tolerance 1e-4.
  - Dominant-key cases, and a check that the data would catch a kernel ignoring the scores.
  - The benchmark runs at the model's head_dim (64).
- Python ≥ 3.12; `pytest` dev dependency; `ninja` for faster extension builds.
