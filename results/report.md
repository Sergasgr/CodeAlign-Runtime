### Level 0 — PyTorch baseline

| Metric (ms) | p50 | p90 | p99 |
| --- | ---: | ---: | ---: |
| TTFT (prefill, 66 tokens) | 15.90 | 16.99 | 18.89 |
| TPOT (decode) | 14.61 | 15.68 | 17.62 |

Bandwidth ceiling: 988 MB/token -> **1.103 ms**; TPOT p50 reaches **7.5%** of it. Peak VRAM: 997 MB.
_NVIDIA GeForce RTX 5070 Ti, driver 595.91.07, torch 2.13.0+cu130, commit 9974682 (dirty), 2026-09-27T19:51:51+00:00_

### GEMV kernels (Levels 1-3)

Shape: {'rows': 4864, 'cols': 896, 'name': 'mlp_up_gate_4864x896'}

| Kernel | L2-hot p50 (ms) | L2-hot GB/s | L2-cold p50 (ms) | L2-cold GB/s | % of 896 GB/s (cold) |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp32_naive | 0.0930 | 187.7 | 0.0979 | 178.4 | 19.9 |
| fp32_optimized | 0.0146 | 1197.6 | 0.0335 | 521.0 | 58.1 |
| int4_naive | 0.0348 | 67.2 | 0.0578 | 40.5 | 4.5 |
| int4_optimized | 0.0313 | 74.8 | 0.0308 | 75.8 | 8.5 |

Validation: 16/16 kernel x shape checks passed (random data, CPU reference).
_NVIDIA GeForce RTX 5070 Ti_

### GEMM kernels

Shape: {'M': 16, 'in_features': 896, 'out_features': 4864, 'num_splits': 4, 'name': 'M16_mlp_up_896->4864'}

| Kernel | L2-hot p50 (ms) | L2-hot GB/s | L2-cold p50 (ms) | L2-cold GB/s | % of 896 GB/s (cold) |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp32_naive | 0.2397 | 74.3 | 0.2447 | 72.8 | 8.1 |
| fp32_tiled | 0.0607 | 293.2 | 0.0759 | 234.4 | 26.2 |
| int4_naive | 0.0520 | 51.6 | 0.0673 | 39.9 | 4.5 |
| int4_tiled | 0.0575 | 46.7 | 0.0620 | 43.3 | 4.8 |
| int4_splitk | 0.0591 | 45.4 | 0.0604 | 44.5 | 5.0 |

Validation: 15/15 kernel x shape checks passed (random data, CPU reference).
_NVIDIA GeForce RTX 5070 Ti_

### Level 4 — Flash-Decoding

Validation: 30/30 cases, worst max|err| = 1.5e-07 (tolerance 0.0001), head_dim=64.

| Seq length | PyTorch p50 (ms) | Custom p50 (ms) | PyTorch / Custom | KV-read ceiling (ms) |
| --- | ---: | ---: | ---: | ---: |
| 256 | 0.0978 | 0.1507 | 0.65x | 0.0001 |
| 4,096 | 0.0995 | 0.1549 | 0.64x | 0.0023 |
| 16,384 | 0.1047 | 0.1550 | 0.68x | 0.0094 |
| 65,536 | 0.1065 | 0.1640 | 0.65x | 0.0374 |
| 131,072 | 0.1228 | 0.2730 | 0.45x | 0.0749 |
| 262,144 | 0.2699 | 0.3070 | 0.88x | 0.1498 |
_NVIDIA GeForce RTX 5070 Ti, driver 595.91.07, torch 2.13.0+cu130, commit 9974682 (dirty), 2026-09-27T19:54:09+00:00_

### Level 5 — Engine

**Single block (layer 0)**

| Context S | p50 (ms) | p99 (ms) | Ceiling (ms) | % of ceiling |
| --- | ---: | ---: | ---: | ---: |
| 128 | 1.3585 | 2.4163 | 0.0090 | 0.66% |
| 512 | 2.0889 | 2.0995 | 0.0094 | 0.45% |
| 1,024 | 2.0895 | 2.2164 | 0.0100 | 0.48% |
| 2,048 | 2.0894 | 2.2155 | 0.0112 | 0.54% |

**Full model (24 layers + lm_head + argmax)**

| Context S | p50 (ms) | p99 (ms) | Ceiling (ms) | % of ceiling | tok/s (p50) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 | 33.5806 | 69.9621 | 0.2967 | 0.88% | 30 |
| 512 | 56.0880 | 128.0063 | 0.3072 | 0.55% | 18 |
| 1,024 | 56.0268 | 90.1767 | 0.3213 | 0.57% | 18 |
| 2,048 | 56.0212 | 57.3317 | 0.3494 | 0.62% | 18 |

End-to-end: prompt 53 tokens, TTFT 553.7 ms (chunked prefill), TPOT p50 31.457 ms over 128 tokens. Resident VRAM 598 MB (tensors 511 + C++ buffers 88); decode peak 599 MB; load-time peak 2270 MB (INT4 quantization temporaries).
_NVIDIA GeForce RTX 5070 Ti, driver 595.91.07, torch 2.13.0+cu130, commit 9974682 (dirty), 2026-09-27T19:55:11+00:00_

### Level 6 — Prompt Lookup Decoding

| Prompt | Greedy tok/s | PLD tok/s | Speedup | Acceptance | Tokens/forward | Identical output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| refactor_type_hints | 21.8 | 24.4 | 1.12x | 91.6% | 5.33 | yes |
| cpp_getters_setters | 25.3 | 20.0 | 0.79x | 36.6% | 1.33 | yes |
| unit_tests | 29.9 | 30.4 | 1.01x | 80.0% | 2.09 | yes |
| open_ended_control | 27.4 | 32.3 | 1.18x | 97.4% | 3.66 | yes |

n-gram=3, max draft=5, 256 new tokens per prompt. Overall speedup: **1.00x**.
_NVIDIA GeForce RTX 5070 Ti, driver 595.91.07, torch 2.13.0+cu130, commit 9974682 (dirty), 2026-09-27T19:57:07+00:00_

### Quantization vs. quality

| Model | HumanEval pass@1 (greedy) |
| --- | ---: |
| bf16 (reference) | 0.2439 |

### llama.cpp reference

| GGUF | File (MB) | Context depth | tok/s | TPOT (ms) |
| --- | ---: | ---: | ---: | ---: |
| F16 | 994 | 128 | 464.4 | 2.153 |
| F16 | 994 | 512 | 454.0 | 2.203 |
| F16 | 994 | 1,024 | 486.6 | 2.055 |
| F16 | 994 | 2,048 | 470.4 | 2.126 |
| Q4_0 | 352 | 128 | 763.0 | 1.311 |
| Q4_0 | 352 | 512 | 753.6 | 1.327 |
| Q4_0 | 352 | 1,024 | 738.9 | 1.353 |
| Q4_0 | 352 | 2,048 | 699.5 | 1.430 |
| Q4_K_M | 398 | 128 | 725.3 | 1.379 |
| Q4_K_M | 398 | 512 | 718.8 | 1.391 |
| Q4_K_M | 398 | 1,024 | 704.0 | 1.421 |
| Q4_K_M | 398 | 2,048 | 670.4 | 1.492 |

### Summary

| System | TTFT p50 (ms) | TPOT p50 (ms) | % BW ceiling | VRAM (MB) | HumanEval pass@1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PyTorch eager (bf16) | 15.90 | 14.612 | 7.5% | 997 | 0.244 |
| C++/CUDA engine, INT4 (S=512) | 553.7 | 56.088 | 0.5% | 598 | — |
| + Prompt Lookup Decoding | — | — | — | — | x1.00 tok/s, identical output |
| llama.cpp Q4_0 (depth 512) | — | 1.327 | — | — | — |
| llama.cpp Q4_K_M (depth 512) | — | 1.391 | — | — | — |
