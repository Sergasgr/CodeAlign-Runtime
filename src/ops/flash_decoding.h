#pragma once
// LEGACY (v1.0): not used by the engine since v1.1. 
//
// The v1.0 decode attention: one partial launch + one final launch per (token, query head), one block per
// 256-token chunk that walks its tokens one by one (two __syncthreads() and a single thread summing the score
// per token), and every K/V row read once per QUERY head (7 times per KV head in Qwen2.5-0.5B).
// Replaced by src/ops/gqa_attention.{h,cu}: two launches per layer for every token and head, K/V read once per
// KV head, warps of a block working on different positions. Decoder block at S=512: 1.82 ms -> 0.135 ms (Level 5).
//
// Kept on purpose, still compiled (codealign_runtime_kernels only), tested and benchmarked:
//   - it is the v1.0 row of the Level 4 single-head table (flash_decoding_v10_forward), so the before/after is
//     measured in the same run on the same data;
//   - it documents the first design and why it was replaced (README, "v1.1: what changed").
#include <cuda_runtime.h>

void run_flash_decoding_partial(const float* d_Q, const float* d_K, const float* d_V, float* d_O_partial, float* d_lse_partial, int D, int S, int chunk_size, cudaStream_t stream = 0);
void run_flash_decoding_final(const float* d_O_partial, const float* d_lse_partial, float* d_O_final, int D, int num_chunks, cudaStream_t stream = 0);