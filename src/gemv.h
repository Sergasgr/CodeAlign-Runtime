#pragma once
#include <cstdint>

void run_gemv_naive(const float* d_mat, const float* d_vec, float* d_out_naive, int rows, int cols);
void run_gemv_optimized(const float* d_mat, const float* d_vec, float* d_out_opt, int rows, int cols);
void run_gemv_int4_naive_kernel(const uint32_t* d_q_mat, const float* d_scales, const float* d_vec, float* d_out_q, int rows, int cols);
void run_gemv_int4_optimized_kernel(const uint32_t* d_q_mat, const float* d_scales, const float* d_vec, float* d_out_q, int rows, int cols);
void run_flash_decoding_partial(const float* d_Q, const float* d_K, const float* d_V, float* d_O_partial, float* d_lse_partial, int D, int S, int chunk_size);
void run_flash_decoding_final(const float* d_O_partial, const float* d_lse_partial, float* d_O_final, int D, int num_chunks);