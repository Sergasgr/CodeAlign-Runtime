#pragma once
#include <cuda_runtime.h>
#include <cstdint>

void run_gemv_naive(const float* d_mat, const float* d_vec, float* d_out_naive, int rows, int cols, cudaStream_t stream = 0);
void run_gemv_optimized(const float* d_mat, const float* d_vec, float* d_out_opt, int rows, int cols, cudaStream_t stream = 0);
void run_gemv_int4_naive_kernel(const uint32_t* d_q_mat, const float* d_scales, const float* d_vec, float* d_out_q, int rows, int cols, cudaStream_t stream = 0);
void run_gemv_int4_optimized_kernel(const uint32_t* d_q_mat, const float* d_scales, const float* d_vec, float* d_out_q, int rows, int cols, cudaStream_t stream = 0);

void run_gemv_int4_v2_kernel(const uint32_t* d_q_mat, const float* d_scales, const float* d_vec, const float* d_bias, float* d_out, int rows, int cols, cudaStream_t stream = 0);
int gemv_int4_v2_rows_per_warp(int cols);
int gemv_int4_v2_grid_size(int rows, int cols);