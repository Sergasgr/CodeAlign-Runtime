#pragma once

void run_gemv_naive(const float* d_mat, const float* d_vec, float* d_out_naive, int rows, int cols);
void run_gemv_optimized(const float* d_mat, const float* d_vec, float* d_out_opt, int rows, int cols);
void run_gemv_int4_kernel(const uint32_t* d_q_mat, const float* d_scales, const float* d_vec, float* d_out_q, int rows, int cols);