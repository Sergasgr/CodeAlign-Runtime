#pragma once
#include <cstdint>

void run_gemm_naive(const float* A, const float* B, float* C, int K, int in_features, int out_features);
void run_gemm_optimized(const float* A, const float* B, float* C, int K, int in_features, int out_features);
void run_gemm_int4_naive(const float* A, const uint32_t* B_quant, const float* B_scales, float* C, int K, int in_features, int out_features);
void run_gemm_int4_optimized(const float* A, const uint32_t* B_quant, const float* B_scales, float* C, int K, int in_features, int out_features);
void run_gemm_int4_splitk_partial_kernel(const float* A, const uint32_t* B_quant, const float* B_scales, float* C_partial, int K, int num_splits, int in_features, int out_features);
void run_gemm_int4_splitk_final_kernel(const float* C_partial, float* C, int K, int out_features, int num_splits); 