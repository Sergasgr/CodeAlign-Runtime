#pragma once
#include <cstdint>

void run_gemm_naive(const float* A, const float* B, float* C, int K, int in_features, int out_features);
// GEMM FP32 Optimizado (Tiling 2D + Memoria Compartida)
void run_gemm_optimized(const float* A, const float* B, float* C, int K, int in_features, int out_features);
// GEMM INT4 Naive (Pesos empaquetados en int32, sin Tiling)
void run_gemm_int4_naive(const float* A, const uint32_t* B_quant, const float* B_scales, float* C, int K, int in_features, int out_features);
// GEMM INT4 Optimizado (La bestia final: Tiling + Cuantización)
void run_gemm_int4_optimized(const float* A, const uint32_t* B_quant, const float* B_scales, float* C, int K, int in_features, int out_features);