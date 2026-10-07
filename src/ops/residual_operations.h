#pragma once
#include <cuda_runtime.h>

void run_add_residual_kernel(float* base_vector, float* add_vector, int d, cudaStream_t stream = 0);
void run_add_bias_kernel(float* x, const float* bias, int num_tokens, int n, cudaStream_t stream = 0);