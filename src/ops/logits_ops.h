#pragma once
#include <cuda_runtime.h>

void run_compute_argmax_kernel(const float* logits, int* predicted_tokens, int num_tokens, int vocab_size, cudaStream_t stream = 0);