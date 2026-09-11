#pragma once
#include <cstdint>

void run_flash_decoding_partial(const float* d_Q, const float* d_K, const float* d_V, float* d_O_partial, float* d_lse_partial, int D, int S, int chunk_size);
void run_flash_decoding_final(const float* d_O_partial, const float* d_lse_partial, float* d_O_final, int D, int num_chunks);