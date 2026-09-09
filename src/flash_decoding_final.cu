#include "gemv.h"
#include <limits>
#include <cmath>

__global__ void flash_decoding_final(const float* d_O_partial, const float* d_lse_partial, float* d_O_final, int D, int num_chunks) {
    __shared__ float shared_global_lse;
    int tid = threadIdx.x;
    if(tid == 0) {
        float m_global = std::numeric_limits<float>::lowest();
        float d_global = 0.0f;
        for(int c = 0; c < num_chunks; c++) {
            float lse_c = d_lse_partial[c];
            float m_new = fmaxf(m_global, lse_c);
            float correction = expf(m_global - m_new);
            d_global = (d_global * correction) + expf(lse_c - m_new);
            m_global = m_new;
        }
        shared_global_lse = m_global + logf(d_global);
    }
    __syncthreads();

    float final_value = 0.0f;
    for(int c = 0; c < num_chunks; c++) {
        float val = d_O_partial[c * D + tid];
        float factor = expf(d_lse_partial[c] - shared_global_lse);
        final_value += val * factor;
    }
    d_O_final[tid] = final_value;
}

void run_flash_decoding_final(const float* d_O_partial, const float* d_lse_partial, float* d_O_final, int D, int num_chunks) {
    int block_size = D;
    int grid_size = 1;
    flash_decoding_final<<<grid_size, block_size>>>(d_O_partial, d_lse_partial, d_O_final, D, num_chunks);
    cudaDeviceSynchronize();
}
