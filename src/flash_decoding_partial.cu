#include "gemv.h"
#include <limits>
#include <cmath>

#define FULL_MASK 0xffffffff
const int WARP = 32;

__global__ void flash_decoding_partial(const float* d_Q, const float* d_K, const float* d_V, float* d_O_partial, float* d_lse_partial, int D, int S, int chunk_size) {
    int chunk_idx = blockIdx.x;
    int tid = threadIdx.x;
    int lane_id = tid % WARP;
    int warp_id = tid / WARP;

    extern __shared__ float s_mem[];
    float* s_Q = s_mem;
    float* warp_sums = s_mem + D;

    if(tid < D) s_Q[tid] = d_Q[tid];
    __syncthreads();

    int start_token = chunk_idx * chunk_size;

    float max_local = std::numeric_limits<float>::lowest();
    float d_local = 0.0f;
    float O_local = 0.0f;

    for(int i = 0; i < chunk_size && (start_token + i) < S; i++) { 
        int token_idx = start_token + i;

        float val = 0.0f;
        if(tid < D) {
            val = d_K[token_idx * D + tid] * s_Q[tid];
        }

        for(int offset = WARP / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(FULL_MASK, val, offset);
        }

        if(lane_id == 0) {
            warp_sums[warp_id] = val; 
        }
        __syncthreads();

        if(tid == 0) {
            float S_i = 0.0f;
            int num_warps = blockDim.x / WARP;
            for(int w = 0; w < num_warps; w++) {
                S_i += warp_sums[w];
            }
            S_i /= sqrtf((float)D);
            warp_sums[0] = S_i;
        }
        __syncthreads(); 

        float S_i = warp_sums[0];

        float max_new = fmaxf(max_local, S_i);
        float correction = expf(max_local - max_new);
        d_local = d_local * correction + expf(S_i - max_new);
        O_local = O_local * correction + expf(S_i - max_new) * d_V[token_idx * D + tid];
        max_local = max_new;
    }

    d_O_partial[chunk_idx * D + tid] = O_local / d_local;

    if(tid == 0) {
        d_lse_partial[chunk_idx] = max_local + logf(d_local);
    }
}

void run_flash_decoding_partial(const float* d_Q, const float* d_K, const float* d_V, float* d_O_partial, float* d_lse_partial, int D, int S, int chunk_size) {
    int block_size = D;
    int grid_size = (S + chunk_size - 1) / chunk_size;
    int shared_mem_bytes = (D + (D / WARP)) * sizeof(float);
    flash_decoding_partial<<<grid_size, block_size, shared_mem_bytes>>>(d_Q, d_K, d_V, d_O_partial, d_lse_partial, D, S, chunk_size);
    cudaDeviceSynchronize();
}
