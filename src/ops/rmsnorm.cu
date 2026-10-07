#include "../include/codealign_cuda.h"
#include "../transformer/transformer.h"
#include <cmath>

#define EPS 1e-6f
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

__global__ void RMSNorm_kernel(float* current_token, float* weights, float* result, int d, int num_tokens) {
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    int thread_idx = threadIdx.x;
    int stride = blockDim.x;

    __shared__ float warp_sums[32]; 
    __shared__ float s_rms;

    float sum = 0.0f;
    for (int feat = thread_idx; feat < d; feat += stride) {
        float val = current_token[token_idx * d + feat];
        sum += val * val;
    }

    int lane_id = thread_idx % WARP_SIZE;
    int warp_id = thread_idx / WARP_SIZE;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(FULL_MASK, sum, offset);
    }
    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float warp_sum = (lane_id < (blockDim.x / WARP_SIZE)) ? warp_sums[lane_id] : 0.0f;
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(FULL_MASK, warp_sum, offset);
        }
        if (lane_id == 0) {
            s_rms = sqrtf((warp_sum / d) + EPS);
        }
    }
    __syncthreads();

    for (int feat = thread_idx; feat < d; feat += stride) {
        float val = current_token[token_idx * d + feat];
        result[token_idx * d + feat] = (val / s_rms) * weights[feat];
    }
}

void run_RMSNorm_kernel(float* current_token, float* weights, float* result, int d, int num_tokens, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = num_tokens;
    CODEALIGN_KERNEL_LAUNCH(RMSNorm_kernel, grid_size, block_size, 0, stream, current_token, weights, result, d, num_tokens);
}