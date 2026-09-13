#include "../transformer/transformer.h"
#include <cmath>

#define EPS 1e-6f
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

__global__ void RMSNorm_kernel(float* current_token, float* weights, float* result, int d) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    __shared__ float warp_sums[32]; 
    __shared__ float s_rms;

    float val = (id < d) ? current_token[id] : 0.0f;
    float sum = val * val;

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

    if (id < d) {
        result[id] = (val / s_rms) * weights[id];
    }
}

void run_RMSNorm_kernel(float* current_token, float* weights, float* result, int d) {
    int block_size = 1024;
    int grid_size = 1;
    RMSNorm_kernel<<<grid_size, block_size>>>(current_token, weights, result, d);
    cudaDeviceSynchronize();
}