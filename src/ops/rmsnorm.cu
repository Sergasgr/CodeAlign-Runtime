#include "../transformer/transformer.h"
#include <cmath>
#define EPS 1e-6f

__global__ void RMSNorm_kernel(float* current_token, float* weights, float* result, int d) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();
    float val = (id < d) ? current_token[id] : 0.0f;
    atomicAdd(&s_sum, val * val);
    __syncthreads();
    float RMS = sqrtf(s_sum / d + EPS);
    if (id < d) result[id] = (val / RMS) * weights[id];
}

void run_RMSNorm_kernel(float* current_token, float* weights, float* result, int d) {
    int block_size = 1024;
    int grid_size = 1;
    RMSNorm_kernel<<<grid_size, block_size>>>(current_token, weights, result, d);
    cudaDeviceSynchronize();
}

/*
The kernel uses `atomicAdd(&s_sum, val * val)` for the reduction, which is functionally correct but:

1. **Non-deterministic ordering** of atomic adds means the sum can vary slightly between runs (floating-point non-associativity).
2. **Slower** than warp-shuffle + shared-memory reduction for D=896 (all 896 threads serialize on one atomic).

This works correctly for the benchmark. If you want to optimize it later (optional polish), you'd use the same warp-shuffle pattern from your GEMV kernels. Not a blocker for Level 5.

---
*/