#include "gemm.h"

__global__ void gemm_naive_kernel(const float* A, const float* B, float* C, int K, int in_features, int out_features) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < K && col < out_features) { 
        float sum = 0.0f;
        for(int i = 0; i < in_features; i++) {
            sum += A[row * in_features + i] * B[col * in_features + i];
        }
        C[row * out_features + col] = sum;
    }
}

void run_gemm_naive(const float* A, const float* B, float* C, int K, int in_features, int out_features) {
    dim3 block_size(16, 16);
    dim3 grid_size((out_features + 15) / 16, (K + 15) / 16);
    gemm_naive_kernel<<<grid_size, block_size>>>(A, B, C, K, in_features, out_features);
    cudaDeviceSynchronize();
}