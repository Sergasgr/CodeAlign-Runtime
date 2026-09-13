#include "gemm.h"
#define TILE_SIZE 16

__global__ void gemm_optimized_kernel(const float* A, const float* B, float* C, int K, int in_features, int out_features) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;
    int num_tiles = (in_features + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        int global_col_A = t * TILE_SIZE + tx;
        if(row < K && global_col_A < in_features) sA[ty][tx] = A[row * in_features + global_col_A];
        else sA[ty][tx] = 0.0f;
        
        int global_col_B = t * TILE_SIZE + ty;
        if(col < out_features && global_col_B < in_features) sB[ty][tx] = B[col * in_features + global_col_B];
        else sB[ty][tx] = 0.0f;

        __syncthreads();

        for(int i = 0; i < TILE_SIZE; i++) {
            sum += sA[ty][i] * sB[i][tx];
        }

        __syncthreads();
    }

    if (row < K && col < out_features) {
        C[row * out_features + col] = sum;
    }
}

void run_gemm_optimized(const float* A, const float* B, float* C, int K, int in_features, int out_features) {
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((out_features + TILE_SIZE - 1) / TILE_SIZE, (K + TILE_SIZE - 1) / TILE_SIZE);
    gemm_optimized_kernel<<<grid_size, block_size>>>(A, B, C, K, in_features, out_features);
    cudaDeviceSynchronize();
}