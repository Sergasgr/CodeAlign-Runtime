#include "gemv.h"
#include <iostream>

__global__ void gemv_naive_kernel(const float* d_mat, const float* d_vec, float* d_out_naive, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < rows) {
        float sum = 0.0f;
        for(int i = 0; i < cols; i++) {
            sum += d_mat[row * cols + i] * d_vec[i]; 
        }
        d_out_naive[row] = sum;
    }
}

void run_gemv_naive(const float* d_mat, const float* d_vec, float* d_out_naive, int rows, int cols) {
    int block_size = 256;
    int grid_size = (rows + block_size - 1) / block_size;
    gemv_naive_kernel<<<grid_size, block_size>>>(d_mat, d_vec, d_out_naive, rows, cols);
    cudaDeviceSynchronize();
}