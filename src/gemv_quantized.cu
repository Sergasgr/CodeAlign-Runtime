#include "gemv.h"
#include <iostream> //innecesario?
#define FULL_MASK 0xffffffff

const int WARP = 32; 

__global__ void gemv_int4_kernel_naive(const uint32_t* d_q_mat, const float* d_scales, const float* d_vec, float* d_out_q, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; 
    if(row < rows) {
        float sum = 0.0f;
        for(int i = 0; i < cols; i += 8) {
            uint32_t weights = d_q_mat[row * (cols / 8) + (i / 8)];
            for(int j = 0; j < 8; j++) {
                int w_int4 = (weights >> (4 * j)) & 0xF;
                if(w_int4 > 7) w_int4 -= 16;
                int idx = (row * cols + i + j) / 128;
                float w_float = w_int4 * d_scales[idx];
                sum += w_float * d_vec[i + j];
            }     
        }
        d_out_q[row] = sum;
    }
}

__global__ void gemv_int4_kernel_optimized(const uint32_t* d_q_mat, const float* d_scales, const float* d_vec, float* d_out_q, int rows, int cols) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / WARP;
}

void run_gemv_int4_kernel_naive(const uint32_t* d_q_mat, const float* d_scales, const float* d_vec, float* d_out_q, int rows, int cols) {
    int block_size = 256;
    int grid_size = (rows + block_size - 1) / block_size;
    gemv_int4_kernel_naive<<<grid_size, block_size>>>(d_q_mat, d_scales, d_vec, d_out_q, rows, cols);
    cudaDeviceSynchronize();
}

void run_gemv_int4_kernel_optimized(const uint32_t* d_q_mat, const float* d_scales, const float* d_vec, float* d_out_q, int rows, int cols) {
    int block_size = 256;
    int grid_size = (rows + block_size - 1) / block_size;
    gemv_int4_kernel_optimized<<<grid_size, block_size>>>(d_q_mat, d_scales, d_vec, d_out_q, rows, cols);
    cudaDeviceSynchronize();
}