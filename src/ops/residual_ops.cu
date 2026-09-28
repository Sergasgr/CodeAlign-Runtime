#include "residual_operations.h"

__global__ void add_residual_kernel(float* base_vector, float* add_vector, int d) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < d) {
        base_vector[id] += add_vector[id];
    }
}

__global__ void add_bias_kernel(float* x, const float* bias, int num_tokens, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < num_tokens * n) {
        x[id] += bias[id % n];
    }
}

void run_add_residual_kernel(float* base_vector, float* add_vector, int d) {
    int block_size = 256;
    int grid_size = (d + block_size - 1) / block_size;
    add_residual_kernel<<<grid_size, block_size>>>(base_vector, add_vector, d);
    cudaDeviceSynchronize();
}

void run_add_bias_kernel(float* x, const float* bias, int num_tokens, int n) {
    int block_size = 256;
    int grid_size = (num_tokens * n + block_size - 1) / block_size;
    add_bias_kernel<<<grid_size, block_size>>>(x, bias, num_tokens, n);
    cudaDeviceSynchronize();
}