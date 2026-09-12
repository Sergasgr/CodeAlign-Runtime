#include "residual_operations.h"

__global__ void add_residual_kernel(float* base_vector, float* add_vector, int d) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < d) {
        base_vector[id] += add_vector[id];
    }
}

void run_add_residual_kernel(float* base_vector, float* add_vector, int d) {
    int block_size = 256; 
    int grid_size = (d + block_size - 1) / block_size;
    add_residual_kernel<<<grid_size, block_size>>>(base_vector, add_vector, d);
    cudaDeviceSynchronize();
}