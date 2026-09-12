#include "../transformer/transformer.h"

__global__ void swiglu_kernel(float* gate, float* up, float* result, int d) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < d) {
        float x = gate[id];
        float silu = x / (1.0 + expf(-x));
        result[id] = silu * up[id];
    }
}

void run_swiglu_kernel(float* gate, float* up, float* result, int d) {
    int block_size = 256; 
    int grid_size = (d + block_size - 1) / block_size;
    swiglu_kernel<<<grid_size, block_size>>>(gate, up, result, d);
    cudaDeviceSynchronize();
}