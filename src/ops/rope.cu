#include "../transformer/transformer.h"
#include <cmath>

__global__ void RoPE_kernel(float* vector, int pos, int d, int head_dim) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < d / 2) {
        int current_head = id / (head_dim / 2);
        int even_idx = id % (head_dim / 2);
        float inv_freq = (float)(1.0 / pow(1000000.0, (double)(2 * even_idx) / (double)head_dim));
        float angle = (float)pos * inv_freq;
        double s, c;
        sincos((double)angle, &s, &c);
        float sin_a = (float)s, cos_a = (float)c;
        int p1 = current_head * head_dim + even_idx;
        int p2 = p1 + (head_dim / 2);
        float x = vector[p1], y = vector[p2];
        vector[p1] = x * cos_a - y * sin_a;
        vector[p2] = x * sin_a + y * cos_a;
    }
}

void run_RoPE_kernel(float* vector, int pos, int d, int head_dim) {
    int block_size = 256;
    int grid_size = (d / 2 + block_size - 1) / block_size;
    RoPE_kernel<<<grid_size, block_size>>>(vector, pos, d, head_dim);
    cudaDeviceSynchronize();
}