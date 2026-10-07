#include "../include/codealign_cuda.h"
#include "../transformer/transformer.h"
#include <cmath>

__global__ void RoPE_kernel(float* vector, int base_pos, int d, int head_dim, int num_tokens) {
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    int half_d = d / 2;
    int half_head = head_dim / 2;
    float pos = (float)(base_pos + token_idx);

    for (int i = threadIdx.x; i < half_d; i += blockDim.x) {
        int current_head = i / half_head;
        int even_idx = i % half_head;

        float inv_freq = (float)(1.0 / pow(1000000.0, (double)(2 * even_idx) / (double)head_dim));
        float angle = pos * inv_freq;
        double s, c;
        sincos((double)angle, &s, &c);
        float sin_a = (float)s, cos_a = (float)c;

        int p1 = token_idx * d + current_head * head_dim + even_idx;
        int p2 = p1 + half_head;
        float x = vector[p1];
        float y = vector[p2];
        vector[p1] = x * cos_a - y * sin_a;
        vector[p2] = x * sin_a + y * cos_a;
    }
}

void run_RoPE_kernel(float* vector, int base_pos, int d, int head_dim, int num_tokens, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = num_tokens;
    CODEALIGN_KERNEL_LAUNCH(RoPE_kernel, grid_size, block_size, 0, stream, vector, base_pos, d, head_dim, num_tokens);
}