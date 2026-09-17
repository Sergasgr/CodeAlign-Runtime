#include "kv_cache_ops.h"
#include <cuda_runtime.h>

__global__ void append_kv_cache_kernel(const float* k_src, const float* v_src, float* k_cache, float* v_cache, int current_seq_len, int max_seq_len, int head_dim, int d, int num_tokens) { // [NUM_HEADS, max_seq_len, HEAD_DIM]
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < num_tokens * d) {
        int t = id / d;
        int feat_id = id % d;
        int head_idx = feat_id / head_dim;
        int inner_feat = feat_id % head_dim;
        int target_idx = head_idx * (max_seq_len * head_dim) + (current_seq_len + t) * head_dim + inner_feat;
        
        k_cache[target_idx] = k_src[id];
        v_cache[target_idx] = v_src[id];
    }
}

void run_append_kv_cache(const float* k_src, const float* v_src,float* k_cache, float* v_cache, int current_seq_len, int max_seq_len, int head_dim, int d, int num_tokens) {
    int threads = 256;
    int blocks = (num_tokens * d + threads - 1) / threads;
    append_kv_cache_kernel<<<blocks, threads>>>(k_src, v_src, k_cache, v_cache, current_seq_len, max_seq_len, head_dim, d, num_tokens);
    cudaDeviceSynchronize();
}