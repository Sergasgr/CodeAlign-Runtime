#include "memory.h"
#include "transformer.h"
#include <cuda_runtime.h>

void init_kv_cache(LayerKVCache& cache, int max_seq_len, int d) {
    cache.max_seq_len = max_seq_len;
    cache.current_seq_len = 0;

    cudaMalloc((void**)&cache.k_cache, d * max_seq_len * sizeof(float));
    cudaMalloc((void**)&cache.v_cache, d * max_seq_len * sizeof(float));
}

void init_buffers(LayerBuffers& buffers, int d, int intermediate_dim, int max_seq_len) {
    cudaMalloc((void**)&buffers.norm_result, d * sizeof(float));     
    cudaMalloc((void**)&buffers.q_result, d * sizeof(float));     
    cudaMalloc((void**)&buffers.k_result, d * sizeof(float));     
    cudaMalloc((void**)&buffers.v_result, d * sizeof(float));     
    cudaMalloc((void**)&buffers.attn_result, d * sizeof(float));     
    cudaMalloc((void**)&buffers.o_result, d * sizeof(float));     
    cudaMalloc((void**)&buffers.mlp_norm_result, d * sizeof(float));     
    cudaMalloc((void**)&buffers.down_result, d * sizeof(float));   

    cudaMalloc((void**)&buffers.gate_result, intermediate_dim * sizeof(float));     
    cudaMalloc((void**)&buffers.up_result, intermediate_dim * sizeof(float));     
    cudaMalloc((void**)&buffers.swiglu_result, intermediate_dim * sizeof(float));  
    
    int max_chunks = (max_seq_len + 256 - 1) / 256;
    cudaMalloc((void**)&buffers.partial_O, max_chunks * d * sizeof(float));     
    cudaMalloc((void**)&buffers.partial_lse, max_chunks * sizeof(float));  
}