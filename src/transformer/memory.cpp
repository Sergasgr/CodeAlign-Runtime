#include "memory.h"
#include "transformer.h"
#include <cuda_runtime.h>
#include <algorithm>

constexpr int MAX_DRAFT_TOKENS = 10;
constexpr int NUM_SPLITS = 4;

void init_kv_cache(LayerKVCache& cache, int max_seq_len, int d) {
    cache.max_seq_len = max_seq_len;
    cache.current_seq_len = 0;

    cudaMalloc((void**)&cache.k_cache, d * max_seq_len * sizeof(float));
    cudaMalloc((void**)&cache.v_cache, d * max_seq_len * sizeof(float));
}

void init_buffers(LayerBuffers& buffers, int d, int intermediate_dim, int max_seq_len) {
    int max_tokens = MAX_DRAFT_TOKENS + 1;  // draft tokens + 1 current token

    cudaMalloc((void**)&buffers.norm_result, max_tokens * d * sizeof(float));     
    cudaMalloc((void**)&buffers.q_result, max_tokens * d * sizeof(float));     
    cudaMalloc((void**)&buffers.k_result, max_tokens * d * sizeof(float));     
    cudaMalloc((void**)&buffers.v_result, max_tokens * d * sizeof(float));     
    cudaMalloc((void**)&buffers.attn_result, max_tokens * d * sizeof(float));     
    cudaMalloc((void**)&buffers.o_result, max_tokens * d * sizeof(float));     
    cudaMalloc((void**)&buffers.mlp_norm_result, max_tokens * d * sizeof(float));     
    cudaMalloc((void**)&buffers.down_result, max_tokens * d * sizeof(float));   

    cudaMalloc((void**)&buffers.gate_result, max_tokens * intermediate_dim * sizeof(float));     
    cudaMalloc((void**)&buffers.up_result, max_tokens * intermediate_dim * sizeof(float));     
    cudaMalloc((void**)&buffers.swiglu_result, max_tokens * intermediate_dim * sizeof(float));  
    
    int max_chunks = (max_seq_len + 256 - 1) / 256;
    cudaMalloc((void**)&buffers.partial_O, std::max(max_chunks * d * sizeof(float), NUM_SPLITS * max_tokens * intermediate_dim * sizeof(float)));     
    cudaMalloc((void**)&buffers.partial_lse, max_chunks * sizeof(float));  
}

void free_kv_cache(LayerKVCache& cache) {
    cudaFree(cache.k_cache);
    cudaFree(cache.v_cache);
}

void free_buffers(LayerBuffers& buffers) {
    cudaFree(buffers.norm_result);     
    cudaFree(buffers.q_result);     
    cudaFree(buffers.k_result);     
    cudaFree(buffers.v_result);     
    cudaFree(buffers.attn_result);     
    cudaFree(buffers.o_result);     
    cudaFree(buffers.mlp_norm_result);     
    cudaFree(buffers.down_result);   

    cudaFree(buffers.gate_result);     
    cudaFree(buffers.up_result);     
    cudaFree(buffers.swiglu_result);  
    
    cudaFree(buffers.partial_O);     
    cudaFree(buffers.partial_lse);  
}