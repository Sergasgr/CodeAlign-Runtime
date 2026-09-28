#include "memory.h"
#include "transformer.h"
#include "cuda_check.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>

namespace {
    float* alloc_floats(size_t count) {
        float* ptr = nullptr;
        CUDA_CHECK(cudaMalloc((void**)&ptr, count * sizeof(float)));
        CUDA_CHECK(cudaMemset(ptr, 0, count * sizeof(float)));
        return ptr;
    }
}

void init_kv_cache(LayerKVCache& cache, int max_seq_len, const ModelDims& dims) {
    cache.max_seq_len = max_seq_len;
    cache.current_seq_len = 0;

    const size_t elems = static_cast<size_t>(dims.kv_dim()) * max_seq_len;
    cache.k_cache = alloc_floats(elems);
    cache.v_cache = alloc_floats(elems);
}

void init_buffers(LayerBuffers& buffers, const ModelDims& dims, int max_seq_len) {
    const size_t max_tokens = MAX_TOKENS_PER_FORWARD;
    const size_t hidden = dims.hidden, inter = dims.intermediate, q_dim = dims.q_dim(), kv_dim = dims.kv_dim();

    buffers.norm_result     = alloc_floats(max_tokens * hidden);
    buffers.q_result        = alloc_floats(max_tokens * q_dim);
    buffers.k_result        = alloc_floats(max_tokens * kv_dim);
    buffers.v_result        = alloc_floats(max_tokens * kv_dim);
    buffers.attn_result     = alloc_floats(max_tokens * q_dim);
    buffers.o_result        = alloc_floats(max_tokens * hidden);
    buffers.mlp_norm_result = alloc_floats(max_tokens * hidden);
    buffers.down_result     = alloc_floats(max_tokens * hidden);

    buffers.gate_result   = alloc_floats(max_tokens * inter);
    buffers.up_result     = alloc_floats(max_tokens * inter);
    buffers.swiglu_result = alloc_floats(max_tokens * inter);

    const size_t max_chunks = (max_seq_len + FLASH_CHUNK_SIZE - 1) / FLASH_CHUNK_SIZE;
    const size_t widest_out = std::max({hidden, inter, q_dim, kv_dim});
    const size_t flash_elems = max_chunks * dims.head_dim;
    const size_t splitk_elems = static_cast<size_t>(NUM_SPLITS) * max_tokens * widest_out;
    buffers.partial_O   = alloc_floats(std::max(flash_elems, splitk_elems));
    buffers.partial_lse = alloc_floats(max_chunks);
}

void free_kv_cache(LayerKVCache& cache) {
    cudaFree(cache.k_cache);
    cudaFree(cache.v_cache);
    cache.k_cache = nullptr;
    cache.v_cache = nullptr;
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
    buffers = LayerBuffers{};
}