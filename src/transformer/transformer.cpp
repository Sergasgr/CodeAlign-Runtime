#include "transformer.h"
#include "../gemv/gemv.h"
#include "../ops/flash_decoding.h"
#include "../ops/residual_operations.h"
#include "../ops/kv_cache_ops.h"
#include <cmath>
#include <cuda_runtime.h>

constexpr int D = 896;
constexpr int HEAD_DIM = 64;
constexpr int CHUNK_SIZE = 256;
constexpr int NUM_HEADS = 14;

void forward_transformer_block(float* hidden_states, const TransformerBlockWeights& weights, LayerKVCache& kv_cache, LayerBuffers& buffers) {
    // ATTENTION
    run_RMSNorm_kernel(hidden_states, weights.attn_norm_weight, buffers.norm_result, D);
    
    run_gemv_int4_optimized_kernel(weights.q_proj.q_weight, weights.q_proj.scales, buffers.norm_result, buffers.q_result, weights.q_proj.out_features, weights.q_proj.in_features);
    run_gemv_int4_optimized_kernel(weights.k_proj.q_weight, weights.k_proj.scales, buffers.norm_result, buffers.k_result, weights.k_proj.out_features, weights.k_proj.in_features);
    run_gemv_int4_optimized_kernel(weights.v_proj.q_weight, weights.v_proj.scales, buffers.norm_result, buffers.v_result, weights.v_proj.out_features, weights.v_proj.in_features);

    run_RoPE_kernel(buffers.q_result, kv_cache.current_seq_len, D, HEAD_DIM);
    run_RoPE_kernel(buffers.k_result, kv_cache.current_seq_len, D, HEAD_DIM);

    run_append_kv_cache(buffers.k_result, buffers.v_result, kv_cache.k_cache, kv_cache.v_cache, kv_cache.current_seq_len, kv_cache.max_seq_len, HEAD_DIM, D);

    int S = kv_cache.current_seq_len + 1;
    int num_chunks = (S + CHUNK_SIZE - 1) / CHUNK_SIZE;

    for (int h = 0; h < NUM_HEADS; h++) { 
        int offset = h * HEAD_DIM;
        int kv_offset = h * kv_cache.max_seq_len * HEAD_DIM;

        run_flash_decoding_partial(
            buffers.q_result + offset,
            kv_cache.k_cache + kv_offset,
            kv_cache.v_cache + kv_offset,
            buffers.partial_O, buffers.partial_lse,
            HEAD_DIM, S, CHUNK_SIZE
        );

        run_flash_decoding_final(
            buffers.partial_O, buffers.partial_lse,
            buffers.attn_result + offset,
            HEAD_DIM, num_chunks
        );
    }

    run_gemv_int4_optimized_kernel(weights.o_proj.q_weight, weights.o_proj.scales, buffers.attn_result, buffers.o_result, weights.o_proj.out_features, weights.o_proj.in_features);

    run_add_residual_kernel(hidden_states, buffers.o_result, D);

    // MLP
    run_RMSNorm_kernel(hidden_states, weights.mlp_norm_weight, buffers.mlp_norm_result, D);

    run_gemv_int4_optimized_kernel(weights.gate_proj.q_weight, weights.gate_proj.scales, buffers.mlp_norm_result, buffers.gate_result, weights.gate_proj.out_features, weights.gate_proj.in_features);
    run_gemv_int4_optimized_kernel(weights.up_proj.q_weight, weights.up_proj.scales, buffers.mlp_norm_result, buffers.up_result, weights.up_proj.out_features, weights.up_proj.in_features);

    run_swiglu_kernel(buffers.gate_result, buffers.up_result, buffers.swiglu_result, weights.gate_proj.out_features);

    run_gemv_int4_optimized_kernel(weights.down_proj.q_weight, weights.down_proj.scales, buffers.swiglu_result, buffers.down_result, weights.down_proj.out_features, weights.down_proj.in_features);

    run_add_residual_kernel(hidden_states, buffers.down_result, D);

    kv_cache.current_seq_len++;
}
