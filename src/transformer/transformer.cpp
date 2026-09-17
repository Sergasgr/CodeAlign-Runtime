#include "transformer.h"
#include "../gemv/gemv.h"
#include "../gemm/gemm.h"
#include "../ops/flash_decoding.h"
#include "../ops/residual_operations.h"
#include "../ops/kv_cache_ops.h"
#include <cmath>
#include <cuda_runtime.h>

constexpr int D = 896;
constexpr int HEAD_DIM = 64;
constexpr int CHUNK_SIZE = 256;
constexpr int NUM_HEADS = 14;

constexpr int NUM_SPLITS = 4;

void run_quantized_linear(const QuantizedLinear& proj, float* input, float* output, float* buffer, int num_tokens) { 
    if(num_tokens == 1) {
        run_gemv_int4_optimized_kernel(proj.q_weight, proj.scales, input, output, proj.out_features, proj.in_features);
    } else {
        run_gemm_int4_splitk_partial_kernel(input, proj.q_weight, proj.scales, buffer, num_tokens, NUM_SPLITS, proj.in_features, proj.out_features);
        run_gemm_int4_splitk_final_kernel(buffer, output, num_tokens, proj.out_features, NUM_SPLITS);
    }
}

void forward_transformer_block(float* hidden_states, const TransformerBlockWeights& weights, LayerKVCache& kv_cache, LayerBuffers& buffers, int num_tokens) {
    // ATTENTION
    for(int t = 0; t < num_tokens; t++) run_RMSNorm_kernel(hidden_states + (t * D), weights.attn_norm_weight, buffers.norm_result + (t * D), D); 
    
    run_quantized_linear(weights.q_proj, buffers.norm_result, buffers.q_result, buffers.partial_O, num_tokens);
    run_quantized_linear(weights.k_proj, buffers.norm_result, buffers.k_result, buffers.partial_O, num_tokens);
    run_quantized_linear(weights.v_proj, buffers.norm_result, buffers.v_result, buffers.partial_O, num_tokens);

    for(int t = 0; t < num_tokens; t++) run_RoPE_kernel(buffers.q_result + (t * D), kv_cache.current_seq_len + t, D, HEAD_DIM);
    for(int t = 0; t < num_tokens; t++) run_RoPE_kernel(buffers.k_result + (t * D), kv_cache.current_seq_len + t, D, HEAD_DIM);

    run_append_kv_cache(buffers.k_result, buffers.v_result, kv_cache.k_cache, kv_cache.v_cache, kv_cache.current_seq_len, kv_cache.max_seq_len, HEAD_DIM, D, num_tokens);

    for (int t = 0; t < num_tokens; t++) {
        int S = kv_cache.current_seq_len + t + 1;
        int num_chunks = (S + CHUNK_SIZE - 1) / CHUNK_SIZE;

        for (int h = 0; h < NUM_HEADS; h++) { 
            int offset = h * HEAD_DIM; 
            int kv_offset = h * kv_cache.max_seq_len * HEAD_DIM;

            run_flash_decoding_partial(
                buffers.q_result + (t * D) + offset, 
                kv_cache.k_cache + kv_offset, 
                kv_cache.v_cache + kv_offset, 
                buffers.partial_O, buffers.partial_lse, 
                HEAD_DIM, S, CHUNK_SIZE
            );

            run_flash_decoding_final(
                buffers.partial_O, buffers.partial_lse, 
                buffers.attn_result + (t * D) + offset, 
                HEAD_DIM, num_chunks
            );
        }
    }

    run_quantized_linear(weights.o_proj, buffers.attn_result, buffers.o_result, buffers.partial_O, num_tokens);

    run_add_residual_kernel(hidden_states, buffers.o_result, num_tokens * D);

    // MLP
    for(int t = 0; t < num_tokens; t++) run_RMSNorm_kernel(hidden_states + (t * D), weights.mlp_norm_weight, buffers.mlp_norm_result + (t * D), D); 
    
    run_quantized_linear(weights.gate_proj, buffers.mlp_norm_result, buffers.gate_result, buffers.partial_O, num_tokens);
    run_quantized_linear(weights.up_proj, buffers.mlp_norm_result, buffers.up_result, buffers.partial_O, num_tokens);

    run_swiglu_kernel(buffers.gate_result, buffers.up_result, buffers.swiglu_result, num_tokens * weights.gate_proj.out_features);

    run_quantized_linear(weights.down_proj, buffers.swiglu_result, buffers.down_result, buffers.partial_O, num_tokens);

    run_add_residual_kernel(hidden_states, buffers.down_result, num_tokens * D);

    kv_cache.current_seq_len += num_tokens;
}
