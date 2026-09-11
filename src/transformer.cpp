#include "transformer.h"
#include "gemv.h"
#include "flash_decoding.h"
#include "residual_operations.h"
#include <cmath>

const int D = 896;
const int HEAD_DIM = 64;
const int CHUNK_SIZE = 256;

void forward_transformer_block(float* hidden_states, const TransformerBlockWeights& weights, LayerKVCache& kv_cache, LayerBuffers& buffers) {
    float* hidden_states_original = hidden_states; 

    run_RMSNorm_kernel(hidden_states, weights.attn_norm_weight, buffers.norm_result, D);
    
    run_gemv_int4_optimized_kernel(weights.q_proj.q_weight, weights.q_proj.scales, buffers.norm_result, buffers.q_result, weights.q_proj.out_features, weights.q_proj.in_features);
    run_gemv_int4_optimized_kernel(weights.k_proj.q_weight, weights.k_proj.scales, buffers.norm_result, buffers.k_result, weights.k_proj.out_features, weights.k_proj.in_features);
    run_gemv_int4_optimized_kernel(weights.v_proj.q_weight, weights.v_proj.scales, buffers.norm_result, buffers.v_result, weights.v_proj.out_features, weights.v_proj.in_features);

    run_RoPE_kernel(buffers.q_result, kv_cache.current_seq_len, D, HEAD_DIM);
    run_RoPE_kernel(buffers.k_result, kv_cache.current_seq_len, D, HEAD_DIM);

    cudaMemcpy(kv_cache.k_cache + (kv_cache.current_seq_len * D), buffers.k_result, D * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(kv_cache.v_cache + (kv_cache.current_seq_len * D), buffers.v_result, D * sizeof(float), cudaMemcpyDeviceToDevice);

    int S = kv_cache.current_seq_len + 1;
    int num_chunks = (S + CHUNK_SIZE - 1) / CHUNK_SIZE;

    run_flash_decoding_partial(buffers.q_result, kv_cache.k_cache, kv_cache.v_cache, buffers.partial_O, buffers.partial_lse, D, S, CHUNK_SIZE);
    run_flash_decoding_final(buffers.partial_O, buffers.partial_lse, buffers.attn_result, D, num_chunks);   

    run_gemv_int4_optimized_kernel(weights.o_proj.q_weight, weights.o_proj.scales, buffers.attn_result, buffers.o_result, weights.o_proj.out_features, weights.o_proj.in_features);

    run_add_residual_kernel(hidden_states, buffers.o_result, D);
}
