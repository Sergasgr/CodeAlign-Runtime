#include "transformer.h"
#include "../gemv/gemv.h"
#include "../gemm/gemm.h"
#include "../ops/gqa_attention.h"
#include "../ops/residual_operations.h"
#include "../ops/kv_cache_ops.h"
#include <cmath>
#include <cuda_runtime.h>

void run_quantized_linear(const QuantizedLinear& proj, float* input, float* output, float* buffer, int num_tokens, cudaStream_t stream) {
    if(num_tokens == 1) {
        run_gemv_int4_v2_kernel(proj.q_weight, proj.scales, input, proj.bias, output, proj.out_features, proj.in_features, stream);
        return;
    }
    run_gemm_int4_splitk_partial_kernel(input, proj.q_weight, proj.scales, buffer, num_tokens, NUM_SPLITS, proj.in_features, proj.out_features, stream);
    run_gemm_int4_splitk_final_kernel(buffer, output, num_tokens, proj.out_features, NUM_SPLITS, stream);
    if(proj.bias != nullptr) {
        run_add_bias_kernel(output, proj.bias, num_tokens, proj.out_features, stream);
    }
}

void forward_transformer_block(float* hidden_states, const TransformerBlockWeights& weights, LayerKVCache& kv_cache,
                               LayerBuffers& buffers, const ModelDims& dims, int num_tokens, cudaStream_t stream) {
    const int D = dims.hidden;
    const int q_dim = dims.q_dim();
    const int kv_dim = dims.kv_dim();
    const int head_dim = dims.head_dim;

    // ATTENTION
    run_RMSNorm_kernel(hidden_states, weights.attn_norm_weight, buffers.norm_result, D, num_tokens, stream);

    run_quantized_linear(weights.q_proj, buffers.norm_result, buffers.q_result, buffers.partial_O, num_tokens, stream);  // [T, q_dim]
    run_quantized_linear(weights.k_proj, buffers.norm_result, buffers.k_result, buffers.partial_O, num_tokens, stream);  // [T, kv_dim]
    run_quantized_linear(weights.v_proj, buffers.norm_result, buffers.v_result, buffers.partial_O, num_tokens, stream);  // [T, kv_dim]

    run_RoPE_kernel(buffers.q_result, kv_cache.current_seq_len, q_dim, head_dim, num_tokens, stream);
    run_RoPE_kernel(buffers.k_result, kv_cache.current_seq_len, kv_dim, head_dim, num_tokens, stream);

    run_append_kv_cache(buffers.k_result, buffers.v_result, kv_cache.k_cache, kv_cache.v_cache, kv_cache.current_seq_len, kv_cache.max_seq_len, head_dim, kv_dim, num_tokens, stream);

    GqaAttentionParams attn{};
    attn.q = buffers.q_result;
    attn.k_cache = kv_cache.k_cache;
    attn.v_cache = kv_cache.v_cache;
    attn.out = buffers.attn_result;
    attn.partial_o = buffers.partial_O;
    attn.partial_lse = buffers.partial_lse;
    attn.num_tokens = num_tokens;
    attn.past_len = kv_cache.current_seq_len;
    attn.num_heads = dims.num_heads;
    attn.num_kv_heads = dims.num_kv_heads;
    attn.head_dim = head_dim;
    attn.max_seq_len = kv_cache.max_seq_len;
    attn.max_chunks = MAX_ATTN_CHUNKS;
    run_gqa_attention(attn, stream);

    run_quantized_linear(weights.o_proj, buffers.attn_result, buffers.o_result, buffers.partial_O, num_tokens, stream);

    run_add_residual_kernel(hidden_states, buffers.o_result, num_tokens * D, stream);

    // MLP
    run_RMSNorm_kernel(hidden_states, weights.mlp_norm_weight, buffers.mlp_norm_result, D, num_tokens, stream);

    run_quantized_linear(weights.gate_proj, buffers.mlp_norm_result, buffers.gate_result, buffers.partial_O, num_tokens, stream);
    run_quantized_linear(weights.up_proj, buffers.mlp_norm_result, buffers.up_result, buffers.partial_O, num_tokens, stream);

    run_swiglu_kernel(buffers.gate_result, buffers.up_result, buffers.swiglu_result, num_tokens * weights.gate_proj.out_features, stream);

    run_quantized_linear(weights.down_proj, buffers.swiglu_result, buffers.down_result, buffers.partial_O, num_tokens, stream);

    run_add_residual_kernel(hidden_states, buffers.down_result, num_tokens * D, stream);

    kv_cache.current_seq_len += num_tokens;
}