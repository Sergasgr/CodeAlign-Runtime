#pragma once
#include <cstdint>

struct QuantizedLinear {
    uint32_t* q_weight; 
    float* scales; 
    int in_features;
    int out_features;
};

struct TransformerBlockWeights {
    float* attn_norm_weight;
    float* mlp_norm_weight;
    QuantizedLinear q_proj;
    QuantizedLinear k_proj;
    QuantizedLinear v_proj;
    QuantizedLinear o_proj;
    QuantizedLinear gate_proj;
    QuantizedLinear up_proj;
    QuantizedLinear down_proj;
};

struct LayerKVCache {
    float* k_cache; 
    float* v_cache;
    int max_seq_len;    
    int current_seq_len; 
};

struct LayerBuffers {
    float* norm_result;
    float* q_result;
    float* k_result;
    float* v_result;
    float* attn_result;
    float* partial_O;
    float* partial_lse;
    float* o_result;
    float* mlp_norm_result;
    float* gate_result;
    float* up_result;
    float* swiglu_result;
    float* down_result;
};

void forward_transformer_block(float* hidden_states, const TransformerBlockWeights& weights, LayerKVCache& kv_cache, LayerBuffers& buffers);
void run_RMSNorm_kernel(float* current_token, float* weights, float* result, int d);
void run_RoPE_kernel(float* vector, int pos, int d, int head_dim);
void run_swiglu_kernel(float* gate, float* up, float* result, int d);