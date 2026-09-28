#pragma once
#include <cstdint>

constexpr int MAX_DRAFT_TOKENS = 10; // Max tokens a single forward() can process: 1 current token + up to 10 draft tokens (Level 6).
constexpr int MAX_TOKENS_PER_FORWARD = MAX_DRAFT_TOKENS + 1; // Prefill longer than this is done in chunks of this size through the same path.
constexpr int NUM_SPLITS = 4; // Number of K-splits used by the INT4 Split-K GEMM for multi-token forwards.
constexpr int FLASH_CHUNK_SIZE = 256; // KV-cache chunk processed by one Flash-Decoding partial block.

// Architecture dimensions of one decoder block (Qwen2: GQA, bias on q/k/v, SwiGLU MLP).
struct ModelDims {
    int hidden;         // 896 for Qwen2.5-0.5B
    int intermediate;   // 4864
    int num_heads;      // 14 query heads
    int num_kv_heads;   // 2 key/value heads (GQA)
    int head_dim;       // 64
    int q_dim() const { return num_heads * head_dim; }
    int kv_dim() const { return num_kv_heads * head_dim; }
    int group_size() const { return num_heads / num_kv_heads; }  // query heads per KV head
};

struct QuantizedLinear {
    uint32_t* q_weight;   // [out_features, in_features / 8], 8 INT4 values per word
    float* scales;        // [out_features, in_features / 128], one fp32 scale per group of 128
    float* bias;          // [out_features] or nullptr
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

// Layout: [num_kv_heads, max_seq_len, head_dim] for both K and V.
struct LayerKVCache {
    float* k_cache;
    float* v_cache;
    int max_seq_len;
    int current_seq_len;
};

struct LayerBuffers {
    float* norm_result;      // [MAX_TOKENS, hidden]
    float* q_result;         // [MAX_TOKENS, q_dim]
    float* k_result;         // [MAX_TOKENS, kv_dim]
    float* v_result;         // [MAX_TOKENS, kv_dim]
    float* attn_result;      // [MAX_TOKENS, q_dim]
    float* partial_O;        // scratch: Flash-Decoding partials / Split-K partials
    float* partial_lse;      // [max_chunks]
    float* o_result;         // [MAX_TOKENS, hidden]
    float* mlp_norm_result;  // [MAX_TOKENS, hidden]
    float* gate_result;      // [MAX_TOKENS, intermediate]
    float* up_result;        // [MAX_TOKENS, intermediate]
    float* swiglu_result;    // [MAX_TOKENS, intermediate]
    float* down_result;      // [MAX_TOKENS, hidden]
};

void run_quantized_linear(const QuantizedLinear& proj, float* input, float* output, float* partial_buffer, int num_tokens);
void forward_transformer_block(float* hidden_states, const TransformerBlockWeights& weights, LayerKVCache& kv_cache, LayerBuffers& buffers, const ModelDims& dims, int num_tokens);
void run_RMSNorm_kernel(float* current_token, float* weights, float* result, int d);
void run_RoPE_kernel(float* vector, int pos, int d, int head_dim);
void run_swiglu_kernel(float* gate, float* up, float* result, int d);