#pragma once
#include <cuda_runtime.h>

constexpr int GQA_MAX_GROUP = 8;     // query heads per KV head kept in registers (Qwen2.5-0.5B: 14 / 2 = 7)
constexpr int GQA_MIN_CHUNK = 32;    // smallest number of KV positions per partial block
constexpr int GQA_WARPS = 4;         // warps per partial block

struct GqaAttentionParams {
    const float* q;          // [num_tokens, num_heads * head_dim], after RoPE
    const float* k_cache;    // [num_kv_heads, max_seq_len, head_dim]
    const float* v_cache;    // [num_kv_heads, max_seq_len, head_dim]
    float* out;              // [num_tokens, num_heads * head_dim]
    float* partial_o;        // scratch: num_tokens * num_heads * max_chunks * head_dim floats
    float* partial_lse;      // scratch: num_tokens * num_heads * max_chunks floats
    int num_tokens;
    int past_len;            // positions cached before the first token of this forward
    int num_heads;
    int num_kv_heads;
    int head_dim;            // 64 or 128
    int max_seq_len;         // KV-cache capacity (row stride between KV heads)
    int max_chunks;          // upper bound on chunks per (token, head): the scratch capacity
};

int gqa_attention_chunk_size(int num_tokens, int past_len, int num_kv_heads, int max_chunks);
int gqa_attention_num_chunks(int num_tokens, int past_len, int num_kv_heads, int max_chunks);

void run_gqa_attention(const GqaAttentionParams& p, cudaStream_t stream = 0);