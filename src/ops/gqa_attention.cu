#include "gqa_attention.h"
#include "../include/codealign_cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {
    constexpr int WARP_SIZE = 32;
    constexpr int PARTIAL_THREADS = GQA_WARPS * WARP_SIZE;
    constexpr int TOKENS_IN_FLIGHT = 4;      
    constexpr int BLOCKS_PER_SM = 2;       
    constexpr int MAX_DEVICES = 64;

    __device__ __forceinline__ void load_vec(const float* ptr, float (&x)[2]) {
        const float2 v = *reinterpret_cast<const float2*>(ptr);
        x[0] = v.x; x[1] = v.y;
    }

    __device__ __forceinline__ void load_vec(const float* ptr, float (&x)[4]) {
        const float4 v = *reinterpret_cast<const float4*>(ptr);
        x[0] = v.x; x[1] = v.y; x[2] = v.z; x[3] = v.w;
    }

    template <int D, int G>
    __global__ void __launch_bounds__(PARTIAL_THREADS)
    gqa_attention_partial_kernel(const float* __restrict__ q, const float* __restrict__ k_cache,
                                const float* __restrict__ v_cache, float* __restrict__ partial_o,
                                float* __restrict__ partial_lse, int past_len, int num_heads, int max_seq_len,
                                int chunk, int num_chunks, float scale) {
        constexpr int E = D / WARP_SIZE;

        const int c = blockIdx.x;
        const int kvh = blockIdx.y;
        const int t = blockIdx.z;
        const int seq_len = past_len + t + 1;          
        const int lo = c * chunk;
        if (lo >= seq_len) return;                    
        const int hi = min(lo + chunk, seq_len);

        const int warp = threadIdx.x / WARP_SIZE;
        const int lane = threadIdx.x % WARP_SIZE;

        float qr[G][E];
        float acc[G][E];
        float m[G];
        float l[G];
        const float* q_group = q + (static_cast<size_t>(t) * num_heads + static_cast<size_t>(kvh) * G) * D + lane * E;
    #pragma unroll
        for (int g = 0; g < G; ++g) {
            load_vec(q_group + g * D, qr[g]);
    #pragma unroll
            for (int e = 0; e < E; ++e) { qr[g][e] *= scale; acc[g][e] = 0.0f; }
            m[g] = -INFINITY;
            l[g] = 0.0f;
        }

        const size_t head_offset = static_cast<size_t>(kvh) * max_seq_len * D + lane * E;
        const float* k_head = k_cache + head_offset;
        const float* v_head = v_cache + head_offset;

        for (int base = lo + warp; base < hi; base += GQA_WARPS * TOKENS_IN_FLIGHT) {
            float kr[TOKENS_IN_FLIGHT][E];
            float vr[TOKENS_IN_FLIGHT][E];
    #pragma unroll
            for (int u = 0; u < TOKENS_IN_FLIGHT; ++u) {
                const int j = base + u * GQA_WARPS;
                if (j < hi) {
                    load_vec(k_head + static_cast<size_t>(j) * D, kr[u]);
                    load_vec(v_head + static_cast<size_t>(j) * D, vr[u]);
                }
            }
    #pragma unroll
            for (int u = 0; u < TOKENS_IN_FLIGHT; ++u) {
                if (base + u * GQA_WARPS >= hi) break;
                float s[G];
    #pragma unroll
                for (int g = 0; g < G; ++g) {
                    s[g] = 0.0f;
    #pragma unroll
                    for (int e = 0; e < E; ++e) s[g] = fmaf(qr[g][e], kr[u][e], s[g]);
                }
    #pragma unroll
                for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    #pragma unroll
                    for (int g = 0; g < G; ++g) s[g] += __shfl_xor_sync(0xffffffffu, s[g], offset);  
                }
    #pragma unroll
                for (int g = 0; g < G; ++g) {
                    const float m_new = fmaxf(m[g], s[g]);
                    const float corr = __expf(m[g] - m_new);          
                    const float p = __expf(s[g] - m_new);
                    l[g] = fmaf(l[g], corr, p);
    #pragma unroll
                    for (int e = 0; e < E; ++e) acc[g][e] = fmaf(acc[g][e], corr, p * vr[u][e]);
                    m[g] = m_new;
                }
            }
        }

        __shared__ float m_s[GQA_WARPS][G];
        __shared__ float l_s[GQA_WARPS][G];
        __shared__ float acc_s[GQA_WARPS][G][D];
    #pragma unroll
        for (int g = 0; g < G; ++g) {
            if (lane == 0) { m_s[warp][g] = m[g]; l_s[warp][g] = l[g]; }
    #pragma unroll
            for (int e = 0; e < E; ++e) acc_s[warp][g][lane * E + e] = acc[g][e];
        }
        __syncthreads();

        for (int idx = threadIdx.x; idx < G * D; idx += PARTIAL_THREADS) {
            const int g = idx / D;
            const int d = idx % D;
            float m_max = -INFINITY;
    #pragma unroll
            for (int w = 0; w < GQA_WARPS; ++w) m_max = fmaxf(m_max, m_s[w][g]);
            float l_sum = 0.0f;
            float o_sum = 0.0f;
    #pragma unroll
            for (int w = 0; w < GQA_WARPS; ++w) {
                const float f = __expf(m_s[w][g] - m_max);
                l_sum = fmaf(l_s[w][g], f, l_sum);
                o_sum = fmaf(acc_s[w][g][d], f, o_sum);
            }
            const size_t row = (static_cast<size_t>(t) * num_heads + static_cast<size_t>(kvh) * G + g) * num_chunks + c;
            partial_o[row * D + d] = o_sum / l_sum;
            if (d == 0) partial_lse[row] = m_max + __logf(l_sum);
        }
    }

    template <int D>
    __global__ void __launch_bounds__(D)
    gqa_attention_final_kernel(const float* __restrict__ partial_o, const float* __restrict__ partial_lse,
                            float* __restrict__ out, int past_len, int num_heads, int chunk, int num_chunks) {
        const int h = blockIdx.x;
        const int t = blockIdx.y;
        const int d = threadIdx.x;
        const int seq_len = past_len + t + 1;
        const int visible_chunks = (seq_len + chunk - 1) / chunk;    
        const size_t row0 = (static_cast<size_t>(t) * num_heads + h) * num_chunks;

        float m_max = -INFINITY;
        for (int c = 0; c < visible_chunks; ++c) m_max = fmaxf(m_max, partial_lse[row0 + c]);
        float w_sum = 0.0f;
        float o_sum = 0.0f;
        for (int c = 0; c < visible_chunks; ++c) {
            const float w = __expf(partial_lse[row0 + c] - m_max);
            w_sum += w;
            o_sum = fmaf(w, partial_o[(row0 + c) * D + d], o_sum);
        }
        out[(static_cast<size_t>(t) * num_heads + h) * D + d] = o_sum / w_sum;
    }

    int ceil_div(int a, int b) { return (a + b - 1) / b; }
    int round_up(int x, int multiple) { return ceil_div(x, multiple) * multiple; }

    int num_sms_of_current_device() {
        static int cached[MAX_DEVICES] = {};
        int device = 0;
        CODEALIGN_CHECK_CUDA(cudaGetDevice(&device));
        if (device < 0 || device >= MAX_DEVICES) device = 0;
        if (cached[device] == 0) {
            int sms = 0;
            CODEALIGN_CHECK_CUDA(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
            cached[device] = sms;
        }
        return cached[device];
    }

    [[noreturn]] void fail(const std::string& message) {
        throw std::runtime_error("run_gqa_attention: " + message);
    }

    bool aligned_to(const void* ptr, size_t bytes) {
        return reinterpret_cast<std::uintptr_t>(ptr) % bytes == 0;
    }

    void validate(const GqaAttentionParams& p) {
        if (!p.q || !p.k_cache || !p.v_cache || !p.out || !p.partial_o || !p.partial_lse) fail("null pointer");
        if (p.num_tokens < 1) fail("num_tokens must be >= 1, got " + std::to_string(p.num_tokens));
        if (p.past_len < 0) fail("past_len must be >= 0, got " + std::to_string(p.past_len));
        if (p.num_kv_heads < 1 || p.num_heads < 1 || p.num_heads % p.num_kv_heads != 0) {
            fail("num_heads (" + std::to_string(p.num_heads) + ") must be a multiple of num_kv_heads (" +
                std::to_string(p.num_kv_heads) + ")");
        }
        if (p.num_heads / p.num_kv_heads > GQA_MAX_GROUP) {
            fail("num_heads / num_kv_heads = " + std::to_string(p.num_heads / p.num_kv_heads) + " > GQA_MAX_GROUP (" +
                std::to_string(GQA_MAX_GROUP) + ")");
        }
        if (p.head_dim != 64 && p.head_dim != 128) fail("head_dim must be 64 or 128, got " + std::to_string(p.head_dim));
        if (p.past_len + p.num_tokens > p.max_seq_len) {
            fail("past_len + num_tokens = " + std::to_string(p.past_len + p.num_tokens) + " > max_seq_len = " +
                std::to_string(p.max_seq_len));
        }
        if (p.max_chunks < 1) fail("max_chunks must be >= 1");
        const size_t vec_bytes = (p.head_dim / WARP_SIZE) * sizeof(float);   // float2 (D=64) or float4 (D=128) loads
        if (!aligned_to(p.q, vec_bytes) || !aligned_to(p.k_cache, vec_bytes) || !aligned_to(p.v_cache, vec_bytes)) {
            fail("q, k_cache and v_cache must be " + std::to_string(vec_bytes) + "-byte aligned");
        }
    }

    template <int D, int G>
    void launch_partial(const GqaAttentionParams& p, int chunk, int num_chunks, float scale, cudaStream_t stream) {
        const dim3 grid(num_chunks, p.num_kv_heads, p.num_tokens);
        CODEALIGN_KERNEL_LAUNCH((gqa_attention_partial_kernel<D, G>), grid, dim3(PARTIAL_THREADS), 0, stream,
                                p.q, p.k_cache, p.v_cache, p.partial_o, p.partial_lse, p.past_len, p.num_heads,
                                p.max_seq_len, chunk, num_chunks, scale);
    }

    template <int D>
    void launch_kernels(const GqaAttentionParams& p, int chunk, int num_chunks, cudaStream_t stream) {
        const float scale = 1.0f / std::sqrt(static_cast<float>(D));
        switch (p.num_heads / p.num_kv_heads) {  
            case 1: launch_partial<D, 1>(p, chunk, num_chunks, scale, stream); break;
            case 2: launch_partial<D, 2>(p, chunk, num_chunks, scale, stream); break;
            case 3: launch_partial<D, 3>(p, chunk, num_chunks, scale, stream); break;
            case 4: launch_partial<D, 4>(p, chunk, num_chunks, scale, stream); break;
            case 5: launch_partial<D, 5>(p, chunk, num_chunks, scale, stream); break;
            case 6: launch_partial<D, 6>(p, chunk, num_chunks, scale, stream); break;
            case 7: launch_partial<D, 7>(p, chunk, num_chunks, scale, stream); break;
            case 8: launch_partial<D, 8>(p, chunk, num_chunks, scale, stream); break;
            default: fail("unsupported group size");
        }
        const dim3 final_grid(p.num_heads, p.num_tokens);
        CODEALIGN_KERNEL_LAUNCH(gqa_attention_final_kernel<D>, final_grid, dim3(D), 0, stream,
                                p.partial_o, p.partial_lse, p.out, p.past_len, p.num_heads, chunk, num_chunks);
    }
} 

static_assert(GQA_MAX_GROUP == 8, "launch_kernels dispatches group sizes 1..8");

int gqa_attention_chunk_size(int num_tokens, int past_len, int num_kv_heads, int max_chunks) {
    const int seq_max = past_len + num_tokens;
    const int target_blocks = BLOCKS_PER_SM * num_sms_of_current_device();
    const int chunks_wanted = std::max(1, ceil_div(target_blocks, num_kv_heads * num_tokens));
    int chunk = std::max(GQA_MIN_CHUNK, round_up(ceil_div(seq_max, chunks_wanted), 32));
    if (ceil_div(seq_max, chunk) > max_chunks) {
        chunk = round_up(ceil_div(seq_max, max_chunks), 32);  
    }
    return chunk;
}

int gqa_attention_num_chunks(int num_tokens, int past_len, int num_kv_heads, int max_chunks) {
    return ceil_div(past_len + num_tokens, gqa_attention_chunk_size(num_tokens, past_len, num_kv_heads, max_chunks));
}

void run_gqa_attention(const GqaAttentionParams& p, cudaStream_t stream) {
    validate(p);
    const int chunk = gqa_attention_chunk_size(p.num_tokens, p.past_len, p.num_kv_heads, p.max_chunks);
    const int num_chunks = ceil_div(p.past_len + p.num_tokens, chunk);
    if (num_chunks > p.max_chunks) fail("internal error: num_chunks > max_chunks");
    if (p.head_dim == 64) {
        launch_kernels<64>(p, chunk, num_chunks, stream);
    } else {
        launch_kernels<128>(p, chunk, num_chunks, stream);
    }
}