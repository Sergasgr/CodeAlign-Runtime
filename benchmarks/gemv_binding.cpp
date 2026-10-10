#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <limits>
#include <optional>
#include "../src/gemv/gemv.h"
#include "../src/ops/flash_decoding.h"
#include "../src/ops/gqa_attention.h"
#include "../src/transformer/transformer.h"

namespace {
    void check_cuda_contiguous(const torch::Tensor& t, const char* name) {
        TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
        TORCH_CHECK(t.is_contiguous(), name, " must be aligned sequentially contiguous in memory");
    }

    void check_last_cuda_error(const char* where) {
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, where, ": CUDA error: ", cudaGetErrorString(err));
    }

    cudaStream_t current_stream() {
        return c10::cuda::getCurrentCUDAStream().stream();
    }

    constexpr int SINGLE_HEAD_MAX_CHUNKS = 1 << 16;

    torch::Tensor run_attention(const torch::Tensor& q, const torch::Tensor& k_cache, const torch::Tensor& v_cache, int past_len, int max_chunks, bool poison_scratch) {
        const int num_tokens = q.size(0), num_heads = q.size(1), head_dim = q.size(2);
        const int num_kv_heads = k_cache.size(0), max_seq_len = k_cache.size(1);
        TORCH_CHECK(num_tokens >= 1 && past_len >= 0 && past_len + num_tokens <= max_seq_len,
                    "need 0 <= past_len and past_len + num_tokens <= max_seq_len, got past_len=", past_len,
                    ", num_tokens=", num_tokens, ", max_seq_len=", max_seq_len);
        TORCH_CHECK(max_chunks >= 1, "max_chunks must be >= 1");
        TORCH_CHECK(num_kv_heads >= 1 && num_heads >= 1 && num_heads % num_kv_heads == 0,
                    "num_heads (", num_heads, ") must be a positive multiple of num_kv_heads (", num_kv_heads, ")");

        const int num_chunks = gqa_attention_num_chunks(num_tokens, past_len, num_kv_heads, max_chunks);
        const int64_t rows = static_cast<int64_t>(num_tokens) * num_heads * num_chunks;
        const auto options = q.options();
        const float nan = std::numeric_limits<float>::quiet_NaN();
        torch::Tensor partial_o = poison_scratch ? torch::full({rows, head_dim}, nan, options) : torch::empty({rows, head_dim}, options);
        torch::Tensor partial_lse = poison_scratch ? torch::full({rows}, nan, options) : torch::empty({rows}, options);
        torch::Tensor out = torch::empty({num_tokens, num_heads, head_dim}, options);

        GqaAttentionParams p{};
        p.q = q.data_ptr<float>();
        p.k_cache = k_cache.data_ptr<float>();
        p.v_cache = v_cache.data_ptr<float>();
        p.out = out.data_ptr<float>();
        p.partial_o = partial_o.data_ptr<float>();
        p.partial_lse = partial_lse.data_ptr<float>();
        p.num_tokens = num_tokens;
        p.past_len = past_len;
        p.num_heads = num_heads;
        p.num_kv_heads = num_kv_heads;
        p.head_dim = head_dim;
        p.max_seq_len = max_seq_len;
        p.max_chunks = max_chunks;
        run_gqa_attention(p, current_stream());   
        return out;
    }
}

namespace {
    void check_gemv_int4_inputs(const torch::Tensor& q_mat, const torch::Tensor& scales, const torch::Tensor& vec) {
        check_cuda_contiguous(q_mat, "q_mat");
        check_cuda_contiguous(scales, "scales");
        check_cuda_contiguous(vec, "vec");
        TORCH_CHECK(q_mat.scalar_type() == torch::kInt32, "q_mat must be int32 (packed INT4)");
        TORCH_CHECK(scales.scalar_type() == torch::kFloat32, "scales must be float32");
        TORCH_CHECK(vec.scalar_type() == torch::kFloat32, "vec must be float32");
        TORCH_CHECK(q_mat.dim() == 2, "q_mat must be 2D [rows, cols/8]");
        TORCH_CHECK(vec.dim() == 1, "vec must be 1D [cols], got ", vec.sizes());
        const int64_t rows = q_mat.size(0), cols = q_mat.size(1) * 8;
        TORCH_CHECK(vec.size(0) == cols, "vec has ", vec.size(0), " elements but q_mat implies cols = ", cols);
        TORCH_CHECK(cols % 128 == 0, "cols must be a multiple of the group size (128)");
        TORCH_CHECK(scales.numel() == rows * (cols / 128), "scales must have rows * cols / 128 elements");
    }
}

torch::Tensor gemv_int4_forward(torch::Tensor q_mat, torch::Tensor scales, torch::Tensor vec, std::optional<torch::Tensor> bias) {
    check_gemv_int4_inputs(q_mat, scales, vec);
    const int rows = q_mat.size(0);
    const int cols = q_mat.size(1) * 8;
    TORCH_CHECK(cols <= 12288, "GEMV v2 stages vec in 48 KB of shared memory: cols must be <= 12288, got ", cols);
    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        check_cuda_contiguous(*bias, "bias");
        TORCH_CHECK(bias->scalar_type() == torch::kFloat32 && bias->numel() == rows, "bias must be float32 [rows]");
        bias_ptr = bias->data_ptr<float>();
    }
    torch::Tensor out = torch::empty({rows}, vec.options());
    run_gemv_int4_v2_kernel(
        reinterpret_cast<const uint32_t*>(q_mat.data_ptr<int32_t>()),
        scales.data_ptr<float>(),
        vec.data_ptr<float>(),
        bias_ptr,
        out.data_ptr<float>(),
        rows,
        cols,
        current_stream()
    );
    check_last_cuda_error("gemv_int4_forward");
    return out;
}

torch::Tensor gemv_int4_v1_forward(torch::Tensor q_mat, torch::Tensor scales, torch::Tensor vec) {
    check_gemv_int4_inputs(q_mat, scales, vec);
    const int rows = q_mat.size(0);
    const int cols = q_mat.size(1) * 8;
    torch::Tensor out = torch::empty({rows}, vec.options());
    run_gemv_int4_optimized_kernel(
        reinterpret_cast<const uint32_t*>(q_mat.data_ptr<int32_t>()),
        scales.data_ptr<float>(),
        vec.data_ptr<float>(),
        out.data_ptr<float>(),
        rows,
        cols,
        current_stream()
    );
    check_last_cuda_error("gemv_int4_v1_forward");
    return out;
}

torch::Tensor flash_decoding_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    check_cuda_contiguous(Q, "Query tensor");
    check_cuda_contiguous(K, "Key tensor");
    check_cuda_contiguous(V, "Value tensor");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32 && K.scalar_type() == torch::kFloat32 && V.scalar_type() == torch::kFloat32, "Q, K and V must be float32");
    TORCH_CHECK(Q.dim() == 3 && K.dim() == 3 && V.dim() == 3, "Q, K and V must be 3D ([1,1,D] and [1,S,D])");
    TORCH_CHECK(Q.size(0) == 1 && Q.size(1) == 1, "batch=1, single query: Q must be [1, 1, D]");
    TORCH_CHECK(K.sizes() == V.sizes() && K.size(0) == 1, "K and V must both be [1, S, D]");
    TORCH_CHECK(K.size(2) == Q.size(2), "K/V head dim must match Q");
    TORCH_CHECK(K.size(1) >= 1, "S must be >= 1");

    const int S = K.size(1);
    torch::Tensor out = run_attention(Q, K, V, S - 1, SINGLE_HEAD_MAX_CHUNKS, false);
    check_last_cuda_error("flash_decoding_forward");
    return out.view({1, 1, -1});
}

torch::Tensor flash_decoding_v10_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    check_cuda_contiguous(Q, "Query tensor");
    check_cuda_contiguous(K, "Key tensor");
    check_cuda_contiguous(V, "Value tensor");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32 && K.scalar_type() == torch::kFloat32 && V.scalar_type() == torch::kFloat32, "Q, K and V must be float32");
    TORCH_CHECK(Q.dim() == 3 && K.dim() == 3 && V.dim() == 3, "Q, K and V must be 3D ([1,1,D] and [1,S,D])");
    TORCH_CHECK(Q.size(0) == 1 && Q.size(1) == 1, "batch=1, single query: Q must be [1, 1, D]");
    TORCH_CHECK(K.sizes() == V.sizes() && K.size(0) == 1, "K and V must both be [1, S, D]");
    const int D = Q.size(2);
    const int S = K.size(1);
    TORCH_CHECK(K.size(2) == D, "K/V head dim must match Q");
    TORCH_CHECK(D % 32 == 0 && D <= 1024, "D must be a multiple of 32 and <= 1024 (one thread per dim, warp reductions)");
    TORCH_CHECK(S >= 1, "S must be >= 1");

    const int chunk_size = 256;
    const int num_chunks = (S + chunk_size - 1) / chunk_size;
    auto options = Q.options();
    torch::Tensor O_partial = torch::empty({num_chunks, D}, options);
    torch::Tensor lse_partial = torch::empty({num_chunks}, options);
    torch::Tensor O = torch::empty({1, 1, D}, options);
    run_flash_decoding_partial(Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), O_partial.data_ptr<float>(), lse_partial.data_ptr<float>(), D, S, chunk_size, current_stream());
    run_flash_decoding_final(O_partial.data_ptr<float>(), lse_partial.data_ptr<float>(), O.data_ptr<float>(), D, num_chunks, current_stream());
    check_last_cuda_error("flash_decoding_v10_forward");
    return O;
}

torch::Tensor gqa_attention_forward(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache, int64_t past_len, int64_t max_chunks, bool poison_scratch) {
    check_cuda_contiguous(q, "q");
    check_cuda_contiguous(k_cache, "k_cache");
    check_cuda_contiguous(v_cache, "v_cache");
    TORCH_CHECK(q.scalar_type() == torch::kFloat32 && k_cache.scalar_type() == torch::kFloat32 && v_cache.scalar_type() == torch::kFloat32, "q, k_cache and v_cache must be float32");
    TORCH_CHECK(q.dim() == 3, "q must be 3D [num_tokens, num_heads, head_dim], got ", q.sizes());
    TORCH_CHECK(k_cache.dim() == 3 && k_cache.sizes() == v_cache.sizes(), "k_cache and v_cache must both be [num_kv_heads, max_seq_len, head_dim]");
    TORCH_CHECK(k_cache.size(2) == q.size(2), "q and the caches must have the same head_dim");
    TORCH_CHECK(past_len >= 0 && past_len <= std::numeric_limits<int>::max(), "past_len out of range");
    TORCH_CHECK(max_chunks >= 1 && max_chunks <= std::numeric_limits<int>::max(), "max_chunks out of range");

    torch::Tensor out = run_attention(q, k_cache, v_cache, static_cast<int>(past_len), static_cast<int>(max_chunks), poison_scratch);
    check_last_cuda_error("gqa_attention_forward");
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemv_int4_forward", &gemv_int4_forward, "INT4 GEMV v2 (engine kernel): W vec (+ bias)",
          pybind11::arg("q_mat"), pybind11::arg("scales"), pybind11::arg("vec"), pybind11::arg("bias") = pybind11::none());
    m.def("gemv_int4_v1_forward", &gemv_int4_v1_forward, "v1.0 INT4 GEMV (reference)");
    m.def("flash_decoding_v10_forward", &flash_decoding_v10_forward, "LEGACY v1.0 Flash-Decoding, single head (Level 4 baseline)");
    m.def("flash_decoding_forward", &flash_decoding_forward, "Single-head decode attention for batch=1 (the GQA attention kernels with 1 head)");
    m.def("gqa_attention_forward", &gqa_attention_forward, "Batched GQA attention (decode / verify)",
          pybind11::arg("q"), pybind11::arg("k_cache"), pybind11::arg("v_cache"), pybind11::arg("past_len"),
          pybind11::arg("max_chunks") = MAX_ATTN_CHUNKS, pybind11::arg("poison_scratch") = false);
}