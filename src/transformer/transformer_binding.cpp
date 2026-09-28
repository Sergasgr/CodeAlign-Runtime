#include <torch/extension.h>
#include <optional>
#include <string>
#include <vector>
#include "memory.h"
#include "cuda_check.h"
#include "../speculative/speculative.h"
#include "../ops/logits_ops.h"
#include "transformer.h"

namespace {
    void check_cuda_contiguous(const torch::Tensor& t, const char* name) {
        TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
        TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    }

    void check_last_cuda_error(const char* where) {
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, where, ": CUDA error: ", cudaGetErrorString(err));
    }

    class QwenBlock {
        private:
            ModelDims dims;
            LayerBuffers buffers{};
            LayerKVCache kv_cache{};
            TransformerBlockWeights weights{};
            std::vector<torch::Tensor> keep_alive;
            bool weights_loaded = false;

            QuantizedLinear make_linear(const char* name, const torch::Tensor& q_weight, const torch::Tensor& scales, const std::optional<torch::Tensor>& bias, int expected_in, int expected_out);
            float* make_vector(const char* name, const torch::Tensor& t, int expected_size);
        public:
            QwenBlock(int hidden, int intermediate, int num_heads, int num_kv_heads, int head_dim, int max_seq_len);
            ~QwenBlock();
            QwenBlock(const QwenBlock&) = delete;             
            QwenBlock& operator=(const QwenBlock&) = delete;

            torch::Tensor forward(torch::Tensor hidden_states);
            void load_weights(
                torch::Tensor attn_norm,
                torch::Tensor q_weight, torch::Tensor q_scales, std::optional<torch::Tensor> q_bias,
                torch::Tensor k_weight, torch::Tensor k_scales, std::optional<torch::Tensor> k_bias,
                torch::Tensor v_weight, torch::Tensor v_scales, std::optional<torch::Tensor> v_bias,
                torch::Tensor o_weight, torch::Tensor o_scales,
                torch::Tensor mlp_norm,
                torch::Tensor gate_weight, torch::Tensor gate_scales,
                torch::Tensor up_weight, torch::Tensor up_scales,
                torch::Tensor down_weight, torch::Tensor down_scales
            );
            void rollback_kv_cache(int rejected_tokens);
            void reset_kv_cache() { kv_cache.current_seq_len = 0; }
            void set_seq_len(int seq_len);
            int seq_len() const { return kv_cache.current_seq_len; }
            int max_seq_len() const { return kv_cache.max_seq_len; }
    };
} 

QwenBlock::QwenBlock(int hidden, int intermediate, int num_heads, int num_kv_heads, int head_dim, int max_seq_len) {
    TORCH_CHECK(hidden > 0 && hidden <= 1024, "hidden must be in (0, 1024]: RMSNorm runs as a single 1024-thread block");
    TORCH_CHECK(hidden % 128 == 0 && intermediate % 128 == 0, "hidden and intermediate must be multiples of the INT4 group size (128)");
    TORCH_CHECK(num_heads > 0 && num_kv_heads > 0 && num_heads % num_kv_heads == 0, "num_heads must be a multiple of num_kv_heads");
    TORCH_CHECK(head_dim % 32 == 0 && head_dim <= 1024, "head_dim must be a multiple of 32 (warp size) and <= 1024 (one thread per dim)");
    TORCH_CHECK((num_heads * head_dim) % 128 == 0, "num_heads * head_dim must be a multiple of 128 (o_proj input)");
    TORCH_CHECK(max_seq_len > 0, "max_seq_len must be positive");
    dims = ModelDims{hidden, intermediate, num_heads, num_kv_heads, head_dim};
    init_kv_cache(kv_cache, max_seq_len, dims);
    init_buffers(buffers, dims, max_seq_len);
}

QwenBlock::~QwenBlock() {
    free_buffers(buffers);
    free_kv_cache(kv_cache);
}

QuantizedLinear QwenBlock::make_linear(const char* name, const torch::Tensor& q_weight, const torch::Tensor& scales,
                                       const std::optional<torch::Tensor>& bias, int expected_in, int expected_out) {
    check_cuda_contiguous(q_weight, name);
    check_cuda_contiguous(scales, name);
    TORCH_CHECK(q_weight.scalar_type() == torch::kInt32, name, ": packed weight must be int32");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat32, name, ": scales must be float32");
    TORCH_CHECK(q_weight.dim() == 2, name, ": packed weight must be 2D [out, in/8]");

    const int out_features = q_weight.size(0);
    const int in_features = q_weight.size(1) * 8;  // 8 INT4 values per int32 word
    TORCH_CHECK(in_features == expected_in && out_features == expected_out, name, ": expected [", expected_out, ", ",
                expected_in, "] but packed weight implies [", out_features, ", ", in_features, "]");
    TORCH_CHECK(scales.numel() == static_cast<int64_t>(out_features) * (in_features / 128), name,
                ": scales must have out * in / 128 elements");

    QuantizedLinear proj{};
    proj.q_weight = reinterpret_cast<uint32_t*>(q_weight.data_ptr<int32_t>());
    proj.scales = scales.data_ptr<float>();
    proj.bias = nullptr;
    proj.in_features = in_features;
    proj.out_features = out_features;
    keep_alive.push_back(q_weight);
    keep_alive.push_back(scales);

    if (bias.has_value()) {
        proj.bias = make_vector(name, *bias, out_features);
    }
    return proj;
}

float* QwenBlock::make_vector(const char* name, const torch::Tensor& t, int expected_size) {
    check_cuda_contiguous(t, name);
    TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, ": must be float32");
    TORCH_CHECK(t.numel() == expected_size, name, ": expected ", expected_size, " elements, got ", t.numel());
    keep_alive.push_back(t);
    return t.data_ptr<float>();
}

void QwenBlock::load_weights(
    torch::Tensor attn_norm,
    torch::Tensor q_weight, torch::Tensor q_scales, std::optional<torch::Tensor> q_bias,
    torch::Tensor k_weight, torch::Tensor k_scales, std::optional<torch::Tensor> k_bias,
    torch::Tensor v_weight, torch::Tensor v_scales, std::optional<torch::Tensor> v_bias,
    torch::Tensor o_weight, torch::Tensor o_scales,
    torch::Tensor mlp_norm,
    torch::Tensor gate_weight, torch::Tensor gate_scales,
    torch::Tensor up_weight, torch::Tensor up_scales,
    torch::Tensor down_weight, torch::Tensor down_scales
) {
    keep_alive.clear();
    const int D = dims.hidden, I = dims.intermediate, Q = dims.q_dim(), KV = dims.kv_dim();

    weights.attn_norm_weight = make_vector("attn_norm", attn_norm, D);
    weights.mlp_norm_weight = make_vector("mlp_norm", mlp_norm, D);

    weights.q_proj = make_linear("q_proj", q_weight, q_scales, q_bias, D, Q);
    weights.k_proj = make_linear("k_proj", k_weight, k_scales, k_bias, D, KV);
    weights.v_proj = make_linear("v_proj", v_weight, v_scales, v_bias, D, KV);
    weights.o_proj = make_linear("o_proj", o_weight, o_scales, std::nullopt, Q, D);
    weights.gate_proj = make_linear("gate_proj", gate_weight, gate_scales, std::nullopt, D, I);
    weights.up_proj = make_linear("up_proj", up_weight, up_scales, std::nullopt, D, I);
    weights.down_proj = make_linear("down_proj", down_weight, down_scales, std::nullopt, I, D);
    weights_loaded = true;
}

torch::Tensor QwenBlock::forward(torch::Tensor hidden_states) {
    TORCH_CHECK(weights_loaded, "load_weights() must be called before forward()");
    check_cuda_contiguous(hidden_states, "hidden_states");
    TORCH_CHECK(hidden_states.scalar_type() == torch::kFloat32, "hidden_states must be float32");
    TORCH_CHECK(hidden_states.dim() == 2 && hidden_states.size(1) == dims.hidden,
                "hidden_states must be 2D [num_tokens, ", dims.hidden, "], got ", hidden_states.sizes());
    const int num_tokens = hidden_states.size(0);
    TORCH_CHECK(num_tokens >= 1 && num_tokens <= MAX_TOKENS_PER_FORWARD,
                "num_tokens must be in [1, ", MAX_TOKENS_PER_FORWARD, "], got ", num_tokens, " (chunk longer inputs)");
    TORCH_CHECK(kv_cache.current_seq_len + num_tokens <= kv_cache.max_seq_len,
                "KV-cache overflow: ", kv_cache.current_seq_len, " + ", num_tokens, " > max_seq_len ", kv_cache.max_seq_len);

    forward_transformer_block(hidden_states.data_ptr<float>(), weights, kv_cache, buffers, dims, num_tokens);
    check_last_cuda_error("QwenBlock.forward");
    return hidden_states;
}

void QwenBlock::rollback_kv_cache(int rejected_tokens) {
    TORCH_CHECK(rejected_tokens >= 0 && rejected_tokens <= kv_cache.current_seq_len,
                "cannot roll back ", rejected_tokens, " tokens from a cache of ", kv_cache.current_seq_len);
    kv_cache.current_seq_len -= rejected_tokens;
}

void QwenBlock::set_seq_len(int seq_len) {
    TORCH_CHECK(seq_len >= 0 && seq_len <= kv_cache.max_seq_len, "seq_len out of range [0, ", kv_cache.max_seq_len, "]");
    kv_cache.current_seq_len = seq_len;
}

torch::Tensor fast_argmax(torch::Tensor logits) {
    TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
    TORCH_CHECK(logits.is_contiguous(), "logits must be aligned sequentially contiguous in memory");
    TORCH_CHECK(logits.scalar_type() == torch::kFloat32, "logits must be float32");
    TORCH_CHECK(logits.dim() == 2, "logits must be 2D [num_tokens, vocab_size]");

    int num_tokens = logits.size(0);
    int vocab_size = logits.size(1);
    TORCH_CHECK(num_tokens >= 1 && vocab_size >= 1, "logits must be non-empty");

    auto options = torch::TensorOptions().dtype(torch::kInt32).device(logits.device());
    torch::Tensor predicted_tokens = torch::empty({num_tokens}, options);

    run_compute_argmax_kernel(logits.data_ptr<float>(), predicted_tokens.data_ptr<int32_t>(), num_tokens, vocab_size);
    check_last_cuda_error("fast_argmax");

    return predicted_tokens;
}

std::vector<int> find_candidate_draft_checked(const std::vector<int>& tokens, int n, int k) {
    TORCH_CHECK(n >= 1, "n-gram size must be >= 1");
    TORCH_CHECK(k >= 0, "max draft length must be >= 0");
    return find_candidate_draft(tokens, n, k);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<QwenBlock>(m, "QwenBlock")
        .def(pybind11::init<int, int, int, int, int, int>(),
             pybind11::arg("hidden"), pybind11::arg("intermediate"), pybind11::arg("num_heads"),
             pybind11::arg("num_kv_heads"), pybind11::arg("head_dim"), pybind11::arg("max_seq_len"))
        .def("forward", &QwenBlock::forward, "In-place decode/verify step over hidden_states [num_tokens, hidden]")
        .def("load_weights", &QwenBlock::load_weights,
             pybind11::arg("attn_norm"),
             pybind11::arg("q_weight"), pybind11::arg("q_scales"), pybind11::arg("q_bias"),
             pybind11::arg("k_weight"), pybind11::arg("k_scales"), pybind11::arg("k_bias"),
             pybind11::arg("v_weight"), pybind11::arg("v_scales"), pybind11::arg("v_bias"),
             pybind11::arg("o_weight"), pybind11::arg("o_scales"),
             pybind11::arg("mlp_norm"),
             pybind11::arg("gate_weight"), pybind11::arg("gate_scales"),
             pybind11::arg("up_weight"), pybind11::arg("up_scales"),
             pybind11::arg("down_weight"), pybind11::arg("down_scales"))
        .def("rollback_kv_cache", &QwenBlock::rollback_kv_cache)
        .def("reset_kv_cache", &QwenBlock::reset_kv_cache)
        .def("set_seq_len", &QwenBlock::set_seq_len)
        .def("seq_len", &QwenBlock::seq_len)
        .def("max_seq_len", &QwenBlock::max_seq_len);

    m.attr("MAX_TOKENS_PER_FORWARD") = MAX_TOKENS_PER_FORWARD;
    m.def("find_candidate_draft", &find_candidate_draft_checked, "Oracle for speculative decoding",
          pybind11::arg("tokens"), pybind11::arg("n"), pybind11::arg("k"));
    m.def("fast_argmax", &fast_argmax, "Fast GPU argmax for speculative decoding");
}