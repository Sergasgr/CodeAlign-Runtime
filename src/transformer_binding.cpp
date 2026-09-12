#include <torch/extension.h>
#include "memory.h"
#include "transformer.h"

class QwenBlock {
    private: 
        LayerBuffers buffers;
        LayerKVCache kv_cache;
        TransformerBlockWeights weights;
    public:
        QwenBlock(int d, int intermediate_dim, int max_seq_len);
        torch::Tensor forward(torch::Tensor hidden_states);
        void load_weights(
            torch::Tensor attn_norm,
            torch::Tensor q_weight, torch::Tensor q_scales,
            torch::Tensor k_weight, torch::Tensor k_scales,
            torch::Tensor v_weight, torch::Tensor v_scales,
            torch::Tensor o_weight, torch::Tensor o_scales,
            torch::Tensor mlp_norm,
            torch::Tensor gate_weight, torch::Tensor gate_scales,
            torch::Tensor up_weight, torch::Tensor up_scales,
            torch::Tensor down_weight, torch::Tensor down_scales
        );
};

QwenBlock::QwenBlock(int d, int intermediate_dim, int max_seq_len) {
    init_kv_cache(kv_cache, max_seq_len, d);
    init_buffers(buffers, d, intermediate_dim, max_seq_len);
}

torch::Tensor QwenBlock::forward(torch::Tensor hidden_states) {
    TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be a CUDA tensor");
    TORCH_CHECK(hidden_states.is_contiguous(), "hidden_states must be aligned sequentially contiguous in memory");
    forward_transformer_block(hidden_states.data_ptr<float>(), weights, kv_cache, buffers);
    return hidden_states;
}

void QwenBlock::load_weights(
    torch::Tensor attn_norm,
    torch::Tensor q_weight, torch::Tensor q_scales,
    torch::Tensor k_weight, torch::Tensor k_scales,
    torch::Tensor v_weight, torch::Tensor v_scales,
    torch::Tensor o_weight, torch::Tensor o_scales,
    torch::Tensor mlp_norm,
    torch::Tensor gate_weight, torch::Tensor gate_scales,
    torch::Tensor up_weight, torch::Tensor up_scales,
    torch::Tensor down_weight, torch::Tensor down_scales
) {
    weights.attn_norm_weight = attn_norm.data_ptr<float>()
    weights.mlp_norm_weight = mlp_norm.data_ptr<float>();

    weights.q_proj.q_weight = (uint32_t*)q_weight.data_ptr<int32_t>();
    weights.q_proj.scales = q_scales.data_ptr<float>();
    weights.q_proj.out_features = q_weight.size(0);
    weights.q_proj.in_features = q_scales.size(1);

    weights.k_proj.q_weight = (uint32_t*)k_weight.data_ptr<int32_t>();
    weights.k_proj.scales = k_scales.data_ptr<float>();
    weights.k_proj.out_features = k_weight.size(0);
    weights.k_proj.in_features = k_scales.size(1);

    weights.v_proj.q_weight = (uint32_t*)v_weight.data_ptr<int32_t>();
    weights.v_proj.scales = v_scales.data_ptr<float>();
    weights.v_proj.out_features = v_weight.size(0);
    weights.v_proj.in_features = v_scales.size(1);

    weights.o_proj.q_weight = (uint32_t*)o_weight.data_ptr<int32_t>();
    weights.o_proj.scales = o_scales.data_ptr<float>();
    weights.o_proj.out_features = o_weight.size(0);
    weights.o_proj.in_features = o_scales.size(1);

    weights.gate_proj.q_weight = (uint32_t*)gate_weight.data_ptr<int32_t>();
    weights.gate_proj.scales = gate_scales.data_ptr<float>();
    weights.gate_proj.out_features = gate_weight.size(0);
    weights.gate_proj.in_features = gate_scales.size(1);

    weights.up_proj.q_weight = (uint32_t*)up_weight.data_ptr<int32_t>();
    weights.up_proj.scales = up_scales.data_ptr<float>();
    weights.up_proj.out_features = up_weight.size(0);
    weights.up_proj.in_features = up_scales.size(1);

    weights.down_proj.q_weight = (uint32_t*)down_weight.data_ptr<int32_t>();
    weights.down_proj.scales = down_scales.data_ptr<float>();
    weights.down_proj.out_features = down_weight.size(0);
    weights.down_proj.in_features = down_scales.size(1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<QwenBlock>(m, "QwenBlock")
        .def(pybind11::init<int, int, int>())
        .def("forward", &QwenBlock::forward)
        .def("load_weights", &QwenBlock::load_weights);
}