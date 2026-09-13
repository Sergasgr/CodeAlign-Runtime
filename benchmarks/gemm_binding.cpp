#include <torch/extension.h>
#include "../src/gemm/gemm.h"

torch::Tensor gemm_int4_forward(torch::Tensor A, torch::Tensor q_mat_B, torch::Tensor scales_B) {
    TORCH_CHECK(A.is_cuda() && A.is_contiguous(), "A must be a CUDA tensor and contiguous");
    TORCH_CHECK(q_mat_B.is_cuda() && q_mat_B.is_contiguous(), "q_mat_B must be a CUDA tensor and contiguous");
    TORCH_CHECK(scales_B.is_cuda() && scales_B.is_contiguous(), "scales_B must be a CUDA tensor and contiguous");

    int K = A.size(0);
    int in_features = A.size(1);
    int out_features = q_mat_B.size(0); 

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(A.device());
    torch::Tensor C = torch::empty({K, out_features}, options);

    run_gemm_int4_optimized(
        A.data_ptr<float>(),
        (uint32_t*)q_mat_B.data_ptr<int32_t>(),
        scales_B.data_ptr<float>(),
        C.data_ptr<float>(),
        K, 
        in_features, 
        out_features
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_int4_forward", &gemm_int4_forward, "GEMM INT4 Optimized Kernel");
}
