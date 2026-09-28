#include <torch/extension.h>
#include <cuda_runtime.h>
#include "../src/gemm/gemm.h"

namespace {
    void check_cuda_contiguous(const torch::Tensor& t, const char* name) {
        TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
        TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    }
} 

torch::Tensor gemm_int4_forward(torch::Tensor A, torch::Tensor q_mat_B, torch::Tensor scales_B) {
    check_cuda_contiguous(A, "A");
    check_cuda_contiguous(q_mat_B, "q_mat_B");
    check_cuda_contiguous(scales_B, "scales_B");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(q_mat_B.scalar_type() == torch::kInt32, "q_mat_B must be int32 (packed INT4)");
    TORCH_CHECK(scales_B.scalar_type() == torch::kFloat32, "scales_B must be float32");
    TORCH_CHECK(A.dim() == 2 && q_mat_B.dim() == 2, "A must be [K, in] and q_mat_B [out, in/8]");

    int K = A.size(0);
    int in_features = A.size(1);
    int out_features = q_mat_B.size(0);
    TORCH_CHECK(K >= 1, "A must have at least one row");
    TORCH_CHECK(q_mat_B.size(1) * 8 == in_features, "A has ", in_features, " columns but q_mat_B implies ", q_mat_B.size(1) * 8);
    TORCH_CHECK(in_features % 128 == 0, "in_features must be a multiple of the group size (128)");
    TORCH_CHECK(scales_B.numel() == static_cast<int64_t>(out_features) * (in_features / 128), "scales_B must have out * in / 128 elements");

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(A.device());
    torch::Tensor C = torch::empty({K, out_features}, options);

    run_gemm_int4_optimized(
        A.data_ptr<float>(),
        reinterpret_cast<const uint32_t*>(q_mat_B.data_ptr<int32_t>()),
        scales_B.data_ptr<float>(),
        C.data_ptr<float>(),
        K,
        in_features,
        out_features
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "gemm_int4_forward: CUDA error: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_int4_forward", &gemm_int4_forward, "GEMM INT4 Optimized Kernel");
}