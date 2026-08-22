#include <torch/extension.h>
#include "gemv.h"

torch::Tensor gemv_int4_forward(torch::Tensor q_mat, torch::Tensor scales, torch::Tensor vec) {
    TORCH_CHECK(q_mat.is_cuda() && q_mat.is_contiguous(), "q_mat must be a CUDA tensor and must be aligned sequentially contiguous in memory");
    TORCH_CHECK(scales.is_cuda() && scales.is_contiguous(), "scales must be a CUDA tensor and must be aligned sequentially contiguous in memory");
    TORCH_CHECK(vec.is_cuda() && vec.is_contiguous(), "vec must be a CUDA tensor and must be aligned sequentially contiguous in memory");

    int rows = q_mat.size(0); 
    int cols = vec.size(0);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(q_mat.device());
    torch::Tensor out = torch::empty({rows}, options);

    run_gemv_int4_optimized_kernel(
        (uint32_t*)q_mat.data_ptr<int32_t>(), 
        scales.data_ptr<float>(), 
        vec.data_ptr<float>(), 
        out.data_ptr<float>(), 
        rows, 
        cols
    );

    return out;
} 

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemv_int4_forward", &gemv_int4_forward, "GEMV INT4 Optimized Kernel");
}