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

torch::Tensor flash_decoding_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    TORCH_CHECK(Q.is_cuda() && Q.is_contiguous(), "Query tensor must be a CUDA tensor and must be aligned sequentially contiguous in memory")
    TORCH_CHECK(K.is_cuda() && K.is_contiguous(), "Key tensor must be a CUDA tensor and must be aligned sequentially contiguous in memory");
    TORCH_CHECK(V.is_cuda() && V.is_contiguous(), "Value tensor must be a CUDA tensor and must be aligned sequentially contiguous in memory");

    int D = Q.size(2);
    int S = K.size(1);

    int chunk_size = 256;
    int num_chunks = (S + chunk_size - 1) / chunk_size;

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());

    torch::Tensor O_partial = torch::empty({num_chunks, D}, options);
    torch::Tensor lse_partial = torch::empty({num_chunks}, options);
    torch::Tensor O = torch::empty({1, 1, D}, options);

    run_flash_decoding_partial(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O_partial.data_ptr<float>(),
        lse_partial.data_ptr<float>(),
        D, S, chunk_size
    );

    run_flash_decoding_final(
        O_partial.data_ptr<float>(),
        lse_partial.data_ptr<float>(),
        O.data_ptr<float>(),
        D, num_chunks
    );

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemv_int4_forward", &gemv_int4_forward, "GEMV INT4 Optimized Kernel");
    m.def("flash_decoding_forward", &flash_decoding_forward, "Flash-Decoding forward pass for batch=1");
}