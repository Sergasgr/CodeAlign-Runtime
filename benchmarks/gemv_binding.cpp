#include <torch/extension.h>
#include <cuda_runtime.h>
#include "../src/gemv/gemv.h"
#include "../src/ops/flash_decoding.h"

namespace {
    void check_cuda_contiguous(const torch::Tensor& t, const char* name) {
        TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
        TORCH_CHECK(t.is_contiguous(), name, " must be aligned sequentially contiguous in memory");
    }

    void check_last_cuda_error(const char* where) {
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, where, ": CUDA error: ", cudaGetErrorString(err));
    }
}  

// q_mat: int32 [rows, cols/8] (8 packed INT4 per word), scales: fp32 [rows, cols/128], vec: fp32 [cols] -> fp32 [rows]
torch::Tensor gemv_int4_forward(torch::Tensor q_mat, torch::Tensor scales, torch::Tensor vec) {
    check_cuda_contiguous(q_mat, "q_mat");
    check_cuda_contiguous(scales, "scales");
    check_cuda_contiguous(vec, "vec");
    TORCH_CHECK(q_mat.scalar_type() == torch::kInt32, "q_mat must be int32 (packed INT4)");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat32, "scales must be float32");
    TORCH_CHECK(vec.scalar_type() == torch::kFloat32, "vec must be float32");
    TORCH_CHECK(q_mat.dim() == 2, "q_mat must be 2D [rows, cols/8]");
    TORCH_CHECK(vec.dim() == 1, "vec must be 1D [cols], got ", vec.sizes());

    int rows = q_mat.size(0);
    int cols = q_mat.size(1) * 8;
    TORCH_CHECK(vec.size(0) == cols, "vec has ", vec.size(0), " elements but q_mat implies cols = ", cols);
    TORCH_CHECK(cols % 128 == 0, "cols must be a multiple of the group size (128)");
    TORCH_CHECK(scales.numel() == static_cast<int64_t>(rows) * (cols / 128), "scales must have rows * cols / 128 elements");

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(q_mat.device());
    torch::Tensor out = torch::empty({rows}, options);

    run_gemv_int4_optimized_kernel(
        reinterpret_cast<const uint32_t*>(q_mat.data_ptr<int32_t>()),
        scales.data_ptr<float>(),
        vec.data_ptr<float>(),
        out.data_ptr<float>(),
        rows,
        cols
    );
    check_last_cuda_error("gemv_int4_forward");

    return out;
}

// Q: [1, 1, D], K/V: [1, S, D] fp32 -> [1, 1, D]
torch::Tensor flash_decoding_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    check_cuda_contiguous(Q, "Query tensor");
    check_cuda_contiguous(K, "Key tensor");
    check_cuda_contiguous(V, "Value tensor");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32 && K.scalar_type() == torch::kFloat32 && V.scalar_type() == torch::kFloat32,
                "Q, K and V must be float32");
    TORCH_CHECK(Q.dim() == 3 && K.dim() == 3 && V.dim() == 3, "Q, K and V must be 3D ([1,1,D] and [1,S,D])");
    TORCH_CHECK(Q.size(0) == 1 && Q.size(1) == 1, "batch=1, single query: Q must be [1, 1, D]");
    TORCH_CHECK(K.sizes() == V.sizes() && K.size(0) == 1, "K and V must both be [1, S, D]");

    int D = Q.size(2);
    int S = K.size(1);
    TORCH_CHECK(K.size(2) == D, "K/V head dim must match Q");
    TORCH_CHECK(D % 32 == 0 && D <= 1024, "D must be a multiple of 32 and <= 1024 (one thread per dim, warp reductions)");
    TORCH_CHECK(S >= 1, "S must be >= 1");

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
    check_last_cuda_error("flash_decoding_forward");

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemv_int4_forward", &gemv_int4_forward, "GEMV INT4 Optimized Kernel");
    m.def("flash_decoding_forward", &flash_decoding_forward, "Flash-Decoding forward pass for batch=1");
}