#include "../src/gemv/gemv.h"
#inlcude "benchmark.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdint>
#include <limits>
#include <cmath>
 
constexpr int GROUP = 128; // Symmetric INT4 group-wise quantization (group size 128)
constexpr int NUM_ITERATIONS = 100;

int main() {
    int rows = 4864, cols = 896; // MLP Up projection dimensions for Qwen2.5-0.5B

    size_t bytes_mat = rows * cols * sizeof(float); 
    size_t bytes_vec = cols * sizeof(float);
    size_t bytes_out = rows * sizeof(float);

    std::vector<float> h_mat(rows * cols, 1.0f); 
    std::vector<float> h_vec(cols, 1.0f);
    std::vector<float> h_out_naive(rows, 1.0f);
    std::vector<float> h_out_opt(rows, 1.0f);
    std::vector<uint32_t> h_q_mat;
    std::vector<float> h_scales;
    std::vector<float> h_out_q(rows, 1.0f);

    symmetric_quantization(h_mat, h_q_mat, h_scales, rows, cols);

    size_t bytes_q_mat = h_q_mat.size() * sizeof(uint32_t);
    size_t bytes_scales = h_scales.size() * sizeof(float);

    float *d_mat, *d_vec, *d_out_naive, *d_out_opt, *d_scales, *d_out_q;
    uint32_t *d_q_mat;

    gemv_malloc(d_mat, d_vec, d_out_naive, d_out_opt, d_scales, d_out_q, d_q_mat);
    gemv_memcpy(d_mat, d_vec, d_out_naive, d_out_opt, d_scales, d_out_q, d_q_mat);
    
    benchmark_kernel("NAIVE KERNEL", [&]() { 
        run_gemv_naive(d_mat, d_vec, d_out_naive, rows, cols); 
    }, bytes_mat + bytes_vec + bytes_out);

    cudaMemcpy(h_out_naive.data(), d_out_naive, bytes_out, cudaMemcpyDeviceToHost);

    benchmark_kernel("OPTIMIZED KERNEL", [&]() { 
        run_gemv_optimized(d_mat, d_vec, d_out_opt, rows, cols); 
    }, bytes_mat + bytes_vec + bytes_out);
    
    cudaMemcpy(h_out_opt.data(), d_out_opt, bytes_out, cudaMemcpyDeviceToHost);

    benchmark_kernel("QUANTIZED NAIVE KERNEL", [&]() { 
        run_gemv_int4_naive_kernel(d_q_mat, d_scales, d_vec, d_out_q, rows, cols); 
    }, bytes_q_mat + bytes_scales + bytes_vec + bytes_out);

    cudaMemcpy(h_out_q.data(), d_out_q, bytes_out, cudaMemcpyDeviceToHost);

    benchmark_kernel("QUANTIZED OPTIMIZED KERNEL", [&]() { 
        run_gemv_int4_optimized_kernel(d_q_mat, d_scales, d_vec, d_out_q, rows, cols); 
    }, bytes_q_mat + bytes_scales + bytes_vec + bytes_out);

    cudaMemcpy(h_out_q.data(), d_out_q, bytes_out, cudaMemcpyDeviceToHost);

    for(int i = 0; i < rows; i++) { 
        if (std::abs(h_out_naive[i] - h_out_opt[i]) > 1e-4 || std::abs(h_out_naive[i] - h_out_q[i]) > 0.05f)  {
            std::cout << "Validation Error in index " << i << "\n";
            break;
        }
    }

    gemv_free(d_mat, d_vec, d_out_naive, d_out_opt, d_scales, d_out_q, d_q_mat);

    return 0;
}