#include "../src/gemm/gemm.h"
#include "benchmark_utils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>

int main() {
    int K = 16; 
    int in_features = 896;
    int out_features = 4864;

    size_t bytes_A = K * in_features * sizeof(float);
    size_t bytes_B = out_features * in_features * sizeof(float);
    size_t bytes_C = K * out_features * sizeof(float);

    std::vector<float> h_A(K * in_features, 1.0f); 
    std::vector<float> h_B(out_features * in_features, 1.0f);
    std::vector<float> h_C_naive(K * out_features, 1.0f);
    std::vector<float> h_C_opt(K * out_features, 1.0f);
    std::vector<float> h_C_q(K * out_features, 1.0f);
    std::vector<float> h_scales;
    std::vector<uint32_t> h_B_quant;

    symmetric_quantization(h_B, h_B_quant, h_scales, out_features, in_features);

    size_t bytes_B_quant = h_B_quant.size() * sizeof(uint32_t);
    size_t bytes_scales = h_scales.size() * sizeof(float);

    float *d_A, *d_B, *d_C_naive, *d_C_opt, *d_C_q, *d_scales;
    uint32_t *d_B_quant;

    cudaMalloc((void**)&d_A, bytes_A);
    cudaMalloc((void**)&d_B, bytes_B);
    cudaMalloc((void**)&d_C_naive, bytes_C);
    cudaMalloc((void**)&d_C_opt, bytes_C);
    cudaMalloc((void**)&d_C_q, bytes_C);
    cudaMalloc((void**)&d_scales, bytes_scales); 
    cudaMalloc((void**)&d_B_quant, bytes_B_quant);

    cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_naive, h_C_naive.data(), bytes_C, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_opt, h_C_opt.data(), bytes_C, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_q, h_C_q.data(), bytes_C, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, h_scales.data(), bytes_scales, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_quant, h_B_quant.data(), bytes_B_quant, cudaMemcpyHostToDevice);
    
    benchmark_kernel("NAIVE KERNEL", [&]() { 
        run_gemm_naive(d_A, d_B, d_C_naive, K, in_features, out_features);
    }, bytes_A + bytes_B + bytes_C);

    cudaMemcpy(h_C_naive.data(), d_C_naive, bytes_C, cudaMemcpyDeviceToHost);

    benchmark_kernel("OPTIMIZED KERNEL", [&]() { 
        run_gemm_optimized(d_A, d_B, d_C_opt, K, in_features, out_features); 
    }, bytes_A + bytes_B + bytes_C);
    
    cudaMemcpy(h_C_opt.data(), d_C_opt, bytes_C, cudaMemcpyDeviceToHost);

    benchmark_kernel("QUANTIZED NAIVE KERNEL", [&]() { 
        run_gemm_int4_naive(d_A, d_B_quant, d_scales, d_C_q, K, in_features, out_features);
    }, bytes_A + bytes_B_quant + bytes_scales + bytes_C);

    cudaMemcpy(h_C_q.data(), d_C_q, bytes_C, cudaMemcpyDeviceToHost);

    benchmark_kernel("QUANTIZED OPTIMIZED KERNEL", [&]() { 
        run_gemm_int4_optimized(d_A, d_B_quant, d_scales, d_C_q, K, in_features, out_features);
    }, bytes_A + bytes_B_quant + bytes_scales + bytes_C);

    cudaMemcpy(h_C_q.data(), d_C_q, bytes_C, cudaMemcpyDeviceToHost);

    for(int i = 0; i < K * out_features; i++) { 
        if (std::abs(h_C_naive[i] - h_C_opt[i]) > 1e-4 || std::abs(h_C_naive[i] - h_C_q[i]) > 0.05f)  {
            std::cout << "Validation Error in index " << i << "\n";
            std::cout << "Naive: " << h_C_naive[i] 
                      << " Opt: " << h_C_opt[i] 
                      << " Q: " << h_C_q[i] << "\n";
            break;
        }
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_naive);
    cudaFree(d_C_opt);
    cudaFree(d_C_q);
    cudaFree(d_scales);
    cudaFree(d_B_quant);

    return 0;
}