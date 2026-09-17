#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

constexpr int NUM_ITERATIONS = 100;

void symmetric_quantization(const std::vector<float>& h_mat, std::vector<uint32_t>& h_q_mat, std::vector<float>& h_scales, int rows, int cols);

template <typename Func>
void benchmark_kernel(const std::string& kernel_name, Func kernel_call, size_t total_bytes, int num_iterations = NUM_ITERATIONS, int warmup = 10) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_ms = 0;
    for(int i = 0; i < num_iterations; i++) { 
        cudaEventRecord(start);
        kernel_call();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        if (i >= warmup) total_ms += milliseconds;
    }

    float avg_ms = total_ms / (num_iterations - warmup);
    float avg_bandwidth = (total_bytes / 1e6) / avg_ms;

    std::cout << "===" << kernel_name << "===" << "\n";
    std::cout << "Average Execution Time: " << avg_ms << " ms\n";
    std::cout << "Average Bandwidth " << avg_bandwidth << " GB/s\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}