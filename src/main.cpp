#include "gemv.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdint>

const int GROUP = 128; // cuantización simétrica INT4 por grupos de 128

void symmetric_quantization(const std::vector<float>& h_mat, std::vector<uint32_t>& h_q_mat, std::vector<float>& h_scales, int rows, int cols) { 
    for(int i = 0; i < rows * cols; i += GROUP) {
        float max_abs = std::numeric_limits<float>::lowest();
        for(int j = i; j < i + GROUP; j++) {
            if (std::abs(h_mat[j]) > max_abs) max_abs = std::abs(h_mat[j]);
        }
        float scale = std::max(max_abs, 1e-9f) / 7.0f; // INT4 -> [-8, 7]
        h_scales.push_back(scale);
        for(int j = i; j < i + GROUP; j+=8) {
            uint32_t packed = 0;
            for(int k = 0; k < 8; k++) {
                int val = round(h_mat[j + k] / scale);
                if(val > 7) val = 7;
                if(val < -8) val = -8;
                packed = packed | ((val & 0xF) << 4 * k);
            }     
            h_q_mat.push_back(packed);
        }
    }
}

const int NUM_ITERATIONS = 100;

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

int main() { // nsight-compute??
    //const int N = 5; MULTIPLICO LOS SIZE_T POR N??
    int rows = 4864, cols = 896;

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

    cudaMalloc((void**)&d_mat, bytes_mat);
    cudaMalloc((void**)&d_vec, bytes_vec);
    cudaMalloc((void**)&d_out_naive, bytes_out);
    cudaMalloc((void**)&d_out_opt, bytes_out);
    cudaMalloc((void**)&d_q_mat, bytes_q_mat); 
    cudaMalloc((void**)&d_scales, bytes_scales);
    cudaMalloc((void**)&d_out_q, bytes_out);

    cudaMemcpy(d_mat, h_mat.data(), bytes_mat, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, h_vec.data(), bytes_vec, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_naive, h_out_naive.data(), bytes_out, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_opt, h_out_opt.data(), bytes_out, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q_mat, h_q_mat.data(), bytes_q_mat, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, h_scales.data(), bytes_scales, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_q, h_out_q.data(), bytes_out, cudaMemcpyHostToDevice);
    
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

    cudaFree(d_mat);
    cudaFree(d_vec);
    cudaFree(d_out_naive);
    cudaFree(d_out_opt);
    cudaFree(d_q_mat);
    cudaFree(d_scales);
    cudaFree(d_out_q);

    return 0;
}