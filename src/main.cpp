#include "gemv.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdint>

const int GROUP = 128;
// cuantización simétrica INT4 por grupos de 128
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
                packed = packed | ((val & 0xF) << 4*k);
            }     
            h_q_mat.push_back(packed);
        }
    }
}

int main() { // nsight-compute??
    const int N = 5;
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

    symmetric_quantization(h_mat, h_q_mat, h_scales, rows, cols);

    float *d_mat, *d_vec, *d_out_naive, *d_out_opt, *d_q_mat, *d_scales;
    cudaMalloc((void**)&d_mat, bytes_mat);
    cudaMalloc((void**)&d_vec, bytes_vec);
    cudaMalloc((void**)&d_out_naive, bytes_out);
    cudaMalloc((void**)&d_out_opt, bytes_out);
    cudaMalloc((void**)&d_q_mat, );
    cudaMalloc((void**)&d_scales, );

    cudaMemcpy(d_mat, h_mat.data(), bytes_mat, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, h_vec.data(), bytes_vec, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_naive, h_out_naive.data(), bytes_out, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_opt, h_out_opt.data(), bytes_out, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q_mat, h_q_mat.data(), , cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, h_scales.data(), , cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_ms = 0;
    const int NUM_ITERATIONS = 100;

    for(int i = 0; i < NUM_ITERATIONS; i++) { 
        cudaEventRecord(start);
        run_gemv_naive(d_mat, d_vec, d_out_naive, rows, cols);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        if (i >= 10) total_ms += milliseconds;
    }

    cudaMemcpy(h_out_naive.data(), d_out_naive, bytes_out, cudaMemcpyDeviceToHost);

    float avg_ms = total_ms / (NUM_ITERATIONS - 10);
    float avg_bandwidth = ((bytes_mat + bytes_vec + bytes_out) / 1e6) / avg_ms;

    std::cout << "-----NAIVE KERNEL-----" << "\n";
    std::cout << "Average Execution Time: " << avg_ms << " ms\n";
    std::cout << "Average Bandwidth " << avg_bandwidth << " GB/s\n";
    
    total_ms = 0;

    for(int i = 0; i < NUM_ITERATIONS; i++) { 
        cudaEventRecord(start);
        run_gemv_optimized(d_mat, d_vec, d_out_opt, rows, cols);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        if (i >= 10) total_ms += milliseconds;
    }

    cudaMemcpy(h_out_opt.data(), d_out_opt, bytes_out, cudaMemcpyDeviceToHost);

    avg_ms = total_ms / (NUM_ITERATIONS - 10);
    avg_bandwidth = ((bytes_mat + bytes_vec + bytes_out) / 1e6) / avg_ms;

    std::cout << "-----OPTIMIZED KERNEL-----" << "\n"; // WARP KERNEL??
    std::cout << "Average Execution Time: " << avg_ms << " ms\n";
    std::cout << "Average Bandwidth " << avg_bandwidth << " GB/s\n";

    total_ms = 0;

    for(int i = 0; i < NUM_ITERATIONS; i++) { 
        cudaEventRecord(start);
        run_gemv_int4_kernel(d_mat, d_vec, d_out_opt, rows, cols);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        if (i >= 10) total_ms += milliseconds;
    }

    //cudaMemcpy(h_out_opt.data(), d_out_opt, bytes_out, cudaMemcpyDeviceToHost);

    avg_ms = total_ms / (NUM_ITERATIONS - 10);
    //avg_bandwidth = ((bytes_mat + bytes_vec + bytes_out) / 1e6) / avg_ms;

    std::cout << "-----QUANTIZED KERNEL-----" << "\n";
    std::cout << "Average Execution Time: " << avg_ms << " ms\n";
    std::cout << "Average Bandwidth " << avg_bandwidth << " GB/s\n";
    
    for(int i = 0; i < rows; i++) { //quantized?
        if (std::abs(h_out_naive[i] - h_out_opt[i]) > 1e-4) {
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

    return 0;
}