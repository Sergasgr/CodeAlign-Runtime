#include "gemv.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

int main() { // nsight-compute??
    const int N = 5;
    int rows = 4864, cols = 896;
    size_t bytes_mat = rows * cols * sizeof(float);
    size_t bytes_vec = cols * sizeof(float);
    size_t bytes_out = rows * sizeof(float);
    
    std::vector<float> h_mat(rows * cols, 1.0f); 
    std::vector<float> h_vec(cols, 1.0f);
    std::vector<float> h_out(rows, 1.0f);

    float *d_mat, *d_vec, *d_out;
    cudaMalloc((void**)&d_mat, bytes_mat);
    cudaMalloc((void**)&d_vec, bytes_vec);
    cudaMalloc((void**)&d_out, bytes_out);

    cudaMemcpy(d_mat, h_mat.data(), bytes_mat, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, h_vec.data(), bytes_vec, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, h_out.data(), bytes_out, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_ms = 0;
    const int NUM_ITERATIONS = 100;

    for(int i = 0; i < NUM_ITERATIONS; i++) { 
        cudaEventRecord(start);
        run_gemv_naive(d_mat, d_vec, d_out, rows, cols);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        if (i >= 10) total_ms += milliseconds;
    }

    cudaMemcpy(h_out.data(), d_out, bytes_out, cudaMemcpyDeviceToHost);
    for(int i = 0; i < 10; i++) {
        std::cout << h_out[i] << " ";
    }
    
    float avg_ms = total_ms / (NUM_ITERATIONS - 10);
    float avg_bandwidth = ((bytes_mat + bytes_vec + bytes_out) / 1e6) / avg_ms;

    std::cout << "\nAverage Execution time: " << avg_ms << " ms\n";
    std::cout << "Average Bandwidth " << avg_bandwidth << " GB/s\n";

    cudaFree(d_mat);
    cudaFree(d_vec);
    cudaFree(d_out);

    return 0;
}