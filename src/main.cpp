#include "gemv.h"
#include <iostream>
#include <vector>

int main() { // thrust::host_vector y thrust::device_vector??
    const int N = 5;
    int rows = 1024, cols = 1024;
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
    cudaMemcpy(d_vec, h_vec, bytes_vec, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, h_out, bytes_out, cudaMemcpyHostToDevice);

    run_gemv_naive(d_mat, d_vec, d_out, rows, cols);

    cudaMemcpy(d_out, h_out.data(), bytes_mat, cudaMemcpyDeviceToHost);

    for(int i = 0; i < 10; i++) {
        std::cout << h_out[i] << " ";
    }

    cudaFree(d_mat);
    cudaFree(d_vec);
    cudaFree(d_out);

    return 0;
}