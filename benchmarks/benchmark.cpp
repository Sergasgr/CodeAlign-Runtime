#include "utils.h"

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
                packed = packed | ((uint32_t)(val & 0xF) << (4 * k));
            }     
            h_q_mat.push_back(packed);
        }
    }
}

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

void gemv_malloc(float *d_mat, float *d_vec, float *d_out_naive, float *d_out_opt, float *d_scales, float *d_out_q, uint32_t *d_q_mat) {
    cudaMalloc((void**)&d_mat, bytes_mat);
    cudaMalloc((void**)&d_vec, bytes_vec);
    cudaMalloc((void**)&d_out_naive, bytes_out);
    cudaMalloc((void**)&d_out_opt, bytes_out);
    cudaMalloc((void**)&d_q_mat, bytes_q_mat); 
    cudaMalloc((void**)&d_scales, bytes_scales);
    cudaMalloc((void**)&d_out_q, bytes_out);
}

void gemv_memcpy(float *d_mat, float *d_vec, float *d_out_naive, float *d_out_opt, float *d_scales, float *d_out_q, uint32_t *d_q_mat) {
    cudaMemcpy(d_mat, h_mat.data(), bytes_mat, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, h_vec.data(), bytes_vec, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_naive, h_out_naive.data(), bytes_out, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_opt, h_out_opt.data(), bytes_out, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q_mat, h_q_mat.data(), bytes_q_mat, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, h_scales.data(), bytes_scales, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_q, h_out_q.data(), bytes_out, cudaMemcpyHostToDevice);
}

void gemv_free(float *d_mat, float *d_vec, float *d_out_naive, float *d_out_opt, float *d_scales, float *d_out_q, uint32_t *d_q_mat) {
    cudaFree(d_mat);
    cudaFree(d_vec);
    cudaFree(d_out_naive);
    cudaFree(d_out_opt);
    cudaFree(d_q_mat);
    cudaFree(d_scales);
    cudaFree(d_out_q);
}