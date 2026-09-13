#pragma once
#include <cstdint> //cuando poner esto??

void symmetric_quantization(const std::vector<float>& h_mat, std::vector<uint32_t>& h_q_mat, std::vector<float>& h_scales, int rows, int cols);
template <typename Func> void benchmark_kernel(const std::string& kernel_name, Func kernel_call, size_t total_bytes, int num_iterations = NUM_ITERATIONS, int warmup = 10);
void gemv_malloc(float *d_mat, float *d_vec, float *d_out_naive, float *d_out_opt, float *d_scales, float *d_out_q, uint32_t *d_q_mat);
void gemv_memcpy(float *d_mat, float *d_vec, float *d_out_naive, float *d_out_opt, float *d_scales, float *d_out_q, uint32_t *d_q_mat);
void gemv_free(float *d_mat, float *d_vec, float *d_out_naive, float *d_out_opt, float *d_scales, float *d_out_q, uint32_t *d_q_mat);