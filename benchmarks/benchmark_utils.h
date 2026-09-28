#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                                      \
    do {                                                                                      \
        cudaError_t err_ = (call);                                                            \
        if (err_ != cudaSuccess) {                                                            \
            std::fprintf(stderr, "CUDA error '%s' at %s:%d: %s\n", #call, __FILE__, __LINE__, \
                         cudaGetErrorString(err_));                                           \
            std::exit(EXIT_FAILURE);                                                          \
        }                                                                                     \
    } while (0)

#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

constexpr int NUM_ITERATIONS = 100;
constexpr int WARMUP_ITERATIONS = 10;
constexpr int GROUP_SIZE = 128;           // INT4 per-group quantization
constexpr double SPEC_BANDWIDTH_GBPS = 896.0;  // RTX 5070 Ti GDDR7 spec sheet (override: env GPU_BW_GBPS)

double spec_bandwidth_gbps();

class L2Flusher {
    public:
        L2Flusher();
        ~L2Flusher();
        L2Flusher(const L2Flusher&) = delete;
        L2Flusher& operator=(const L2Flusher&) = delete;
        void flush();
        int l2_bytes() const { return l2_bytes_; }
    private:
        void* buffer_ = nullptr;
        size_t bytes_ = 0;
        int l2_bytes_ = 0;
};

struct BenchStats {
    std::string kernel;
    std::string cache;
    double p50_ms = 0, p90_ms = 0, p99_ms = 0, mean_ms = 0;
    double bytes = 0;
    double gbps_p50 = 0;
    double pct_of_spec_bw = 0;
};

struct CheckResult {
    std::string kernel;
    std::string reference;
    double max_abs_err = 0;
    double max_abs_ref = 0;
    double atol = 0, rtol = 0;
    bool pass = false;
};

double percentile(std::vector<float> samples, double p);
void print_stats(const BenchStats& s);

template <typename Func>
BenchStats benchmark_kernel(const std::string& kernel_name, Func kernel_call, size_t total_bytes,
                            L2Flusher* flusher = nullptr, int num_iterations = NUM_ITERATIONS,
                            int warmup = WARMUP_ITERATIONS) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> samples;
    samples.reserve(num_iterations);
    for (int i = 0; i < num_iterations + warmup; i++) {
        if (flusher) flusher->flush();
        CUDA_CHECK(cudaEventRecord(start));
        kernel_call();
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        if (i >= warmup) samples.push_back(ms);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    BenchStats s;
    s.kernel = kernel_name;
    s.cache = flusher ? "cold" : "hot";
    s.p50_ms = percentile(samples, 50);
    s.p90_ms = percentile(samples, 90);
    s.p99_ms = percentile(samples, 99);
    double sum = 0;
    for (float v : samples) sum += v;
    s.mean_ms = sum / samples.size();
    s.bytes = static_cast<double>(total_bytes);
    s.gbps_p50 = (s.bytes / 1e6) / s.p50_ms;
    s.pct_of_spec_bw = 100.0 * s.gbps_p50 / spec_bandwidth_gbps();
    print_stats(s);
    return s;
}

std::vector<float> random_uniform(size_t n, float lo, float hi, uint32_t seed);

void symmetric_quantization(const std::vector<float>& h_mat, std::vector<uint32_t>& h_q_mat, std::vector<float>& h_scales, int rows, int cols);
std::vector<float> dequantize_int4(const std::vector<uint32_t>& h_q_mat, const std::vector<float>& h_scales, int rows, int cols);

std::vector<float> cpu_gemv(const std::vector<float>& mat, const std::vector<float>& vec, int rows, int cols);
std::vector<float> cpu_gemm(const std::vector<float>& A, const std::vector<float>& B, int M, int in_features, int out_features);

CheckResult compare(const std::string& kernel, const std::string& reference, const std::vector<float>& ref, const std::vector<float>& out, double atol, double rtol);
void print_check(const CheckResult& c);

void fill_nan(float* d_ptr, size_t count);

std::string device_json();
void write_results_json(const std::string& path, const std::string& benchmark, const std::string& shape_json, const std::vector<CheckResult>& checks, const std::vector<BenchStats>& stats);