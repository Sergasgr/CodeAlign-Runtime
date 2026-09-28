#include "benchmark_utils.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>

double spec_bandwidth_gbps() {
    const char* env = std::getenv("GPU_BW_GBPS");
    return env ? std::atof(env) : SPEC_BANDWIDTH_GBPS;
}

L2Flusher::L2Flusher() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&l2_bytes_, cudaDevAttrL2CacheSize, device));
    bytes_ = static_cast<size_t>(std::max(l2_bytes_, 1 << 20)) * 2;
    CUDA_CHECK(cudaMalloc(&buffer_, bytes_));
}

L2Flusher::~L2Flusher() { cudaFree(buffer_); }

void L2Flusher::flush() {
    CUDA_CHECK(cudaMemset(buffer_, 0, bytes_));
    CUDA_CHECK(cudaDeviceSynchronize());
}

double percentile(std::vector<float> samples, double p) {
    if (samples.empty()) return 0.0;
    std::sort(samples.begin(), samples.end());
    double rank = (p / 100.0) * (samples.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(rank));
    size_t hi = static_cast<size_t>(std::ceil(rank));
    double frac = rank - lo;
    return samples[lo] + (samples[hi] - samples[lo]) * frac;
}

void print_stats(const BenchStats& s) {
    std::cout << std::fixed << std::setprecision(4)
              << "  " << std::left << std::setw(30) << s.kernel << " [" << s.cache << "]"
              << "  p50 " << s.p50_ms << " ms | p90 " << s.p90_ms << " | p99 " << s.p99_ms
              << " | " << std::setprecision(1) << s.gbps_p50 << " GB/s (" << s.pct_of_spec_bw << "% of spec)\n";
}

std::vector<float> random_uniform(size_t n, float lo, float hi, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(gen);
    return v;
}

void symmetric_quantization(const std::vector<float>& h_mat, std::vector<uint32_t>& h_q_mat, std::vector<float>& h_scales, int rows, int cols) {
    if (cols % GROUP_SIZE != 0 || h_mat.size() != static_cast<size_t>(rows) * cols) {
        std::fprintf(stderr, "symmetric_quantization: cols (%d) must be a multiple of %d and h_mat must be rows x cols\n", cols, GROUP_SIZE);
        std::exit(EXIT_FAILURE);
    }
    h_q_mat.clear();
    h_scales.clear();
    for(size_t i = 0; i < h_mat.size(); i += GROUP_SIZE) {
        float max_abs = std::numeric_limits<float>::lowest();
        for(size_t j = i; j < i + GROUP_SIZE; j++) {
            if (std::abs(h_mat[j]) > max_abs) max_abs = std::abs(h_mat[j]);
        }
        float scale = std::max(max_abs, 1e-9f) / 7.0f; // symmetric: values land in [-7, 7] 
        h_scales.push_back(scale);
        for(size_t j = i; j < i + GROUP_SIZE; j += 8) {
            uint32_t packed = 0;
            for(int k = 0; k < 8; k++) {
                int val = static_cast<int>(std::round(h_mat[j + k] / scale));
                if(val > 7) val = 7;
                if(val < -8) val = -8;
                packed = packed | ((uint32_t)(val & 0xF) << (4 * k));
            }
            h_q_mat.push_back(packed);
        }
    }
}

std::vector<float> dequantize_int4(const std::vector<uint32_t>& h_q_mat, const std::vector<float>& h_scales, int rows, int cols) {
    std::vector<float> w(static_cast<size_t>(rows) * cols);
    for (size_t idx = 0; idx < w.size(); idx++) {
        int q = (h_q_mat[idx / 8] >> (4 * (idx % 8))) & 0xF;
        if (q > 7) q -= 16;
        w[idx] = q * h_scales[idx / GROUP_SIZE];
    }
    return w;
}

std::vector<float> cpu_gemv(const std::vector<float>& mat, const std::vector<float>& vec, int rows, int cols) {
    std::vector<float> out(rows);
    for (int r = 0; r < rows; r++) {
        double acc = 0.0;
        for (int c = 0; c < cols; c++) acc += static_cast<double>(mat[static_cast<size_t>(r) * cols + c]) * vec[c];
        out[r] = static_cast<float>(acc);
    }
    return out;
}

std::vector<float> cpu_gemm(const std::vector<float>& A, const std::vector<float>& B, int M, int in_features, int out_features) {
    std::vector<float> C(static_cast<size_t>(M) * out_features);
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < out_features; n++) {
            double acc = 0.0;
            for (int k = 0; k < in_features; k++)
                acc += static_cast<double>(A[static_cast<size_t>(m) * in_features + k]) * B[static_cast<size_t>(n) * in_features + k];
            C[static_cast<size_t>(m) * out_features + n] = static_cast<float>(acc);
        }
    }
    return C;
}

CheckResult compare(const std::string& kernel, const std::string& reference, const std::vector<float>& ref, const std::vector<float>& out, double atol, double rtol) {
    CheckResult c;
    c.kernel = kernel;
    c.reference = reference;
    c.atol = atol;
    c.rtol = rtol;
    c.pass = ref.size() == out.size();
    for (size_t i = 0; i < ref.size() && i < out.size(); i++) {
        double err = std::abs(static_cast<double>(out[i]) - ref[i]);
        if (std::isnan(out[i])) err = std::numeric_limits<double>::infinity();
        c.max_abs_err = std::max(c.max_abs_err, err);
        c.max_abs_ref = std::max(c.max_abs_ref, static_cast<double>(std::abs(ref[i])));
        if (!(err <= atol + rtol * std::abs(ref[i]))) c.pass = false;
    }
    return c;
}

void print_check(const CheckResult& c) {
    std::cout << "  " << (c.pass ? "PASS" : "FAIL") << "  " << std::left << std::setw(30) << c.kernel
              << " vs " << std::setw(22) << c.reference << std::scientific << std::setprecision(2)
              << " max|err| " << c.max_abs_err << " (max|ref| " << c.max_abs_ref << ")" << std::fixed << "\n";
}

void fill_nan(float* d_ptr, size_t count) {
    CUDA_CHECK(cudaMemset(d_ptr, 0xFF, count * sizeof(float)));  
}

std::string device_json() {
    int device = 0, driver = 0, runtime = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    std::ostringstream os;
    os << "{\"name\": \"" << prop.name << "\", \"compute_capability\": \"" << prop.major << "." << prop.minor
       << "\", \"sm_count\": " << prop.multiProcessorCount << ", \"l2_bytes\": " << prop.l2CacheSize
       << ", \"total_mem_bytes\": " << prop.totalGlobalMem << ", \"cuda_driver\": " << driver
       << ", \"cuda_runtime\": " << runtime << ", \"spec_bandwidth_gbps\": " << spec_bandwidth_gbps() << "}";
    return os.str();
}

namespace {
    std::string num(double v) {
        if (!std::isfinite(v)) return "null";
        std::ostringstream os;
        os << std::setprecision(8) << v;
        return os.str();
    }
}

void write_results_json(const std::string& path, const std::string& benchmark, const std::string& shape_json,
                        const std::vector<CheckResult>& checks, const std::vector<BenchStats>& stats) {
    std::filesystem::path p(path);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
    std::ofstream f(path);
    f << std::setprecision(8);
    f << "{\n  \"benchmark\": \"" << benchmark << "\",\n  \"device\": " << device_json() << ",\n  \"shape\": " << shape_json
      << ",\n  \"iterations\": " << NUM_ITERATIONS << ", \"warmup\": " << WARMUP_ITERATIONS << ",\n  \"validation\": [\n";
    for (size_t i = 0; i < checks.size(); i++) {
        const auto& c = checks[i];
        f << "    {\"kernel\": \"" << c.kernel << "\", \"reference\": \"" << c.reference << "\", \"pass\": "
          << (c.pass ? "true" : "false") << ", \"max_abs_err\": " << num(c.max_abs_err) << ", \"max_abs_ref\": " << num(c.max_abs_ref)
          << ", \"atol\": " << num(c.atol) << ", \"rtol\": " << num(c.rtol) << "}" << (i + 1 < checks.size() ? "," : "") << "\n";
    }
    f << "  ],\n  \"timings\": [\n";
    for (size_t i = 0; i < stats.size(); i++) {
        const auto& s = stats[i];
        f << "    {\"kernel\": \"" << s.kernel << "\", \"cache\": \"" << s.cache << "\", \"p50_ms\": " << num(s.p50_ms)
          << ", \"p90_ms\": " << num(s.p90_ms) << ", \"p99_ms\": " << num(s.p99_ms) << ", \"mean_ms\": " << num(s.mean_ms)
          << ", \"bytes\": " << num(s.bytes) << ", \"gbps_p50\": " << num(s.gbps_p50) << ", \"pct_of_spec_bw\": " << num(s.pct_of_spec_bw)
          << "}" << (i + 1 < stats.size() ? "," : "") << "\n";
    }
    f << "  ]\n}\n";
    std::cout << "\nResults written to " << path << "\n";
}