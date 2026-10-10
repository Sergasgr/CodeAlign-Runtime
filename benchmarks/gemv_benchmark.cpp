#include "../src/gemv/gemv.h"
#include "benchmark_utils.h"
#include <cstdint>
#include <sstream>
#include <vector>

namespace {
    struct Shape {
        const char* name;
        int rows;
        int cols;
    };

    constexpr double ATOL = 1e-3;
    constexpr double RTOL = 1e-4;

    struct DeviceGemv {
        float *mat = nullptr, *vec = nullptr, *out = nullptr, *scales = nullptr, *bias = nullptr;
        uint32_t* q_mat = nullptr;
        size_t bytes_mat = 0, bytes_vec = 0, bytes_out = 0, bytes_q = 0, bytes_scales = 0;
        int rows = 0, cols = 0;

        DeviceGemv(const std::vector<float>& h_mat, const std::vector<float>& h_vec, const std::vector<uint32_t>& h_q,
                   const std::vector<float>& h_scales, const std::vector<float>& h_bias, int r, int c) : rows(r), cols(c) {
            bytes_mat = h_mat.size() * sizeof(float);
            bytes_vec = h_vec.size() * sizeof(float);
            bytes_out = static_cast<size_t>(rows) * sizeof(float);
            bytes_q = h_q.size() * sizeof(uint32_t);
            bytes_scales = h_scales.size() * sizeof(float);
            CUDA_CHECK(cudaMalloc((void**)&mat, bytes_mat));
            CUDA_CHECK(cudaMalloc((void**)&vec, bytes_vec));
            CUDA_CHECK(cudaMalloc((void**)&out, bytes_out));
            CUDA_CHECK(cudaMalloc((void**)&bias, bytes_out));
            CUDA_CHECK(cudaMalloc((void**)&q_mat, bytes_q));
            CUDA_CHECK(cudaMalloc((void**)&scales, bytes_scales));
            CUDA_CHECK(cudaMemcpy(mat, h_mat.data(), bytes_mat, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(vec, h_vec.data(), bytes_vec, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(bias, h_bias.data(), bytes_out, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(q_mat, h_q.data(), bytes_q, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(scales, h_scales.data(), bytes_scales, cudaMemcpyHostToDevice));
        }
        ~DeviceGemv() {
            cudaFree(mat); cudaFree(vec); cudaFree(out); cudaFree(bias); cudaFree(q_mat); cudaFree(scales);
        }

        template <typename Launch>
        std::vector<float> run_once(Launch launch) {
            fill_nan(out, rows);
            launch();
            CUDA_CHECK_LAST();
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<float> h(rows);
            CUDA_CHECK(cudaMemcpy(h.data(), out, bytes_out, cudaMemcpyDeviceToHost));
            return h;
        }
    };

    struct HostGemv {
        std::vector<float> mat, vec, bias, scales, deq;
        std::vector<uint32_t> q;

        HostGemv(const Shape& sh, uint32_t seed) {
            mat = random_uniform(static_cast<size_t>(sh.rows) * sh.cols, -1.0f, 1.0f, 1234 + seed);
            vec = random_uniform(sh.cols, -1.0f, 1.0f, 4321 + seed);
            bias = random_uniform(sh.rows, -1.0f, 1.0f, 777 + seed);
            symmetric_quantization(mat, q, scales, sh.rows, sh.cols);
        }
    };

    std::vector<float> plus(std::vector<float> a, const std::vector<float>& b) {
        for (size_t i = 0; i < a.size(); i++) a[i] += b[i];
        return a;
    }
}

int main() {
    const std::vector<Shape> shapes = {
        {"mlp_up_gate_4864x896", 4864, 896},
        {"mlp_down_896x4864", 896, 4864},
        {"attn_q_o_896x896", 896, 896},
        {"attn_k_v_128x896", 128, 896},
        {"lm_head_151936x896", 151936, 896},
    };

    std::vector<CheckResult> checks;
    bool all_pass = true;

    std::cout << "=== Validation (random data, CPU reference with double accumulation) ===\n";
    for (size_t s = 0; s < shapes.size(); s++) {
        const Shape& sh = shapes[s];
        std::cout << sh.name << "\n";
        HostGemv h(sh, static_cast<uint32_t>(s));
        const auto ref_fp32 = cpu_gemv(h.mat, h.vec, sh.rows, sh.cols);
        const auto ref_int4 = cpu_gemv(dequantize_int4(h.q, h.scales, sh.rows, sh.cols), h.vec, sh.rows, sh.cols);

        DeviceGemv d(h.mat, h.vec, h.q, h.scales, h.bias, sh.rows, sh.cols);
        const std::string suffix = std::string(" ") + sh.name;
        std::vector<CheckResult> shape_checks = {
            compare(std::string("fp32_naive") + suffix, "cpu_fp32", ref_fp32,
                    d.run_once([&] { run_gemv_naive(d.mat, d.vec, d.out, sh.rows, sh.cols); }), ATOL, RTOL),
            compare(std::string("fp32_optimized") + suffix, "cpu_fp32", ref_fp32,
                    d.run_once([&] { run_gemv_optimized(d.mat, d.vec, d.out, sh.rows, sh.cols); }), ATOL, RTOL),
            compare(std::string("int4_naive") + suffix, "cpu_dequantized_int4", ref_int4,
                    d.run_once([&] { run_gemv_int4_naive_kernel(d.q_mat, d.scales, d.vec, d.out, sh.rows, sh.cols); }), ATOL, RTOL),
            compare(std::string("int4_optimized") + suffix, "cpu_dequantized_int4", ref_int4,
                    d.run_once([&] { run_gemv_int4_optimized_kernel(d.q_mat, d.scales, d.vec, d.out, sh.rows, sh.cols); }), ATOL, RTOL),
            compare(std::string("int4_v2") + suffix, "cpu_dequantized_int4", ref_int4,
                    d.run_once([&] { run_gemv_int4_v2_kernel(d.q_mat, d.scales, d.vec, nullptr, d.out, sh.rows, sh.cols); }), ATOL, RTOL),
            compare(std::string("int4_v2+bias") + suffix, "cpu_dequantized_int4+bias", plus(ref_int4, h.bias),
                    d.run_once([&] { run_gemv_int4_v2_kernel(d.q_mat, d.scales, d.vec, d.bias, d.out, sh.rows, sh.cols); }), ATOL, RTOL),
        };
        for (auto& c : shape_checks) {
            print_check(c);
            all_pass = all_pass && c.pass;
            checks.push_back(c);
        }
        auto qerr = compare("int4_quantization_error", "cpu_fp32", ref_fp32, ref_int4, 0.0, 0.0);
        std::cout << "  info  INT4 quantization error vs fp32: max|err| " << std::scientific << qerr.max_abs_err
                  << " (max|ref| " << qerr.max_abs_ref << ")" << std::fixed << "\n";
    }

    if (!all_pass) {
        write_results_json("results/gemv_benchmark.json", "gemv", "{}", checks, {});
        std::cout << "\nSOME VALIDATIONS FAILED: benchmark skipped\n";
        return 1;
    }

    L2Flusher flusher;
    std::cout << "\nL2 size: " << flusher.l2_bytes() / (1024 * 1024) << " MB\n";

    const Shape& main_shape = shapes[0];
    std::vector<BenchStats> stats;
    {
        std::cout << "\n=== Benchmark: " << main_shape.name << " (" << NUM_ITERATIONS << " iterations, "
                  << WARMUP_ITERATIONS << " warmup) ===\n";
        HostGemv h(main_shape, 0);
        DeviceGemv d(h.mat, h.vec, h.q, h.scales, h.bias, main_shape.rows, main_shape.cols);
        const int rows = main_shape.rows, cols = main_shape.cols;
        const size_t bytes_fp32 = d.bytes_mat + d.bytes_vec + d.bytes_out;
        const size_t bytes_int4 = d.bytes_q + d.bytes_scales + d.bytes_vec + d.bytes_out;
        std::cout << "fp32 matrix: " << d.bytes_mat / 1e6 << " MB; INT4 matrix + scales: " << (d.bytes_q + d.bytes_scales) / 1e6 << " MB\n";
        for (L2Flusher* f : {static_cast<L2Flusher*>(nullptr), &flusher}) {
            stats.push_back(benchmark_kernel("fp32_naive", [&] { run_gemv_naive(d.mat, d.vec, d.out, rows, cols); }, bytes_fp32, f));
            stats.push_back(benchmark_kernel("fp32_optimized", [&] { run_gemv_optimized(d.mat, d.vec, d.out, rows, cols); }, bytes_fp32, f));
            stats.push_back(benchmark_kernel("int4_naive", [&] { run_gemv_int4_naive_kernel(d.q_mat, d.scales, d.vec, d.out, rows, cols); }, bytes_int4, f));
            stats.push_back(benchmark_kernel("int4_optimized", [&] { run_gemv_int4_optimized_kernel(d.q_mat, d.scales, d.vec, d.out, rows, cols); }, bytes_int4, f));
            stats.push_back(benchmark_kernel("int4_v2", [&] { run_gemv_int4_v2_kernel(d.q_mat, d.scales, d.vec, nullptr, d.out, rows, cols); }, bytes_int4, f));
        }
    }

    const Shape& lm_shape = shapes.back();
    std::vector<BenchStats> lm_stats;
    {
        std::cout << "\n=== Benchmark: " << lm_shape.name << " ===\n";
        HostGemv h(lm_shape, static_cast<uint32_t>(shapes.size() - 1));
        DeviceGemv d(h.mat, h.vec, h.q, h.scales, h.bias, lm_shape.rows, lm_shape.cols);
        const int rows = lm_shape.rows, cols = lm_shape.cols;
        const size_t bytes_fp32 = d.bytes_mat + d.bytes_vec + d.bytes_out;
        const size_t bytes_int4 = d.bytes_q + d.bytes_scales + d.bytes_vec + d.bytes_out;
        std::cout << "fp32 matrix: " << d.bytes_mat / 1e6 << " MB; INT4 matrix + scales: " << (d.bytes_q + d.bytes_scales) / 1e6
                  << " MB; v2 grid: " << gemv_int4_v2_grid_size(rows, cols) << " blocks\n";
        for (L2Flusher* f : {static_cast<L2Flusher*>(nullptr), &flusher}) {
            lm_stats.push_back(benchmark_kernel("fp32_optimized", [&] { run_gemv_optimized(d.mat, d.vec, d.out, rows, cols); }, bytes_fp32, f));
            lm_stats.push_back(benchmark_kernel("int4_optimized", [&] { run_gemv_int4_optimized_kernel(d.q_mat, d.scales, d.vec, d.out, rows, cols); }, bytes_int4, f));
            lm_stats.push_back(benchmark_kernel("int4_v2", [&] { run_gemv_int4_v2_kernel(d.q_mat, d.scales, d.vec, nullptr, d.out, rows, cols); }, bytes_int4, f));
        }
    }

    std::ostringstream shape_json, lm_json;
    shape_json << "{\"rows\": " << main_shape.rows << ", \"cols\": " << main_shape.cols << ", \"name\": \"" << main_shape.name << "\"}";
    lm_json << "\"lm_head\": {\"shape\": {\"rows\": " << lm_shape.rows << ", \"cols\": " << lm_shape.cols << ", \"name\": \""
            << lm_shape.name << "\"}, \"timings\": " << timings_json(lm_stats, "  ") << "}";
    write_results_json("results/gemv_benchmark.json", "gemv", shape_json.str(), checks, stats, lm_json.str());

    std::cout << "\nALL VALIDATIONS PASSED\n";
    return 0;
}