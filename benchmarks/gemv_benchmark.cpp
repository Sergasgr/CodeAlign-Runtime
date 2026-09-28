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
        float *mat = nullptr, *vec = nullptr, *out = nullptr, *scales = nullptr;
        uint32_t* q_mat = nullptr;
        size_t bytes_mat = 0, bytes_vec = 0, bytes_out = 0, bytes_q = 0, bytes_scales = 0;
        int rows = 0, cols = 0;

        DeviceGemv(const std::vector<float>& h_mat, const std::vector<float>& h_vec, const std::vector<uint32_t>& h_q,
                const std::vector<float>& h_scales, int r, int c)
            : rows(r), cols(c) {
            bytes_mat = h_mat.size() * sizeof(float);
            bytes_vec = h_vec.size() * sizeof(float);
            bytes_out = static_cast<size_t>(rows) * sizeof(float);
            bytes_q = h_q.size() * sizeof(uint32_t);
            bytes_scales = h_scales.size() * sizeof(float);
            CUDA_CHECK(cudaMalloc((void**)&mat, bytes_mat));
            CUDA_CHECK(cudaMalloc((void**)&vec, bytes_vec));
            CUDA_CHECK(cudaMalloc((void**)&out, bytes_out));
            CUDA_CHECK(cudaMalloc((void**)&q_mat, bytes_q));
            CUDA_CHECK(cudaMalloc((void**)&scales, bytes_scales));
            CUDA_CHECK(cudaMemcpy(mat, h_mat.data(), bytes_mat, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(vec, h_vec.data(), bytes_vec, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(q_mat, h_q.data(), bytes_q, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(scales, h_scales.data(), bytes_scales, cudaMemcpyHostToDevice));
        }
        ~DeviceGemv() {
            cudaFree(mat); cudaFree(vec); cudaFree(out); cudaFree(q_mat); cudaFree(scales);
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
}  

int main() {
    const std::vector<Shape> shapes = {
        {"mlp_up_gate_4864x896", 4864, 896},
        {"mlp_down_896x4864", 896, 4864},
        {"attn_q_o_896x896", 896, 896},
        {"attn_k_v_128x896", 128, 896},
    };

    std::vector<CheckResult> checks;
    std::vector<BenchStats> stats;
    bool all_pass = true;

    std::cout << "=== Validation (random data, CPU reference with double accumulation) ===\n";
    for (size_t s = 0; s < shapes.size(); s++) {
        const Shape& sh = shapes[s];
        std::cout << sh.name << "\n";
        auto h_mat = random_uniform(static_cast<size_t>(sh.rows) * sh.cols, -1.0f, 1.0f, 1234 + s);
        auto h_vec = random_uniform(sh.cols, -1.0f, 1.0f, 4321 + s);
        std::vector<uint32_t> h_q;
        std::vector<float> h_scales;
        symmetric_quantization(h_mat, h_q, h_scales, sh.rows, sh.cols);
        auto h_deq = dequantize_int4(h_q, h_scales, sh.rows, sh.cols);

        const auto ref_fp32 = cpu_gemv(h_mat, h_vec, sh.rows, sh.cols);
        const auto ref_int4 = cpu_gemv(h_deq, h_vec, sh.rows, sh.cols);

        DeviceGemv d(h_mat, h_vec, h_q, h_scales, sh.rows, sh.cols);
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

    const Shape& main_shape = shapes[0];
    std::cout << "\n=== Benchmark: " << main_shape.name << " (" << NUM_ITERATIONS << " iterations, "
              << WARMUP_ITERATIONS << " warmup) ===\n";
    auto h_mat = random_uniform(static_cast<size_t>(main_shape.rows) * main_shape.cols, -1.0f, 1.0f, 1234);
    auto h_vec = random_uniform(main_shape.cols, -1.0f, 1.0f, 4321);
    std::vector<uint32_t> h_q;
    std::vector<float> h_scales;
    symmetric_quantization(h_mat, h_q, h_scales, main_shape.rows, main_shape.cols);
    DeviceGemv d(h_mat, h_vec, h_q, h_scales, main_shape.rows, main_shape.cols);
    const int rows = main_shape.rows, cols = main_shape.cols;

    const size_t bytes_fp32 = d.bytes_mat + d.bytes_vec + d.bytes_out;
    const size_t bytes_int4 = d.bytes_q + d.bytes_scales + d.bytes_vec + d.bytes_out;

    L2Flusher flusher;
    std::cout << "L2 size: " << flusher.l2_bytes() / (1024 * 1024) << " MB; fp32 matrix: " << d.bytes_mat / 1e6
              << " MB; INT4 matrix + scales: " << (d.bytes_q + d.bytes_scales) / 1e6 << " MB\n";
    for (L2Flusher* f : {static_cast<L2Flusher*>(nullptr), &flusher}) {
        stats.push_back(benchmark_kernel("fp32_naive", [&] { run_gemv_naive(d.mat, d.vec, d.out, rows, cols); }, bytes_fp32, f));
        stats.push_back(benchmark_kernel("fp32_optimized", [&] { run_gemv_optimized(d.mat, d.vec, d.out, rows, cols); }, bytes_fp32, f));
        stats.push_back(benchmark_kernel("int4_naive", [&] { run_gemv_int4_naive_kernel(d.q_mat, d.scales, d.vec, d.out, rows, cols); }, bytes_int4, f));
        stats.push_back(benchmark_kernel("int4_optimized", [&] { run_gemv_int4_optimized_kernel(d.q_mat, d.scales, d.vec, d.out, rows, cols); }, bytes_int4, f));
    }

    std::ostringstream shape_json;
    shape_json << "{\"rows\": " << rows << ", \"cols\": " << cols << ", \"name\": \"" << main_shape.name << "\"}";
    write_results_json("results/gemv_benchmark.json", "gemv", shape_json.str(), checks, stats);

    std::cout << "\nALL VALIDATIONS PASSED\n";
    return 0;
}