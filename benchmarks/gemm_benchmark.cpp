#include "../src/gemm/gemm.h"
#include "benchmark_utils.h"
#include <cstdint>
#include <sstream>
#include <vector>

namespace {
    struct Shape {
        const char* name;
        int M;            
        int in_features;
        int out_features;
    };

    constexpr int NUM_SPLITS = 4;  
    constexpr double ATOL = 1e-3;
    constexpr double RTOL = 1e-4;

    struct DeviceGemm {
        float *A = nullptr, *B = nullptr, *C = nullptr, *scales = nullptr, *partial = nullptr;
        uint32_t* B_q = nullptr;
        size_t bytes_A = 0, bytes_B = 0, bytes_C = 0, bytes_Bq = 0, bytes_scales = 0;
        int M = 0, in = 0, out = 0;

        DeviceGemm(const std::vector<float>& h_A, const std::vector<float>& h_B, const std::vector<uint32_t>& h_Bq,
                   const std::vector<float>& h_scales, int m, int in_f, int out_f)
            : M(m), in(in_f), out(out_f) {
            bytes_A = h_A.size() * sizeof(float);
            bytes_B = h_B.size() * sizeof(float);
            bytes_C = static_cast<size_t>(M) * out * sizeof(float);
            bytes_Bq = h_Bq.size() * sizeof(uint32_t);
            bytes_scales = h_scales.size() * sizeof(float);
            CUDA_CHECK(cudaMalloc((void**)&A, bytes_A));
            CUDA_CHECK(cudaMalloc((void**)&B, bytes_B));
            CUDA_CHECK(cudaMalloc((void**)&C, bytes_C));
            CUDA_CHECK(cudaMalloc((void**)&B_q, bytes_Bq));
            CUDA_CHECK(cudaMalloc((void**)&scales, bytes_scales));
            CUDA_CHECK(cudaMalloc((void**)&partial, NUM_SPLITS * bytes_C));
            CUDA_CHECK(cudaMemcpy(A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(B_q, h_Bq.data(), bytes_Bq, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(scales, h_scales.data(), bytes_scales, cudaMemcpyHostToDevice));
        }
        ~DeviceGemm() {
            cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(B_q); cudaFree(scales); cudaFree(partial);
        }

        void splitk() const {
            run_gemm_int4_splitk_partial_kernel(A, B_q, scales, partial, M, NUM_SPLITS, in, out);
            run_gemm_int4_splitk_final_kernel(partial, C, M, out, NUM_SPLITS);
        }

        template <typename Launch>
        std::vector<float> run_once(Launch launch) {
            fill_nan(C, static_cast<size_t>(M) * out);
            launch();
            CUDA_CHECK_LAST();
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<float> h(static_cast<size_t>(M) * out);
            CUDA_CHECK(cudaMemcpy(h.data(), C, bytes_C, cudaMemcpyDeviceToHost));
            return h;
        }
    };
}  

int main() {
    const std::vector<Shape> shapes = {
        {"M16_mlp_up_896->4864", 16, 896, 4864},
        {"M11_mlp_down_4864->896", 11, 4864, 896},
        {"M3_attn_k_896->128", 3, 896, 128},
    };

    std::vector<CheckResult> checks;
    std::vector<BenchStats> stats;
    bool all_pass = true;

    std::cout << "=== Validation (random data, CPU reference with double accumulation) ===\n";
    for (size_t s = 0; s < shapes.size(); s++) {
        const Shape& sh = shapes[s];
        std::cout << sh.name << "\n";
        auto h_A = random_uniform(static_cast<size_t>(sh.M) * sh.in_features, -1.0f, 1.0f, 777 + s);
        auto h_B = random_uniform(static_cast<size_t>(sh.out_features) * sh.in_features, -1.0f, 1.0f, 999 + s);
        std::vector<uint32_t> h_Bq;
        std::vector<float> h_scales;
        symmetric_quantization(h_B, h_Bq, h_scales, sh.out_features, sh.in_features);
        auto h_Bdeq = dequantize_int4(h_Bq, h_scales, sh.out_features, sh.in_features);

        const auto ref_fp32 = cpu_gemm(h_A, h_B, sh.M, sh.in_features, sh.out_features);
        const auto ref_int4 = cpu_gemm(h_A, h_Bdeq, sh.M, sh.in_features, sh.out_features);

        DeviceGemm d(h_A, h_B, h_Bq, h_scales, sh.M, sh.in_features, sh.out_features);
        const std::string suffix = std::string(" ") + sh.name;
        std::vector<CheckResult> shape_checks = {
            compare("fp32_naive" + suffix, "cpu_fp32", ref_fp32,
                    d.run_once([&] { run_gemm_naive(d.A, d.B, d.C, d.M, d.in, d.out); }), ATOL, RTOL),
            compare("fp32_tiled" + suffix, "cpu_fp32", ref_fp32,
                    d.run_once([&] { run_gemm_optimized(d.A, d.B, d.C, d.M, d.in, d.out); }), ATOL, RTOL),
            compare("int4_naive" + suffix, "cpu_dequantized_int4", ref_int4,
                    d.run_once([&] { run_gemm_int4_naive(d.A, d.B_q, d.scales, d.C, d.M, d.in, d.out); }), ATOL, RTOL),
            compare("int4_tiled" + suffix, "cpu_dequantized_int4", ref_int4,
                    d.run_once([&] { run_gemm_int4_optimized(d.A, d.B_q, d.scales, d.C, d.M, d.in, d.out); }), ATOL, RTOL),
            compare("int4_splitk" + suffix, "cpu_dequantized_int4", ref_int4,
                    d.run_once([&] { d.splitk(); }), ATOL, RTOL),
        };
        for (auto& c : shape_checks) {
            print_check(c);
            all_pass = all_pass && c.pass;
            checks.push_back(c);
        }
    }

    if (!all_pass) {
        write_results_json("results/gemm_benchmark.json", "gemm", "{}", checks, {});
        std::cout << "\nSOME VALIDATIONS FAILED: benchmark skipped\n";
        return 1;
    }

    const Shape& sh = shapes[0];
    std::cout << "\n=== Benchmark: " << sh.name << " (" << NUM_ITERATIONS << " iterations, " << WARMUP_ITERATIONS
              << " warmup) ===\n";
    auto h_A = random_uniform(static_cast<size_t>(sh.M) * sh.in_features, -1.0f, 1.0f, 777);
    auto h_B = random_uniform(static_cast<size_t>(sh.out_features) * sh.in_features, -1.0f, 1.0f, 999);
    std::vector<uint32_t> h_Bq;
    std::vector<float> h_scales;
    symmetric_quantization(h_B, h_Bq, h_scales, sh.out_features, sh.in_features);
    DeviceGemm d(h_A, h_B, h_Bq, h_scales, sh.M, sh.in_features, sh.out_features);

    const size_t bytes_fp32 = d.bytes_A + d.bytes_B + d.bytes_C;
    const size_t bytes_int4 = d.bytes_A + d.bytes_Bq + d.bytes_scales + d.bytes_C;

    L2Flusher flusher;
    for (L2Flusher* f : {static_cast<L2Flusher*>(nullptr), &flusher}) {
        stats.push_back(benchmark_kernel("fp32_naive", [&] { run_gemm_naive(d.A, d.B, d.C, d.M, d.in, d.out); }, bytes_fp32, f));
        stats.push_back(benchmark_kernel("fp32_tiled", [&] { run_gemm_optimized(d.A, d.B, d.C, d.M, d.in, d.out); }, bytes_fp32, f));
        stats.push_back(benchmark_kernel("int4_naive", [&] { run_gemm_int4_naive(d.A, d.B_q, d.scales, d.C, d.M, d.in, d.out); }, bytes_int4, f));
        stats.push_back(benchmark_kernel("int4_tiled", [&] { run_gemm_int4_optimized(d.A, d.B_q, d.scales, d.C, d.M, d.in, d.out); }, bytes_int4, f));
        stats.push_back(benchmark_kernel("int4_splitk", [&] { d.splitk(); }, bytes_int4, f));
    }

    std::ostringstream shape_json;
    shape_json << "{\"M\": " << sh.M << ", \"in_features\": " << sh.in_features << ", \"out_features\": " << sh.out_features
               << ", \"num_splits\": " << NUM_SPLITS << ", \"name\": \"" << sh.name << "\"}";
    write_results_json("results/gemm_benchmark.json", "gemm", shape_json.str(), checks, stats);

    std::cout << "\nALL VALIDATIONS PASSED\n";
    return 0;
}