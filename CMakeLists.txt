cmake_minimum_required(VERSION 3.24)
project(CodeAlign-Runtime LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(gemv_benchmark
    benchmarks/gemv_benchmark.cpp
    benchmarks/benchmark_utils.cpp
    src/gemv/gemv_naive.cu
    src/gemv/gemv_optimized.cu
    src/gemv/gemv_quantized.cu
    src/ops/flash_decoding_partial.cu
    src/ops/flash_decoding_final.cu
)
target_include_directories(gemv_benchmark PRIVATE src benchmarks)

add_executable(gemm_benchmark
    benchmarks/gemm_benchmark.cpp
    benchmarks/benchmark_utils.cpp
    src/gemm/gemm_naive.cu
    src/gemm/gemm_optimized.cu
    src/gemm/gemm_quantized.cu
)
target_include_directories(gemm_benchmark PRIVATE src benchmarks)