cmake_minimum_required(VERSION 3.24)
project(CodeAlign-Runtime LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit REQUIRED)

add_library(codealign_kernels STATIC
    src/gemv/gemv_naive.cu
    src/gemv/gemv_optimized.cu
    src/gemv/gemv_quantized.cu
    src/gemm/gemm_naive.cu
    src/gemm/gemm_optimized.cu
    src/gemm/gemm_quantized.cu
)
target_include_directories(codealign_kernels PUBLIC src)
target_compile_options(codealign_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)
target_link_libraries(codealign_kernels PUBLIC CUDA::cudart)

add_executable(gemv_benchmark
    benchmarks/gemv_benchmark.cpp
    benchmarks/benchmark_utils.cpp
)
target_include_directories(gemv_benchmark PRIVATE benchmarks)
target_link_libraries(gemv_benchmark PRIVATE codealign_kernels)

add_executable(gemm_benchmark
    benchmarks/gemm_benchmark.cpp
    benchmarks/benchmark_utils.cpp
)
target_include_directories(gemm_benchmark PRIVATE benchmarks)
target_link_libraries(gemm_benchmark PRIVATE codealign_kernels)