cmake_minimum_required(VERSION 3.24)
project(CodeAlign-Runtime LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(gemv_benchmark
    src/main.cpp
    src/gemv_naive.cu
    src/gemv_optimized.cu
    src/gemv_quantized.cu
)

target_include_directories(gemv_benchmark PRIVATE src)