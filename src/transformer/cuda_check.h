#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// Throws std::runtime_error (surfaced in Python as RuntimeError by pybind11) on any CUDA API failure.
inline void cuda_check(cudaError_t err, const char* call, const char* file, int line) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err) + " at " + file + ":" +
                                 std::to_string(line) + " (" + call + ")");
    }
}

#define CUDA_CHECK(call) cuda_check((call), #call, __FILE__, __LINE__)