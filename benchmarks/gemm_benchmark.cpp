#include "../src/gemv/gemm.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdint>
#include <limits>
#include <cmath>
 
constexpr int GROUP = 128; // Symmetric INT4 group-wise quantization (group size 128)
constexpr int NUM_ITERATIONS = 100;

int main() {
    int K = 16; 
    int in_features = 896;
    int out_features = 4864;

    size_t bytes_A = K * in_features * sizeof(float);
    size_t bytes_B = out_features * in_features * sizeof(float);
    size_t bytes_C = K * out_features * sizeof(float);

    return 0;
}