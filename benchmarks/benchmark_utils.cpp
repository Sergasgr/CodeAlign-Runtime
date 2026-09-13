#include "benchmark_utils.h"

constexpr int GROUP = 128; // Symmetric INT4 group-wise quantization (group size 128)

void symmetric_quantization(const std::vector<float>& h_mat, std::vector<uint32_t>& h_q_mat, std::vector<float>& h_scales, int rows, int cols) { 
    for(int i = 0; i < rows * cols; i += GROUP) {
        float max_abs = std::numeric_limits<float>::lowest();
        for(int j = i; j < i + GROUP; j++) {
            if (std::abs(h_mat[j]) > max_abs) max_abs = std::abs(h_mat[j]);
        }
        float scale = std::max(max_abs, 1e-9f) / 7.0f; // INT4 -> [-8, 7]
        h_scales.push_back(scale);
        for(int j = i; j < i + GROUP; j += 8) {
            uint32_t packed = 0;
            for(int k = 0; k < 8; k++) {
                int val = round(h_mat[j + k] / scale);
                if(val > 7) val = 7;
                if(val < -8) val = -8;
                packed = packed | ((uint32_t)(val & 0xF) << (4 * k));
            }     
            h_q_mat.push_back(packed);
        }
    }
}