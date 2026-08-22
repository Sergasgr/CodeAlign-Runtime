#include "gemv.h"
#include <cstdint>
#define FULL_MASK 0xffffffff

const int WARP = 32; 

__global__ void gemv_int4_naive_kernel(const uint32_t* d_q_mat, const float* d_scales, const float* d_vec, float* d_out_q, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; 
    if(row < rows) {
        float sum = 0.0f;
        for(int i = 0; i < cols; i += 8) {
            uint32_t weights = d_q_mat[row * (cols / 8) + (i / 8)];
            for(int j = 0; j < 8; j++) {
                int w_int4 = (weights >> (4 * j)) & 0xF;
                if(w_int4 > 7) w_int4 -= 16;
                int idx = (row * cols + i + j) / 128;
                float w_float = w_int4 * d_scales[idx];
                sum += w_float * d_vec[i + j];
            }     
        }
        d_out_q[row] = sum;
    }
}

__global__ void gemv_int4_optimized_kernel(const uint32_t* d_q_mat, const float* d_scales, const float* d_vec, float* d_out_q, int rows, int cols) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / WARP;
    int lane_id = threadIdx.x % WARP;
    const uint4* d_q_mat4 = reinterpret_cast<const uint4*>(d_q_mat);
    const float4* d_vec4 = reinterpret_cast<const float4*>(d_vec); // NOT USED -> Checkear
    if(cols % 4 == 0 && row < rows) { 
        float sum = 0.0f;
        for(int i = lane_id; i < cols / 32; i += WARP) {
            uint4 q_weights = d_q_mat4[row * (cols / 32) + i];
            int idx = (row * cols + i * 32) / 128;
            float scale = d_scales[idx];
            uint32_t packed_weights[4] = {q_weights.x, q_weights.y, q_weights.z, q_weights.w};
            for(int j = 0; j < 4; j++) {
                for(int k = 0; k < 8; k++) {
                    int w_int4 = (packed_weights[j] >> (4 * k)) & 0xF;
                    if(w_int4 > 7) w_int4 -= 16;
                    sum += w_int4 * scale * d_vec[i * 32 + j * 8 + k];
                } 
            }
        }
        int offset = WARP / 2;
        for(int i = 0; i < 5; i++) {
            sum += __shfl_down_sync(FULL_MASK, sum, offset);
            offset /= 2;
        } 

        if(lane_id == 0) {
            d_out_q[row] = sum;
        }
    }
}

void run_gemv_int4_naive_kernel(const uint32_t* d_q_mat, const float* d_scales, const float* d_vec, float* d_out_q, int rows, int cols) {
    int block_size = 256;
    int grid_size = (rows + block_size - 1) / block_size;
    gemv_int4_naive_kernel<<<grid_size, block_size>>>(d_q_mat, d_scales, d_vec, d_out_q, rows, cols);
    cudaDeviceSynchronize();
}

void run_gemv_int4_optimized_kernel(const uint32_t* d_q_mat, const float* d_scales, const float* d_vec, float* d_out_q, int rows, int cols) {
    int block_size = 256;
    int grid_size = (rows * 32 + block_size - 1) / block_size;
    gemv_int4_optimized_kernel<<<grid_size, block_size>>>(d_q_mat, d_scales, d_vec, d_out_q, rows, cols);
    cudaDeviceSynchronize();
}