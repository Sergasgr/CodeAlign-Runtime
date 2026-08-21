#include "gemv.h"
#include <iostream> //innecesario?
#define FULL_MASK 0xffffffff

const int WARP = 32; 

__global__ void gemv_optimized_kernel(const float* d_mat, const float* d_vec, float* d_out_opt, int rows, int cols) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / WARP;
    int lane_id = threadIdx.x % WARP;
    const float4* d_mat4 = reinterpret_cast<const float4*>(d_mat);
    const float4* d_vec4 = reinterpret_cast<const float4*>(d_vec); 
    if(cols % 4 == 0 && row < rows) { 
        float sum = 0.0f;
        for(int i = lane_id; i < cols / 4; i += WARP) {
            sum += d_mat4[row * (cols / 4) + i].x * d_vec4[i].x +
                   d_mat4[row * (cols / 4) + i].y * d_vec4[i].y +
                   d_mat4[row * (cols / 4) + i].z * d_vec4[i].z +
                   d_mat4[row * (cols / 4) + i].w * d_vec4[i].w;
        }
        int offset = WARP / 2;
        for(int i = 0; i < 5; i++) {
            sum += __shfl_down_sync(FULL_MASK, sum, offset);
            offset /= 2;
        } 
        if(lane_id == 0) {
            d_out_opt[row] = sum;
        }
    }
}

void run_gemv_optimized(const float* d_mat, const float* d_vec, float* d_out_opt, int rows, int cols) {
    int block_size = 256;
    int grid_size = (rows * 32 + block_size - 1) / block_size;
    gemv_optimized_kernel<<<grid_size, block_size>>>(d_mat, d_vec, d_out_opt, rows, cols);
    cudaDeviceSynchronize();
}