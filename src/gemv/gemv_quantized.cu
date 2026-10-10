#include "../include/codealign_cuda.h"
#include "gemv.h"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#define FULL_MASK 0xffffffff

constexpr int WARP = 32;

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
    if(cols % 32 == 0 && row < rows) {
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

void run_gemv_int4_naive_kernel(const uint32_t* d_q_mat, const float* d_scales, const float* d_vec, float* d_out_q, int rows, int cols, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (rows + block_size - 1) / block_size;
    CODEALIGN_KERNEL_LAUNCH(gemv_int4_naive_kernel, grid_size, block_size, 0, stream, d_q_mat, d_scales, d_vec, d_out_q, rows, cols);
}

void run_gemv_int4_optimized_kernel(const uint32_t* d_q_mat, const float* d_scales, const float* d_vec, float* d_out_q, int rows, int cols, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (rows * 32 + block_size - 1) / block_size;
    CODEALIGN_KERNEL_LAUNCH(gemv_int4_optimized_kernel, grid_size, block_size, 0, stream, d_q_mat, d_scales, d_vec, d_out_q, rows, cols);
}

namespace {
    constexpr int V2_WARPS = 8;                    
    constexpr int V2_THREADS = V2_WARPS * WARP;
    constexpr int V2_LONG_ROW_COLS = 2048;         
    constexpr int V2_WORDS_PER_LANE = 8;          
    constexpr int V2_BLOCKS_PER_SM = 4;
    constexpr int V2_MAX_SMEM_BYTES = 48 * 1024;   

    __device__ __forceinline__ float nibble_to_float(uint32_t offset_binary, int k) {
        return __uint_as_float(0x4B000000u | ((offset_binary >> (4 * k)) & 0xFu)) - 8388616.0f;
    }

    __device__ __forceinline__ float dot8(uint32_t word, const float4& a, const float4& b) {
        const uint32_t u = word ^ 0x88888888u;
        float d = nibble_to_float(u, 0) * a.x;
        d = fmaf(nibble_to_float(u, 1), a.y, d);
        d = fmaf(nibble_to_float(u, 2), a.z, d);
        d = fmaf(nibble_to_float(u, 3), a.w, d);
        d = fmaf(nibble_to_float(u, 4), b.x, d);
        d = fmaf(nibble_to_float(u, 5), b.y, d);
        d = fmaf(nibble_to_float(u, 6), b.z, d);
        d = fmaf(nibble_to_float(u, 7), b.w, d);
        return d;
    }
}  

template <int ROWS>
__global__ void __launch_bounds__(V2_THREADS, V2_BLOCKS_PER_SM)
gemv_int4_v2_kernel(const uint32_t* __restrict__ q, const float* __restrict__ scales, const float* __restrict__ x,
                    const float* __restrict__ bias, float* __restrict__ y, int rows, int cols) {
    extern __shared__ float4 x_smem[];
    const int words = cols / 8;            
    const int groups = cols / 128;      
    float4* xa = x_smem;
    float4* xb = x_smem + words;

    const float4* x4 = reinterpret_cast<const float4*>(x);
    for (int i = threadIdx.x; i < 2 * words; i += V2_THREADS) {
        const float4 v = x4[i];
        if (i & 1) xb[i >> 1] = v; else xa[i >> 1] = v;
    }
    __syncthreads();

    const int lane = threadIdx.x % WARP;
    const int num_warps = gridDim.x * V2_WARPS;
    for (int r0 = (blockIdx.x * V2_WARPS + threadIdx.x / WARP) * ROWS; r0 < rows; r0 += num_warps * ROWS) {
        const int nr = min(ROWS, rows - r0);
        const int span = nr * words;                                 
        const uint32_t* q_span = q + static_cast<size_t>(r0) * words;
        const float* s_span = scales + static_cast<size_t>(r0) * groups;   

        float acc[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;

        for (int f0 = 0; f0 < span; f0 += WARP * V2_WORDS_PER_LANE) {
            uint32_t word[V2_WORDS_PER_LANE];
            float scale[V2_WORDS_PER_LANE];
#pragma unroll
            for (int u = 0; u < V2_WORDS_PER_LANE; ++u) {      
                const int f = f0 + u * WARP + lane;
                word[u] = 0u;
                scale[u] = 0.0f;
                if (f < span) {
                    word[u] = q_span[f];
                    scale[u] = s_span[f >> 4];
                }
            }
#pragma unroll
            for (int u = 0; u < V2_WORDS_PER_LANE; ++u) {
                const int f = f0 + u * WARP + lane;
                if (f < span) {
                    int r_local = 0;                               
#pragma unroll
                    for (int k = 1; k < ROWS; ++k) r_local += (f >= k * words);
                    const int w = f - r_local * words;         
                    const float v = scale[u] * dot8(word[u], xa[w], xb[w]);
#pragma unroll
                    for (int r = 0; r < ROWS; ++r) {
                        if (r == r_local) acc[r] += v;
                    }
                }
            }
        }

#pragma unroll
        for (int offset = WARP / 2; offset > 0; offset >>= 1) {
#pragma unroll
            for (int r = 0; r < ROWS; ++r) acc[r] += __shfl_down_sync(FULL_MASK, acc[r], offset);
        }
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                if (r < nr) y[r0 + r] = acc[r] + (bias != nullptr ? bias[r0 + r] : 0.0f);
            }
        }
    }
}

int gemv_int4_v2_rows_per_warp(int cols) {
    return cols >= V2_LONG_ROW_COLS ? 1 : 2;
}

int gemv_int4_v2_grid_size(int rows, int cols) {
    const int rows_per_warp = gemv_int4_v2_rows_per_warp(cols);
    const int warps_needed = (rows + rows_per_warp - 1) / rows_per_warp;
    const int blocks_needed = (warps_needed + V2_WARPS - 1) / V2_WARPS;
    return std::max(1, std::min(blocks_needed, V2_BLOCKS_PER_SM * codealign::current_device_sm_count()));
}

void run_gemv_int4_v2_kernel(const uint32_t* d_q_mat, const float* d_scales, const float* d_vec, const float* d_bias,
                             float* d_out, int rows, int cols, cudaStream_t stream) {
    const size_t shared_mem_bytes = static_cast<size_t>(cols) * sizeof(float);
    if (rows < 1 || cols < 128 || cols % 128 != 0 || shared_mem_bytes > V2_MAX_SMEM_BYTES) {
        throw std::runtime_error("run_gemv_int4_v2_kernel: need rows >= 1 and cols a multiple of 128 with cols <= " +
                                 std::to_string(V2_MAX_SMEM_BYTES / 4) + ", got rows=" + std::to_string(rows) +
                                 ", cols=" + std::to_string(cols));
    }
    if (reinterpret_cast<std::uintptr_t>(d_vec) % 16 != 0) {
        throw std::runtime_error("run_gemv_int4_v2_kernel: the input vector must be 16-byte aligned (float4 loads)");
    }
    const int grid = gemv_int4_v2_grid_size(rows, cols);
    if (gemv_int4_v2_rows_per_warp(cols) == 1) {
        CODEALIGN_KERNEL_LAUNCH(gemv_int4_v2_kernel<1>, grid, V2_THREADS, shared_mem_bytes, stream, d_q_mat, d_scales, d_vec, d_bias, d_out, rows, cols);
    } else {
        CODEALIGN_KERNEL_LAUNCH(gemv_int4_v2_kernel<2>, grid, V2_THREADS, shared_mem_bytes, stream, d_q_mat, d_scales, d_vec, d_bias, d_out, rows, cols);
    }
}