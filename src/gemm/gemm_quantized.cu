#include "gemm.h"
#include <cstdint>
#define TILE_SIZE 16

__global__ void gemm_int4_naive_kernel(const float* A, const uint32_t* B_quant, const float* B_scales, float* C, int K, int in_features, int out_features) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < K && col < out_features) { 
        float sum = 0.0f;
        for(int i = 0; i < in_features; i += 8) {
            uint32_t weights = B_quant[col * (in_features / 8) + (i / 8)];
            int scale_idx = (col * in_features + i) / 128;
            float scale = B_scales[scale_idx];
            for(int j = 0; j < 8; j++) {
                int w_int4 = (weights >> (4 * j)) & 0xF;
                if(w_int4 > 7) w_int4 -= 16;
                float w_float = w_int4 * scale;
                sum += A[row * in_features + i + j] * w_float;
            }
        }
        C[row * out_features + col] = sum;
    }
}

__global__ void gemm_int4_optimized_kernel(const float* A, const uint32_t* B_quant, const float* B_scales, float* C, int K, int in_features, int out_features) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;
    int num_tiles = (in_features + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < num_tiles; t++) {
        int global_col_A = t * TILE_SIZE + tx;
        if(row < K && global_col_A < in_features) sA[ty][tx] = A[row * in_features + global_col_A];
        else sA[ty][tx] = 0.0f;

        int global_row_B = col; 
        int global_col_B = t * TILE_SIZE + ty; 
        
        if (global_row_B < out_features && global_col_B < in_features) {
            int pack_idx = global_row_B * (in_features / 8) + (global_col_B / 8);
            uint32_t packed_val = B_quant[pack_idx];

            int bit_shift = (global_col_B % 8) * 4;
            int w_int4 = (packed_val >> bit_shift) & 0xF;
            if (w_int4 > 7) w_int4 -= 16;
            
            int scale_idx = (global_row_B * in_features + global_col_B) / 128;
            float scale = B_scales[scale_idx];
            
            sB[ty][tx] = w_int4 * scale;
        } else {
            sB[ty][tx] = 0.0f;
        }

        __syncthreads();

        for(int i = 0; i < TILE_SIZE; i++) {
            sum += sA[ty][i] * sB[i][tx];
        }

        __syncthreads();
    }

    if (row < K && col < out_features) {
        C[row * out_features + col] = sum;
    }
}

__global__ void gemm_int4_splitk_partial_kernel(const float* A, const uint32_t* B_quant, const float* B_scales, float* C_partial, int K, int num_splits, int in_features, int out_features) { 
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int p = blockIdx.z;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    int total_tiles = (in_features + TILE_SIZE - 1) / TILE_SIZE;
    int tiles_per_split = (total_tiles + num_splits - 1) / num_splits;
    int start_tile = p * tiles_per_split;
    int end_tile = min(start_tile + tiles_per_split, total_tiles);

    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];
    float sum = 0.0f;

    for(int t = start_tile; t < end_tile; t++) {
        int global_col_A = t * TILE_SIZE + tx;
        if(row < K && global_col_A < in_features) sA[ty][tx] = A[row * in_features + global_col_A];
        else sA[ty][tx] = 0.0f;

        int global_row_B = col; 
        int global_col_B = t * TILE_SIZE + ty; 
        
        if (global_row_B < out_features && global_col_B < in_features) {
            int pack_idx = global_row_B * (in_features / 8) + (global_col_B / 8);
            uint32_t packed_val = B_quant[pack_idx];

            int bit_shift = (global_col_B % 8) * 4;
            int w_int4 = (packed_val >> bit_shift) & 0xF;
            if (w_int4 > 7) w_int4 -= 16;
            
            int scale_idx = (global_row_B * in_features + global_col_B) / 128;
            float scale = B_scales[scale_idx];
            
            sB[ty][tx] = w_int4 * scale;
        } else {
            sB[ty][tx] = 0.0f;
        }

        __syncthreads();

        for(int i = 0; i < TILE_SIZE; i++) {
            sum += sA[ty][i] * sB[i][tx];
        }

        __syncthreads();
    }

     if (row < K && col < out_features) {
        C_partial[p * (K * out_features) + row * out_features + col] = sum;
    }

}

__global__ void gemm_int4_splitk_final_kernel(const float* C_partial, float* C, int K, int out_features, int num_splits) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    if(row < K && col < out_features) {
        float final_sum = 0.0f;
        for(int s = 0; s < num_splits; s++) {
            final_sum += C_partial[s * (K * out_features) + row * out_features + col];
        }
        C[row * out_features + col] = final_sum;
    }
}

void run_gemm_int4_naive(const float* A, const uint32_t* B_quant, const float* B_scales, float* C, int K, int in_features, int out_features) {
    dim3 block_size(16, 16);
    dim3 grid_size((out_features + 15) / 16, (K + 15) / 16);
    gemm_int4_naive_kernel<<<grid_size, block_size>>>(A, B_quant, B_scales, C, K, in_features, out_features);
    cudaDeviceSynchronize();
}

void run_gemm_int4_optimized(const float* A, const uint32_t* B_quant, const float* B_scales, float* C, int K, int in_features, int out_features) {
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((out_features + TILE_SIZE - 1) / TILE_SIZE, (K + TILE_SIZE - 1) / TILE_SIZE);
    gemm_int4_optimized_kernel<<<grid_size, block_size>>>(A, B_quant, B_scales, C, K, in_features, out_features);
    cudaDeviceSynchronize();
}

void run_gemm_int4_splitk_partial_kernel(const float* A, const uint32_t* B_quant, const float* B_scales, float* C_partial, int K, int num_splits, int in_features, int out_features) {
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((out_features + TILE_SIZE - 1) / TILE_SIZE, (K + TILE_SIZE - 1) / TILE_SIZE, num_splits);
    gemm_int4_splitk_partial_kernel<<<grid_size, block_size>>>(A, B_quant, B_scales, C_partial, K, num_splits, in_features, out_features);
    cudaDeviceSynchronize();
}

void run_gemm_int4_splitk_final_kernel(const float* C_partial, float* C, int K, int out_features, int num_splits) {
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((out_features + TILE_SIZE - 1) / TILE_SIZE, (K + TILE_SIZE - 1) / TILE_SIZE);
    gemm_int4_splitk_final_kernel<<<grid_size, block_size>>>(C_partial, C, K, out_features, num_splits);
    cudaDeviceSynchronize();
}