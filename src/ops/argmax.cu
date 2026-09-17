#include "logits_ops.h"
#include <cuda_runtime.h>

#define FULL_MASK 0xffffffff
#define WARP_SIZE 32 

__global__ void compute_argmax_kernel(const float* logits, int* predicted_tokens, int vocab_size) {
    int token_id = blockIdx.x;
    int tid = threadIdx.x;   
    int row_offset = token_id * vocab_size; 

    float max_val = -1e9f;
    int max_idx = -1;

    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = logits[row_offset + i];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float neighbor_val = __shfl_down_sync(FULL_MASK, max_val, offset);
        int neighbor_idx = __shfl_down_sync(FULL_MASK, max_idx, offset);
        
        if (neighbor_val > max_val) {
            max_val = neighbor_val;
            max_idx = neighbor_idx;
        }
    }

    __shared__ float max_values[WARP_SIZE];
    __shared__ int max_indices[WARP_SIZE];

    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    if (lane_id == 0) {
        max_values[warp_id] = max_val;
        max_indices[warp_id] = max_idx;
    }
    __syncthreads(); 

    if (warp_id == 0) {
        int num_warps = blockDim.x / WARP_SIZE;
        max_val = (lane_id < num_warps) ? max_values[lane_id] : -1e9f;
        max_idx = (lane_id < num_warps) ? max_indices[lane_id] : -1;

        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            float neighbor_val = __shfl_down_sync(FULL_MASK, max_val, offset);
            int neighbor_idx = __shfl_down_sync(FULL_MASK, max_idx, offset);
            if (neighbor_val > max_val) {
                max_val = neighbor_val;
                max_idx = neighbor_idx;
            }
        }

        if (tid == 0) {
            predicted_tokens[token_id] = max_idx;
        }
    }

} 

void run_compute_argmax_kernel(float* logits, int num_tokens) {
    int block_size = 256; // 512?
    int grid_size = num_tokens;
    compute_argmax_kernel<<<grid_size, block_size>>>(logits, predicted_tokens, vocab_size);
    cudaDeviceSynchronize();
}


