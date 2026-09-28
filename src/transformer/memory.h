#pragma once
#include <cstdint>
#include "transformer.h"

void init_kv_cache(LayerKVCache& cache, int max_seq_len, const ModelDims& dims);
void init_buffers(LayerBuffers& buffers, const ModelDims& dims, int max_seq_len);
void free_kv_cache(LayerKVCache& cache);
void free_buffers(LayerBuffers& buffers);