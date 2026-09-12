#pragma once
#include <cstdint>

void init_kv_cache(LayerKVCache& cache, int max_seq_len, int d);
void init_buffers(LayerBuffers& buffers, int d, int intermediate_dim, int max_seq_len);