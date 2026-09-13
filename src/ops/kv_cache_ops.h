#pragma once

void run_append_kv_cache(const float* k_src, const float* v_src, float* k_cache, float* v_cache, int current_seq_len, int max_seq_len, int head_dim, int d);