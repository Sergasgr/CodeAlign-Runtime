#pragma once

void run_compute_argmax_kernel(const float* logits, int* predicted_tokens, int num_tokens, int vocab_size);