#pragma once

void run_add_residual_kernel(float* base_vector, float* add_vector, int d);
void run_add_bias_kernel(float* x, const float* bias, int num_tokens, int n);