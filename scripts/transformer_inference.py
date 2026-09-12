import torch
import codealign_runtime_transformer as cuda_engine
from scripts.inference_config import (
    D, 
    INTERMEDIATE_DIM,
    MAX_SEQ_LEN,
    NUM_ITERATIONS
)

inference_start = torch.cuda.Event(enable_timing=True)
inference_end = torch.cuda.Event(enable_timing=True)
warmup_steps = min(10, max(1, NUM_ITERATIONS // 10))

block = cuda_engine.QwenBlock(D, INTERMEDIATE_DIM, MAX_SEQ_LEN)

attn_norm = torch.ones(D, dtype=torch.float32, device="cuda")
mlp_norm = torch.ones(D, dtype=torch.float32, device="cuda")

def make_quant_linear(in_features, out_features):
    q_weight = torch.randint(-2000000000, 2000000000, (out_features, in_features // 8), dtype=torch.int32, device="cuda")
    scales = torch.ones((out_features, 1), dtype=torch.float32, device="cuda")
    return q_weight, scales

q_w, q_s = make_quant_linear(D, D)
k_w, k_s = make_quant_linear(D, D)
v_w, v_s = make_quant_linear(D, D)
o_w, o_s = make_quant_linear(D, D)

gate_w, gate_s = make_quant_linear(D, INTERMEDIATE_DIM)
up_w, up_s = make_quant_linear(D, INTERMEDIATE_DIM)
down_w, down_s = make_quant_linear(INTERMEDIATE_DIM, D)

block.load_weights(
    attn_norm, 
    q_w, q_s, k_w, k_s, v_w, v_s, o_w, o_s, 
    mlp_norm, 
    gate_w, gate_s, up_w, up_s, down_w, down_s
)

hidden_states = torch.randn(D, dtype=torch.float32, device="cuda")

for _ in range(warmup_steps):
    block.forward(hidden_states)
torch.cuda.synchronize()

print(f"Running benchmark for {NUM_ITERATIONS} tokens...")
inference_start.record()
for i in range(NUM_ITERATIONS): 
    block.forward(hidden_states)
inference_end.record()
torch.cuda.synchronize()
    
total_time_ms = inference_start.elapsed_time(inference_end)
tpot_ms = total_time_ms / NUM_ITERATIONS

print("=== C++ TRANSFORMER ENGINE PERFORMANCE ===")
print(f"Total time: {total_time_ms:.2f} ms")
print(f"Latency (TPOT): {tpot_ms:.4f} ms/token")
print("==========================================")