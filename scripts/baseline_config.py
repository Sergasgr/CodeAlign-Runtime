MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

PROMPT = """def calculate_fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    result = [0, 1]
    for i in range(2, n):
        result.append(result[i-1] + result[i-2])
    return result
"""

NUM_ITERATIONS = 100
MAX_TOKENS = 50

# RTX 5070 Ti GDDR7 — theoretical memory bandwidth from spec sheet
GPU_BANDWIDTH_GBPS = 896.0
# Model parameters (0.5B = 0.5)
MODEL_PARAMS_BILLIONS = 0.5 
# Calculate the model weight (2 bytes per parameter in fp16/bf16)
MODEL_BYTES = MODEL_PARAMS_BILLIONS * 1e9 * 2
# Physical limit: minimum possible ms/token (Bytes / Bandwidth)s
THEORETICAL_MIN_TPOT_MS = (MODEL_BYTES / (GPU_BANDWIDTH_GBPS * 1e9)) * 1000