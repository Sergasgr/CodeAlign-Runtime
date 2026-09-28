from scripts.inference_config import MODEL
from scripts.roofline import GPU_BANDWIDTH_GBPS

__all__ = ["MODEL", "GPU_BANDWIDTH_GBPS", "PROMPT", "NUM_ITERATIONS", "WARMUP_ITERATIONS", "MAX_TOKENS"]

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
WARMUP_ITERATIONS = 10
MAX_TOKENS = 50