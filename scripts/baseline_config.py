MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = ""
NUM_ITERATIONS = 100
MAX_TOKENS = 50

# TODO: Busca en la ficha técnica de tu GPU (ej. RTX 3060 -> 360 GB/s)
GPU_BANDWIDTH_GBPS = 360.0 
# Parámetros del modelo (0.5B = 0.5)
MODEL_PARAMS_BILLIONS = 0.5 
# Calculamos el peso del modelo (2 bytes por parámetro en fp16/bf16)
MODEL_BYTES = MODEL_PARAMS_BILLIONS * 1e9 * 2
# Límite físico: ms/token mínimo posible (Bytes / Ancho de banda)
THEORETICAL_MIN_TPOT_MS = (MODEL_BYTES / (GPU_BANDWIDTH_GBPS * 1e9)) * 1000