MODEL = "Qwen/Qwen2.5-Coder-0.5B"

# Architecture (Qwen2: GQA, bias on q/k/v, SwiGLU, tied embeddings)
HIDDEN = 896
INTERMEDIATE = 4864
NUM_HEADS = 14
NUM_KV_HEADS = 2
HEAD_DIM = 64
NUM_LAYERS = 24
VOCAB_SIZE = 151936

# Values hard-coded in the CUDA kernels (src/ops/rope.cu, src/ops/rmsnorm.cu)
KERNEL_ROPE_THETA = 1_000_000.0
KERNEL_RMS_EPS = 1e-6

GROUP_SIZE = 128          # INT4 per-group quantization
MAX_SEQ_LEN = 2048        # KV-cache capacity per layer
QUANTIZE_LM_HEAD = True   # the lm_head (tied to the embedding) also runs on the INT4 kernels

# Level 5 benchmark: decode latency is measured at these fixed context lengths
CONTEXT_LENGTHS = [128, 512, 1024, 2048]
BENCH_ITERATIONS = 200
WARMUP_ITERATIONS = 20

# Names kept for compatibility with earlier scripts
D = HIDDEN
INTERMEDIATE_DIM = INTERMEDIATE
NUM_ITERATIONS = BENCH_ITERATIONS
