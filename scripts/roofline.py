from __future__ import annotations

from dataclasses import dataclass

from scripts import inference_config as C

GPU_BANDWIDTH_GBPS = 896.0   # RTX 5070 Ti, GDDR7 256-bit, spec sheet
GROUP_SIZE = C.GROUP_SIZE
FP32 = 4

@dataclass(frozen=True)
class Dims:
    hidden: int
    intermediate: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    num_layers: int
    vocab_size: int

    @property
    def q_dim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    @classmethod
    def from_hf_config(cls, cfg) -> "Dims":
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        return cls(cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads, cfg.num_key_value_heads,
                   head_dim, cfg.num_hidden_layers, cfg.vocab_size)

QWEN25_CODER_05B = Dims(C.HIDDEN, C.INTERMEDIATE, C.NUM_HEADS, C.NUM_KV_HEADS, C.HEAD_DIM, C.NUM_LAYERS, C.VOCAB_SIZE)

def int4_bytes(numel: int, group_size: int = GROUP_SIZE) -> int:
    return numel // 2 + (numel // group_size) * FP32

def block_linear_shapes(d: Dims) -> dict[str, tuple[int, int]]:
    return {
        "q_proj": (d.q_dim, d.hidden), "k_proj": (d.kv_dim, d.hidden), "v_proj": (d.kv_dim, d.hidden),
        "o_proj": (d.hidden, d.q_dim),
        "gate_proj": (d.intermediate, d.hidden), "up_proj": (d.intermediate, d.hidden),
        "down_proj": (d.hidden, d.intermediate),
    }

def block_params(d: Dims) -> int:
    return sum(o * i for o, i in block_linear_shapes(d).values())

def block_bytes_int4(d: Dims) -> int:
    weights = sum(int4_bytes(o * i) for o, i in block_linear_shapes(d).values())
    bias = (d.q_dim + 2 * d.kv_dim) * FP32
    norms = 2 * d.hidden * FP32
    return weights + bias + norms

def kv_cache_bytes(d: Dims, context_len: int, bytes_per_elem: int = FP32) -> int:
    return 2 * d.kv_dim * context_len * bytes_per_elem

def lm_head_bytes(d: Dims, quantized: bool) -> int:
    numel = d.vocab_size * d.hidden
    return int4_bytes(numel) if quantized else numel * 2  

def model_bytes_int4(d: Dims, quantize_lm_head: bool = True) -> int:
    embedding_row = d.hidden * 2
    return d.num_layers * block_bytes_int4(d) + d.hidden * FP32 + lm_head_bytes(d, quantize_lm_head) + embedding_row

def ceiling_ms(num_bytes: float, bandwidth_gbps: float = GPU_BANDWIDTH_GBPS) -> float:
    return num_bytes / (bandwidth_gbps * 1e9) * 1e3

def decode_ceiling_ms(d: Dims, context_len: int, scope: str = "model", quantize_lm_head: bool = True) -> float:
    if scope == "block":
        return ceiling_ms(block_bytes_int4(d) + kv_cache_bytes(d, context_len))
    return ceiling_ms(model_bytes_int4(d, quantize_lm_head) + d.num_layers * kv_cache_bytes(d, context_len))

if __name__ == "__main__":
    d = QWEN25_CODER_05B
    print(f"block: {block_params(d) / 1e6:.2f} M params, {block_bytes_int4(d) / 1e6:.2f} MB INT4 "
          f"-> {ceiling_ms(block_bytes_int4(d)) * 1e3:.2f} us")
    for q in (True, False):
        b = model_bytes_int4(d, q)
        print(f"model INT4 (lm_head {'INT4' if q else 'bf16'}): {b / 1e6:.0f} MB -> {ceiling_ms(b):.3f} ms/token")
    for s in C.CONTEXT_LENGTHS:
        print(f"S={s:5d}: block {decode_ceiling_ms(d, s, 'block') * 1e3:6.2f} us | model {decode_ceiling_ms(d, s):.3f} ms")