from __future__ import annotations

from collections.abc import Sequence

import torch
from transformers import AutoModelForCausalLM

import codealign_runtime_transformer as cuda_engine
from scripts.inference_config import KERNEL_RMS_EPS, KERNEL_ROPE_THETA, MAX_SEQ_LEN, MODEL, QUANTIZE_LM_HEAD
from scripts.quantization import QuantizedLinearINT4, quantize_to_int4
from scripts.roofline import Dims

def _rope_theta(cfg) -> float | None:
    theta = getattr(cfg, "rope_theta", None)                      
    if theta is None:
        theta = (getattr(cfg, "rope_parameters", None) or {}).get("rope_theta") 
    return theta

def check_supported_config(cfg) -> None:
    problems = []
    
    if cfg.model_type != "qwen2":
        problems.append(f"model_type={cfg.model_type} (expected qwen2)")
    if cfg.hidden_act != "silu":
        problems.append(f"hidden_act={cfg.hidden_act} (SwiGLU kernel is SiLU)")
        
    theta = _rope_theta(cfg)
    if theta is None or abs(theta - KERNEL_ROPE_THETA) > 1e-3:
        problems.append(f"rope_theta={theta} (src/ops/rope.cu uses {KERNEL_ROPE_THETA:g})")
    rope = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
    rope_type = rope.get("rope_type", rope.get("type", "default"))
    
    if rope_type != "default":
        problems.append(f"RoPE scaling ({rope_type}) is not supported")
    if abs(cfg.rms_norm_eps - KERNEL_RMS_EPS) > 1e-12:
        problems.append(f"rms_norm_eps={cfg.rms_norm_eps} (src/ops/rmsnorm.cu uses {KERNEL_RMS_EPS:g})")
    if cfg.hidden_size > 1024:
        problems.append(f"hidden_size={cfg.hidden_size} > 1024 (single-block RMSNorm)")
    if getattr(cfg, "use_sliding_window", False):
        problems.append("sliding-window attention is not supported")
    if problems:
        raise ValueError("Unsupported model config for the CUDA engine: " + "; ".join(problems))

class CodeAlignEngine:
    def __init__(self, hf_model, max_seq_len: int = MAX_SEQ_LEN, quantize_lm_head: bool = QUANTIZE_LM_HEAD, device: str | torch.device = "cuda"):
        cfg = hf_model.config
        check_supported_config(cfg)
        self.device = torch.device(device)
        self.dims = Dims.from_hf_config(cfg)
        self.max_seq_len = max_seq_len
        self.quantize_lm_head = quantize_lm_head
        self.eps = cfg.rms_norm_eps
        self.max_tokens_per_forward = int(cuda_engine.MAX_TOKENS_PER_FORWARD)
        d = self.dims

        def int4(linear):
            return quantize_to_int4(linear.weight.detach().to(self.device))

        def fp32(t):
            return t.detach().to(self.device, torch.float32).contiguous()

        self.blocks = []
        for layer in hf_model.model.layers:
            attn, mlp = layer.self_attn, layer.mlp
            q_w, q_s = int4(attn.q_proj)
            k_w, k_s = int4(attn.k_proj)
            v_w, v_s = int4(attn.v_proj)
            o_w, o_s = int4(attn.o_proj)
            g_w, g_s = int4(mlp.gate_proj)
            u_w, u_s = int4(mlp.up_proj)
            dn_w, dn_s = int4(mlp.down_proj)
            block = cuda_engine.QwenBlock(d.hidden, d.intermediate, d.num_heads, d.num_kv_heads, d.head_dim, max_seq_len)

            block.load_weights(
                attn_norm=fp32(layer.input_layernorm.weight),
                q_weight=q_w, q_scales=q_s, q_bias=fp32(attn.q_proj.bias) if attn.q_proj.bias is not None else None,
                k_weight=k_w, k_scales=k_s, k_bias=fp32(attn.k_proj.bias) if attn.k_proj.bias is not None else None,
                v_weight=v_w, v_scales=v_s, v_bias=fp32(attn.v_proj.bias) if attn.v_proj.bias is not None else None,
                o_weight=o_w, o_scales=o_s,
                mlp_norm=fp32(layer.post_attention_layernorm.weight),
                gate_weight=g_w, gate_scales=g_s,
                up_weight=u_w, up_scales=u_s,
                down_weight=dn_w, down_scales=dn_s,
            )
            self.blocks.append(block)

        self.embed_weight = hf_model.model.embed_tokens.weight.detach().to(self.device)  # lookup only: stays bf16
        self.final_norm_weight = fp32(hf_model.model.norm.weight)
        lm_head_weight = hf_model.lm_head.weight.detach().to(self.device)
        if quantize_lm_head:
            self.lm_head = QuantizedLinearINT4.from_weight(lm_head_weight)
            self.lm_head_weight = None
        else:
            self.lm_head = None
            self.lm_head_weight = lm_head_weight.to(torch.float32)

    @classmethod
    def from_pretrained(cls, model_name: str = MODEL, **kwargs) -> "CodeAlignEngine":
        hf_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
        engine = cls(hf_model, **kwargs)
        del hf_model
        return engine

    @property
    def seq_len(self) -> int:
        return self.blocks[0].seq_len()

    def reset(self) -> None:
        for block in self.blocks:
            block.reset_kv_cache()

    def rollback(self, num_tokens: int) -> None:
        if num_tokens:
            for block in self.blocks:
                block.rollback_kv_cache(num_tokens)

    def set_seq_len(self, seq_len: int) -> None:
        for block in self.blocks:
            block.set_seq_len(seq_len)

    def cuda_buffer_bytes(self) -> int:
        num_splits, flash_chunk, fp32 = 4, 256, 4
        d, t = self.dims, self.max_tokens_per_forward
        kv_cache = 2 * d.kv_dim * self.max_seq_len
        activations = t * (4 * d.hidden + 2 * d.q_dim + 2 * d.kv_dim + 3 * d.intermediate)
        max_chunks = -(-self.max_seq_len // flash_chunk)
        scratch = max(max_chunks * d.head_dim, num_splits * t * max(d.hidden, d.intermediate, d.q_dim, d.kv_dim))
        return len(self.blocks) * (kv_cache + activations + scratch + max_chunks) * fp32

    def embed(self, token_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
        ids = torch.as_tensor(token_ids, dtype=torch.long, device=self.device).view(-1)
        return self.embed_weight.index_select(0, ids).to(torch.float32).contiguous()

    def final_norm(self, h: torch.Tensor) -> torch.Tensor:
        return (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps) * self.final_norm_weight).contiguous()

    def lm_logits(self, h: torch.Tensor) -> torch.Tensor:
        if self.lm_head is not None:
            return self.lm_head(h).contiguous()
        return (h @ self.lm_head_weight.T).contiguous()

    @torch.inference_mode()
    def forward(self, token_ids: Sequence[int] | torch.Tensor, logits: str = "all") -> torch.Tensor | None:
        n = len(token_ids)
        if not 1 <= n <= self.max_tokens_per_forward:
            raise ValueError(f"forward takes 1..{self.max_tokens_per_forward} tokens, got {n}")
        h = self.embed(token_ids)
        for block in self.blocks:
            block.forward(h) 
        if logits == "none":
            return None
        if logits == "last":
            h = h[-1:]
        elif logits != "all":
            raise ValueError(f"logits must be 'all', 'last' or 'none', got {logits!r}")
        return self.lm_logits(self.final_norm(h))

    @torch.inference_mode()
    def prefill(self, token_ids: Sequence[int], logits: str = "last") -> torch.Tensor | None:
        token_ids = list(token_ids)
        if not token_ids:
            return None
        step = self.max_tokens_per_forward
        out = None
        for start in range(0, len(token_ids), step):
            chunk = token_ids[start:start + step]
            is_last = start + step >= len(token_ids)
            out = self.forward(chunk, logits=logits if is_last else "none")
        return out

    def argmax(self, logits: torch.Tensor) -> list[int]:
        return cuda_engine.fast_argmax(logits.contiguous()).tolist()

    @torch.inference_mode()
    def generate(self, prompt_ids: Sequence[int], max_new_tokens: int, eos_token_ids: Sequence[int] = ()) -> list[int]:
        self.reset()
        if len(prompt_ids) + max_new_tokens > self.max_seq_len:
            raise ValueError("prompt + max_new_tokens exceeds max_seq_len")
        eos = set(eos_token_ids)
        token = self.argmax(self.prefill(prompt_ids, logits="last"))[0]
        new_tokens = [token]
        while len(new_tokens) < max_new_tokens and token not in eos:
            token = self.argmax(self.forward([token], logits="last"))[0]
            new_tokens.append(token)
        return new_tokens