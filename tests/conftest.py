from __future__ import annotations

import importlib
import os

import pytest
import torch

DEVICE = os.environ.get("CODEALIGN_TEST_DEVICE", "cuda")

# Random-weight Qwen2 with kernel-compatible dimensions (hidden and intermediate multiples of 128, GQA 4/2)
TINY_QWEN2 = dict(hidden_size=256, intermediate_size=512, num_attention_heads=4, num_key_value_heads=2,
                  num_hidden_layers=2, vocab_size=1024, rope_theta=1_000_000.0, rms_norm_eps=1e-6,
                  tie_word_embeddings=True, max_position_embeddings=4096, initializer_range=0.1)

EXTENSIONS = ("codealign_runtime_kernels", "codealign_runtime_gemm", "codealign_runtime_transformer")

def _extension_status() -> tuple[bool, str]:
    for name in EXTENSIONS:
        try:
            importlib.import_module(name)
        except (ImportError, OSError) as e:
            return False, f"{name}: {type(e).__name__}: {e}"
    return True, "all 3 importable"

def pytest_report_header(config):
    cuda = torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(0) if cuda else "none"
    return [f"codealign: torch {torch.__version__} (built for CUDA {torch.version.cuda}), "
            f"cuda available: {cuda}, GPU: {gpu}, test device: {DEVICE}",
            f"codealign: CUDA extensions: {_extension_status()[1]}"]

def pytest_collection_modifyitems(config, items):
    reason = None
    if DEVICE == "cuda" and not torch.cuda.is_available():
        reason = (f"torch {torch.__version__} (CUDA {torch.version.cuda}) sees no CUDA GPU: "
                  "check nvidia-smi and that the driver supports this CUDA version")
    else:
        ok, detail = _extension_status()
        if not ok:
            reason = f"CUDA extensions not importable ({detail}); run ./build.sh from the repo root"
    if reason:
        skip = pytest.mark.skip(reason=reason)
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip)

@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device(DEVICE)

def _tiny_qwen(qk_bias_std: float):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    model = Qwen2ForCausalLM(Qwen2Config(**TINY_QWEN2)).eval()
    with torch.no_grad(): 
        for layer in model.model.layers:
            layer.self_attn.q_proj.bias.normal_(0.0, qk_bias_std)
            layer.self_attn.k_proj.bias.normal_(0.0, qk_bias_std)
            layer.self_attn.v_proj.bias.normal_(0.0, 0.5)
            layer.input_layernorm.weight.normal_(1.0, 0.2)
            layer.post_attention_layernorm.weight.normal_(1.0, 0.2)
        model.model.norm.weight.normal_(1.0, 0.2)
    return model

@pytest.fixture(scope="session")
def tiny_qwen():
    return _tiny_qwen(qk_bias_std=0.5)

@pytest.fixture(scope="session")
def tiny_qwen_large_bias():
    return _tiny_qwen(qk_bias_std=20.0)

@pytest.fixture(scope="session")
def real_qwen():
    from transformers import AutoModelForCausalLM

    from scripts.inference_config import MODEL

    return AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()

@pytest.fixture(params=["tiny", "tiny_large_bias", pytest.param("real", marks=pytest.mark.real_model)])
def hf_model(request):
    return request.getfixturevalue({"tiny": "tiny_qwen", "tiny_large_bias": "tiny_qwen_large_bias", "real": "real_qwen"}[request.param])