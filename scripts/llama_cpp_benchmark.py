from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.inference_config import MODEL
from scripts.results_io import save_results

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
VARIANTS = ["F16", "Q4_0", "Q4_K_M"]

def find_binary(llama_dir: Path, name: str) -> Path:
    for candidate in (llama_dir / "build" / "bin" / name, llama_dir / "build" / "bin" / "Release" / name, llama_dir / name):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{name} not found under {llama_dir}/build/bin — build llama.cpp with -DGGML_CUDA=ON first")

def run(cmd: list[str]) -> subprocess.CompletedProcess:
    print("$ " + " ".join(map(str, cmd)))
    return subprocess.run(list(map(str, cmd)), capture_output=True, text=True)

def ensure_ggufs(llama_dir: Path) -> dict[str, Path]:
    from huggingface_hub import snapshot_download

    MODELS_DIR.mkdir(exist_ok=True)
    name = MODEL.split("/")[-1]
    f16 = MODELS_DIR / f"{name}-F16.gguf"
    if not f16.exists():
        hf_dir = snapshot_download(MODEL, local_dir=MODELS_DIR / "hf" / name)
        converter = llama_dir / "convert_hf_to_gguf.py"
        proc = run([sys.executable, converter, hf_dir, "--outtype", "f16", "--outfile", f16])
        if proc.returncode != 0:
            raise RuntimeError(f"GGUF conversion failed:\n{proc.stderr[-3000:]}\n"
                               "Hint: run with `uv run --with gguf --with sentencepiece ...`")
    paths = {"F16": f16}
    quantize = find_binary(llama_dir, "llama-quantize")
    for variant in VARIANTS[1:]:
        out = MODELS_DIR / f"{name}-{variant}.gguf"
        if not out.exists():
            proc = run([quantize, f16, out, variant])
            if proc.returncode != 0:
                raise RuntimeError(f"llama-quantize {variant} failed:\n{proc.stderr[-3000:]}")
        paths[variant] = out
    return paths

def llama_bench(bench: Path, gguf: Path, depths: list[int], n_gen: int, reps: int) -> list[dict]:
    base = [bench, "-m", gguf, "-p", "0", "-n", n_gen, "-ngl", "99", "-r", reps, "-o", "json"]
    proc = run(base + ["-d", ",".join(map(str, depths))])
    if proc.returncode != 0: 
        print("  (llama-bench without -d support: measuring at depth 0 only)")
        proc = run(base)
    if proc.returncode != 0:
        raise RuntimeError(f"llama-bench failed:\n{proc.stderr[-3000:]}")
    rows = []
    for r in json.loads(proc.stdout):
        tok_s = r["avg_ts"]
        rows.append({"n_depth": r.get("n_depth", 0), "n_gen": r["n_gen"], "tokens_per_s": tok_s,
                     "stddev_tokens_per_s": r.get("stddev_ts"), "tpot_ms": 1000.0 / tok_s,
                     "model_size_bytes": r.get("model_size"), "build_commit": r.get("build_commit"),
                     "gpu_info": r.get("gpu_info"), "flash_attn": r.get("flash_attn")})
    return rows

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llama-cpp-dir", default=os.environ.get("LLAMA_CPP_DIR"))
    parser.add_argument("--depths", type=int, nargs="+", default=[128, 512, 1024, 2048])
    parser.add_argument("--n-gen", type=int, default=128)
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()
    if not args.llama_cpp_dir:
        parser.error("set LLAMA_CPP_DIR or pass --llama-cpp-dir")
    llama_dir = Path(args.llama_cpp_dir).expanduser().resolve()

    ggufs = ensure_ggufs(llama_dir)
    bench = find_binary(llama_dir, "llama-bench")
    results = {"model": MODEL, "n_gen": args.n_gen, "reps": args.reps, "variants": {}}
    for variant, path in ggufs.items():
        print(f"\n=== {variant}: {path.name} ({path.stat().st_size / 1e6:.0f} MB) ===")
        rows = llama_bench(bench, path, args.depths, args.n_gen, args.reps)
        for r in rows:
            print(f"  depth {r['n_depth']:5d}: {r['tokens_per_s']:8.1f} tok/s  ->  TPOT {r['tpot_ms']:.3f} ms")
        results["variants"][variant] = {"gguf": path.name, "file_bytes": path.stat().st_size, "decode": rows}
    save_results("llama_cpp", results)

if __name__ == "__main__":
    main()