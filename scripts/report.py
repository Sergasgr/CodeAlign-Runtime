from __future__ import annotations

import argparse
import json

from scripts.results_io import RESULTS_DIR

def load(name: str) -> dict | None:
    path = RESULTS_DIR / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None

def fmt(v, digits=3) -> str:
    return "—" if v is None else f"{v:.{digits}f}"

def table(headers: list[str], rows: list[list[str]], align: str | None = None) -> str:
    align = align or "l" + "r" * (len(headers) - 1)
    sep = ["---:" if a == "r" else ":---:" if a == "c" else "---" for a in align]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)

def env_line(r: dict) -> str:
    e = r.get("environment", {})
    return (f"_{e.get('gpu', e.get('name', '?'))}, driver {e.get('nvidia_driver', '?')}, torch {e.get('torch', '?')}, "
            f"commit {e.get('git_commit', '?')}{' (dirty)' if e.get('git_dirty') else ''}, {e.get('timestamp_utc', '')}_")

def level0() -> str | None:
    r = load("level0_baseline")
    if not r:
        return None
    t, p = r["ttft"], r["tpot"]
    rows = [["TTFT (prefill, %d tokens)" % r["prompt_tokens"], fmt(t["p50_ms"], 2), fmt(t["p90_ms"], 2), fmt(t["p99_ms"], 2)],
            ["TPOT (decode)", fmt(p["p50_ms"], 2), fmt(p["p90_ms"], 2), fmt(p["p99_ms"], 2)]]
    return "\n".join([
        table(["Metric (ms)", "p50", "p90", "p99"], rows),
        "",
        f"Bandwidth ceiling: {r['weights_bytes_per_token'] / 1e6:.0f} MB/token -> **{r['tpot_ceiling_ms']:.3f} ms**; "
        f"TPOT p50 reaches **{r['pct_of_ceiling_p50']:.1f}%** of it. Peak VRAM: {r['peak_vram_mb']:.0f} MB.",
        env_line(r)])

def kernel_table(name: str, title_shape: str) -> str | None:
    r = load(name)
    if not r:
        return None
    by = {(t["kernel"], t["cache"]): t for t in r["timings"]}
    kernels = list(dict.fromkeys(t["kernel"] for t in r["timings"]))
    rows = []
    for k in kernels:
        hot, cold = by.get((k, "hot")), by.get((k, "cold"))
        rows.append([k, fmt(hot and hot["p50_ms"], 4), fmt(hot and hot["gbps_p50"], 1),
                     fmt(cold and cold["p50_ms"], 4), fmt(cold and cold["gbps_p50"], 1),
                     fmt(cold and cold["pct_of_spec_bw"], 1)])
    passed = sum(c["pass"] for c in r["validation"])
    return "\n".join([
        f"{title_shape}: {r['shape']}",
        "",
        table(["Kernel", "L2-hot p50 (ms)", "L2-hot GB/s", "L2-cold p50 (ms)", "L2-cold GB/s", "% of 896 GB/s (cold)"], rows),
        "",
        f"Validation: {passed}/{len(r['validation'])} kernel x shape checks passed (random data, CPU reference).",
        f"_{r['device'].get('name', '?')}_"])

def level4() -> str | None:
    r = load("level4_flash_decoding")
    if not r or not r.get("benchmark"):
        return None
    worst = max(v["max_abs_err"] for v in r["validation"])
    rows = [[f"{b['S']:,}", fmt(b["pytorch"]["p50_ms"], 4), fmt(b["custom"]["p50_ms"], 4),
             f"{b['ratio_pytorch_over_custom']:.2f}x", fmt(b["ceiling_ms"], 4)] for b in r["benchmark"]]
    return "\n".join([
        f"Validation: {sum(v['pass'] for v in r['validation'])}/{len(r['validation'])} cases, worst max|err| = {worst:.1e} "
        f"(tolerance {r['tolerance']:g}), head_dim={r['head_dim']}.",
        "",
        table(["Seq length", "PyTorch p50 (ms)", "Custom p50 (ms)", "PyTorch / Custom", "KV-read ceiling (ms)"], rows),
        env_line(r)])

def level5() -> str | None:
    r = load("level5_engine")
    if not r:
        return None
    out = []
    for key, label in (("block", "Single block (layer 0)"), ("model_decode", "Full model (24 layers + lm_head + argmax)")):
        rows = [[f"{x['context']:,}", fmt(x["p50_ms"], 4), fmt(x["p99_ms"], 4), fmt(x["ceiling_ms"], 4),
                 f"{x['pct_of_ceiling_p50']:.2f}%"] + ([f"{x['tokens_per_s_p50']:.0f}"] if key == "model_decode" else [])
                for x in r[key]]
        headers = ["Context S", "p50 (ms)", "p99 (ms)", "Ceiling (ms)", "% of ceiling"] + (["tok/s (p50)"] if key == "model_decode" else [])
        out += [f"**{label}**", "", table(headers, rows), ""]
    if r.get("e2e"):
        e = r["e2e"]
        out.append(f"End-to-end: prompt {e['prompt_tokens']} tokens, TTFT {e['ttft_ms']:.1f} ms (chunked prefill), "
                   f"TPOT p50 {fmt(e['tpot'].get('p50_ms'), 3)} ms over {e['generated_tokens']} tokens. "
                   + _engine_memory_text(r))
    out.append(env_line(r))
    return "\n".join(out)

def _engine_vram_mb(r: dict) -> float:
    return r["memory"]["resident_mb"] if "memory" in r else r["peak_vram_mb"]

def _engine_memory_text(r: dict) -> str:
    if "memory" not in r:
        return f"Peak VRAM {r['peak_vram_mb']:.0f} MB (torch allocations only)."
    m = r["memory"]
    return (f"Resident VRAM {m['resident_mb']:.0f} MB (tensors {m['torch_tensors_mb']:.0f} + C++ buffers "
            f"{m['cuda_buffers_mb']:.0f}); decode peak {m['decode_peak_mb']:.0f} MB; load-time peak "
            f"{m['load_peak_torch_mb']:.0f} MB (INT4 quantization temporaries).")

def level6() -> str | None:
    r = load("level6_speculative")
    if not r:
        return None
    rows = [[p["prompt"], f"{p['greedy']['tokens_per_second']:.1f}", f"{p['pld']['tokens_per_second']:.1f}",
             f"{p['speedup']:.2f}x", f"{100 * p['pld']['acceptance_rate']:.1f}%", f"{p['pld']['tokens_per_forward']:.2f}",
             "yes" if p["identical_to_greedy"] else f"no (token {p['first_divergence']})"] for p in r["prompts"]]
    return "\n".join([
        table(["Prompt", "Greedy tok/s", "PLD tok/s", "Speedup", "Acceptance", "Tokens/forward", "Identical output"], rows),
        "",
        f"n-gram={r['ngram']}, max draft={r['max_draft']}, {r['max_new_tokens']} new tokens per prompt. "
        f"Overall speedup: **{r['overall_speedup']:.2f}x**.",
        env_line(r)])

def quality() -> str | None:
    bf16, int4 = load("humaneval_bf16"), load("humaneval_int4")
    if not (bf16 or int4):
        return None
    rows = []
    if bf16:
        rows.append(["bf16 (reference)", fmt(bf16["pass@1"], 4)])
    if int4:
        rows.append([f"INT4 g=128, {int4['int4_layers']} layers (lm_head {'INT4' if int4['quantize_lm_head'] else 'bf16'})",
                     fmt(int4["pass@1"], 4)])
    if bf16 and int4:
        rows.append(["Δ", f"{int4['pass@1'] - bf16['pass@1']:+.4f}"])
    return table(["Model", "HumanEval pass@1 (greedy)"], rows)

def llama_cpp() -> str | None:
    r = load("llama_cpp")
    if not r:
        return None
    rows = []
    for variant, v in r["variants"].items():
        for d in v["decode"]:
            rows.append([variant, f"{v['file_bytes'] / 1e6:.0f}", f"{d['n_depth']:,}", f"{d['tokens_per_s']:.1f}",
                         fmt(d["tpot_ms"], 3)])
    return table(["GGUF", "File (MB)", "Context depth", "tok/s", "TPOT (ms)"], rows)

def summary() -> str:
    rows = []
    l0, l5, l6, ll = load("level0_baseline"), load("level5_engine"), load("level6_speculative"), load("llama_cpp")
    bf16, int4 = load("humaneval_bf16"), load("humaneval_int4")
    if l0:
        rows.append(["PyTorch eager (bf16)", fmt(l0["ttft"]["p50_ms"], 2), fmt(l0["tpot"]["p50_ms"], 3),
                     f"{l0['pct_of_ceiling_p50']:.1f}%", f"{l0['peak_vram_mb']:.0f}", fmt(bf16 and bf16["pass@1"], 3)])
    if l5:
        m = min(l5["model_decode"], key=lambda x: abs(x["context"] - 512))
        rows.append([f"C++/CUDA engine, INT4 (S={m['context']})", fmt(l5.get("e2e", {}).get("ttft_ms"), 1),
                     fmt(m["p50_ms"], 3), f"{m['pct_of_ceiling_p50']:.1f}%", f"{_engine_vram_mb(l5):.0f}",
                     fmt(int4 and int4["pass@1"], 3)])
    if l6:
        lossless = all(p["identical_to_greedy"] for p in l6["prompts"])
        rows.append(["+ Prompt Lookup Decoding", "—", "—", "—", "—",
                     f"x{l6['overall_speedup']:.2f} tok/s" + (", identical output" if lossless else ", output differs (see Level 6)")])
    if ll:
        for variant in ("Q4_0", "Q4_K_M"):
            if variant in ll["variants"]:
                d = min(ll["variants"][variant]["decode"], key=lambda x: abs(x["n_depth"] - 512))
                rows.append([f"llama.cpp {variant} (depth {d['n_depth']})", "—", fmt(d["tpot_ms"], 3), "—", "—", "—"])
    return table(["System", "TTFT p50 (ms)", "TPOT p50 (ms)", "% BW ceiling", "VRAM (MB)", "HumanEval pass@1"], rows)

SECTIONS = [
    ("Level 0 — PyTorch baseline", level0),
    ("GEMV kernels (Levels 1-3)", lambda: kernel_table("gemv_benchmark", "Shape")),
    ("GEMM kernels", lambda: kernel_table("gemm_benchmark", "Shape")),
    ("Level 4 — Flash-Decoding", level4),
    ("Level 5 — Engine", level5),
    ("Level 6 — Prompt Lookup Decoding", level6),
    ("Quantization vs. quality", quality),
    ("llama.cpp reference", llama_cpp),
]

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if not args.summary:
        for title, fn in SECTIONS:
            body = fn()
            print(f"### {title}\n\n{body if body else '_no results yet_'}\n")
    print("### Summary\n\n" + summary())

if __name__ == "__main__":
    main()