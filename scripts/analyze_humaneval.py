from __future__ import annotations

import argparse
import gzip
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from scripts.results_io import RESULTS_DIR, save_results

HUMANEVAL_DATASET = "openai/openai_humaneval"
TIMEOUT_S = 3.0
WHITESPACE_RUN = 50                  
REFERENCE_TAG = "bf16"
TAG_ORDER = ["bf16", "int4", "int4_dequant", "int4_bf16head", "int4_dequant_bf16head"]

def load_problems(path: str | None = None) -> list[dict]:
    if path:
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as f:
            return [json.loads(line) for line in f if line.strip()]
    from datasets import load_dataset

    return [dict(row) for row in load_dataset(HUMANEVAL_DATASET, split="test")]

def generations_path(tag: str) -> Path:
    return RESULTS_DIR / f"humaneval_generations_{tag}_humaneval.json"

def available_tags() -> list[str]:
    found = [p.name[len("humaneval_generations_"):-len("_humaneval.json")]
             for p in RESULTS_DIR.glob("humaneval_generations_*_humaneval.json")]
    found = [t for t in found if "_limit" not in t]
    return sorted(found, key=lambda t: (TAG_ORDER.index(t) if t in TAG_ORDER else len(TAG_ORDER), t))

def load_generations(tag: str) -> list[str]:
    with open(generations_path(tag)) as f:
        return [samples[0] for samples in json.load(f)]

def build_program(code: str, problem: dict) -> str:
    return code + "\n" + problem["test"] + "\n" + f"check({problem['entry_point']})"

def completion_of(generation: str, prompt: str) -> str:
    head = prompt.strip()
    return generation[len(head):] if generation.startswith(head) else generation

def body_only(generation: str, prompt: str) -> str:
    head = prompt.strip()
    if not generation.startswith(head):
        return generation
    kept = []
    for line in generation[len(head):].split("\n"):
        if line.strip() and not line[0].isspace():
            break                                  
        kept.append(line)
    return head + "\n".join(kept).rstrip() + "\n"

def has_whitespace_run(text: str, min_len: int = WHITESPACE_RUN) -> bool:
    return re.search(r"\s{%d,}" % min_len, text) is not None

def classify_failure(stderr: str) -> str:
    lines = [line for line in stderr.strip().splitlines() if line.strip()]
    if not lines:
        return "unknown"
    match = re.match(r"([A-Za-z_][\w.]*)(?::|$)", lines[-1].strip())
    return match.group(1).split(".")[-1] if match else "unknown"

def run_program(program: str, timeout_s: float = TIMEOUT_S) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "program.py"
        path.write_text(program)
        try:
            proc = subprocess.run([sys.executable, "-I", str(path)], cwd=tmp, capture_output=True, text=True,
                                  timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return "timeout"
    return "passed" if proc.returncode == 0 else "failed: " + classify_failure(proc.stderr)

def analyze_tag(tag: str, problems: list[dict], timeout_s: float) -> dict:
    generations = load_generations(tag)
    if len(generations) != len(problems):
        raise ValueError(f"{tag}: {len(generations)} generations for {len(problems)} problems")
    per_problem = []
    for i, (gen, prob) in enumerate(zip(generations, problems)):
        status = run_program(build_program(gen, prob), timeout_s)
        body_status = status if body_only(gen, prob["prompt"]) == gen else \
            run_program(build_program(body_only(gen, prob["prompt"]), prob), timeout_s)
        per_problem.append({"task_id": prob["task_id"], "status": status, "body_only_status": body_status,
                            "whitespace_run": has_whitespace_run(completion_of(gen, prob["prompt"]))})
        print(f"\r  {tag}: {i + 1}/{len(problems)}", end="", flush=True)
    print()
    solved = sum(p["status"] == "passed" for p in per_problem)
    summary = {
        "solved": solved,
        "pass@1": solved / len(problems),
        "body_only_solved": sum(p["body_only_status"] == "passed" for p in per_problem),
        "whitespace_runs": sum(p["whitespace_run"] for p in per_problem),
        "failures": dict(Counter(p["status"].removeprefix("failed: ") for p in per_problem
                                 if p["status"] != "passed").most_common()),
        "generations_file": str(generations_path(tag).relative_to(RESULTS_DIR.parent)),
    }
    bigcode = RESULTS_DIR / f"humaneval_{tag}.json"
    if bigcode.exists():
        reported = json.loads(bigcode.read_text())["pass@1"]
        summary["bigcode_pass@1"] = reported
        summary["matches_bigcode"] = round(reported * len(problems)) == solved
    return {"summary": summary, "per_problem": per_problem}


def overlap(ref: list[dict], other: list[dict]) -> dict:
    a = {p["task_id"] for p in ref if p["status"] == "passed"}
    b = {p["task_id"] for p in other if p["status"] == "passed"}
    order = [p["task_id"] for p in ref]
    return {"both": len(a & b), "only_reference": len(a - b), "only_other": len(b - a),
            "neither": len(order) - len(a | b),
            "only_reference_tasks": [t for t in order if t in a - b],
            "only_other_tasks": [t for t in order if t in b - a]}

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tags", nargs="+", default=None, help="default: every full run in results/")
    parser.add_argument("--problems", default=None, help="local HumanEval .jsonl(.gz) instead of the Hub dataset")
    parser.add_argument("--timeout", type=float, default=TIMEOUT_S, help="seconds per program")
    args = parser.parse_args()

    tags = args.tags or available_tags()
    if not tags:
        raise SystemExit(f"no generations in {RESULTS_DIR}: run scripts/evaluate_quality.py first")
    problems = load_problems(args.problems)
    print(f"{len(problems)} problems, tags: {', '.join(tags)} (timeout {args.timeout:g} s, serial)")

    runs = {tag: analyze_tag(tag, problems, args.timeout) for tag in tags}
    overlaps = {}
    if REFERENCE_TAG in runs:
        for tag in tags:
            if tag != REFERENCE_TAG:
                overlaps[f"{REFERENCE_TAG}_vs_{tag}"] = overlap(runs[REFERENCE_TAG]["per_problem"], runs[tag]["per_problem"])

    print(f"\n{'tag':<24} {'solved':>6} {'body only':>9} {'ws runs':>7} {'SyntaxError':>11} {'bigcode':>8}")
    for tag, r in runs.items():
        s = r["summary"]
        check = "—" if "matches_bigcode" not in s else ("ok" if s["matches_bigcode"] else "DIFFERS")
        print(f"{tag:<24} {s['solved']:>6} {s['body_only_solved']:>9} {s['whitespace_runs']:>7} "
              f"{s['failures'].get('SyntaxError', 0):>11} {check:>8}")
    for name, o in overlaps.items():
        print(f"{name}: both {o['both']}, only {REFERENCE_TAG} {o['only_reference']}, only other {o['only_other']}, "
              f"neither {o['neither']}")

    save_results("humaneval_analysis", {
        "dataset": args.problems or HUMANEVAL_DATASET, "num_problems": len(problems), "timeout_s": args.timeout,
        "whitespace_run_chars": WHITESPACE_RUN, "reference_tag": REFERENCE_TAG,
        "summary": {tag: r["summary"] for tag, r in runs.items()},
        "overlap": overlaps,
        "per_problem": {tag: r["per_problem"] for tag, r in runs.items()},
    })

if __name__ == "__main__":
    main()