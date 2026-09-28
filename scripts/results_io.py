"""Writes benchmark results to results/<name>.json together with the environment that produced them,
so every number in the README can be traced back to a run (GPU, driver, versions, git commit)."""
from __future__ import annotations

import datetime as _dt
import json
import platform
import subprocess
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def environment_info() -> dict:
    info: dict = {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _run(["git", "rev-parse", "--short", "HEAD"]),
        # results/ itself is excluded: otherwise the JSON written by earlier benchmarks marks every later run dirty
        "git_dirty": bool(_run(["git", "status", "--porcelain", "--", ".", ":(exclude)results"])),
        "nvidia_driver": _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["compute_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    except ImportError:
        pass
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    return info


def save_results(name: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump({"environment": environment_info(), **payload}, f, indent=2)
    print(f"\nResults written to {path.relative_to(RESULTS_DIR.parent)}")
    return path


def percentiles(samples_ms: list[float]) -> dict:
    import numpy as np

    arr = np.asarray(samples_ms, dtype=float)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
        "n": int(arr.size),
    }
