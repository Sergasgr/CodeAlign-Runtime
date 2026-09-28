from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from bigcode_eval.arguments import EvalArguments
from bigcode_eval.evaluator import Evaluator
import bigcode_eval.tasks as tasks

from scripts.inference_config import MODEL, QUANTIZE_LM_HEAD
from scripts.quantization import replace_linear_layers
from scripts.results_io import RESULTS_DIR, save_results

def bigcode_args(limit: int | None, max_length: int, generations_path: Path, metrics_path: Path) -> argparse.Namespace:
    args = dataclasses.asdict(EvalArguments())
    args.update(
        model=MODEL,
        modeltype="causal",
        peft_model=None,
        revision=None,
        use_auth_token=False,
        trust_remote_code=False,
        tasks="humaneval",
        instruction_tokens=None,
        batch_size=1,
        max_length_generation=max_length,
        precision="bf16",
        load_in_8bit=False,
        load_in_4bit=False,
        left_padding=False,
        limit=limit,
        limit_start=0,
        save_every_k_tasks=-1,
        postprocess=True,
        allow_code_execution=True,
        generation_only=False,
        load_generations_path=None,
        load_data_path=None,
        metric_output_path=str(metrics_path),
        save_generations=True,
        load_generations_intermediate_paths=None,
        save_generations_path=str(generations_path),
        save_references=False,
        save_references_path="references.json",
        prompt="prompt",
        max_memory_per_gpu=None,
        check_references=False,
    )
    args.update(do_sample=False, n_samples=1, temperature=None, top_p=None, top_k=None)  # greedy pass@1
    return argparse.Namespace(**args)

HUMANEVAL_DATASET = "openai/openai_humaneval"

def patch_humaneval_task(num_workers: int = 1) -> None:
    base = tasks.TASK_REGISTRY["humaneval"]
    if getattr(base, "_codealign_patched", False):
        return

    class PatchedHumanEval(base):
        _codealign_patched = True
        DATASET_PATH = HUMANEVAL_DATASET

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            if not hasattr(self, "dataset"):  
                raise RuntimeError(f"could not load {HUMANEVAL_DATASET} from the Hugging Face Hub (see the warning above)")
            self.num_workers = num_workers

    tasks.TASK_REGISTRY["humaneval"] = PatchedHumanEval

def run_evaluation(precision: str, limit: int | None, max_length: int, quantize_lm_head: bool) -> dict:
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Loading {MODEL} ({dtype})...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL, truncation_side="left", padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, device_map="cuda")
    model.eval()

    replaced = 0
    if precision == "int4":
        skip = () if quantize_lm_head else ("lm_head",)
        replaced = replace_linear_layers(model, skip=skip)
        torch.cuda.empty_cache()
        print(f"Quantized {replaced} nn.Linear layers to INT4 (lm_head {'INT4' if quantize_lm_head else 'kept in ' + str(dtype)})")

    tag = f"{precision}" + ("" if limit is None else f"_limit{limit}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    args = bigcode_args(limit, max_length, RESULTS_DIR / f"humaneval_generations_{tag}.json", RESULTS_DIR / f"humaneval_metrics_{tag}.json")

    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    patch_humaneval_task()
    evaluator = Evaluator(Accelerator(), model, tokenizer, args)
    print("Starting HumanEval evaluation (greedy pass@1)...")
    results = evaluator.evaluate("humaneval")
    if results is None:
        raise RuntimeError("bigcode Evaluator returned no results")

    pass_at_1 = results["pass@1"] if "pass@1" in results else results.get("humaneval", {}).get("pass@1")
    print(f"\n=== HumanEval pass@1 ({precision}): {pass_at_1} ===")
    save_results(f"humaneval_{tag}", {
        "model": MODEL, "precision": precision, "dtype": str(dtype), "int4_layers": replaced,
        "quantize_lm_head": quantize_lm_head if precision == "int4" else None,
        "limit": limit, "max_length_generation": max_length, "decoding": "greedy, n_samples=1",
        "pass@1": pass_at_1, "raw": results,
    })
    return results

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--precision", choices=["bf16", "int4"], default="int4")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N problems")
    parser.add_argument("--max-length", type=int, default=512, help="max prompt + completion tokens")
    parser.add_argument("--keep-lm-head", action="store_true", help="do not quantize the lm_head")
    args = parser.parse_args()
    run_evaluation(args.precision, args.limit, args.max_length, QUANTIZE_LM_HEAD and not args.keep_lm_head)

if __name__ == "__main__":
    main()