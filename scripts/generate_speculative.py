from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import torch

from transformers import AutoTokenizer

import codealign_runtime_transformer as cuda_engine
from scripts.engine import CodeAlignEngine
from scripts.inference_config import MAX_SEQ_LEN, MODEL, QUANTIZE_LM_HEAD
from scripts.results_io import save_results

@dataclass
class GenerationStats:
    new_tokens: int = 0
    forwards: int = 0        # engine forward calls (= decode steps)
    drafted: int = 0         # draft tokens proposed by the oracle
    accepted: int = 0        # draft tokens accepted
    seconds: float = 0.0          # decode only (after the prompt prefill)
    prefill_seconds: float = 0.0  # identical for greedy and PLD, reported separately

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0

    @property
    def tokens_per_forward(self) -> float:
        return self.new_tokens / self.forwards if self.forwards else 0.0

    @property
    def tokens_per_second(self) -> float:
        return self.new_tokens / self.seconds if self.seconds else 0.0

    def as_dict(self) -> dict:
        return {**asdict(self), "acceptance_rate": self.acceptance_rate, "tokens_per_forward": self.tokens_per_forward, "tokens_per_second": self.tokens_per_second}

def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        
class SpeculativeDecoder:
    def __init__(self, engine: CodeAlignEngine, ngram: int = 3, max_draft: int = 5):
        self.engine = engine
        self.ngram = ngram
        self.max_draft = max_draft

    @torch.inference_mode()
    def generate(self, prompt_ids: Sequence[int], max_new_tokens: int, eos_token_ids: Sequence[int] = (), speculative: bool = True) -> tuple[list[int], GenerationStats]:
        engine = self.engine
        eos = set(eos_token_ids)
        stats = GenerationStats()
        history = list(prompt_ids)
        new_tokens: list[int] = []

        _sync(engine.device)
        t0 = time.perf_counter()

        engine.reset()
        engine.prefill(history[:-1], logits="none")
        _sync(engine.device)
        t_decode = time.perf_counter()
        stats.prefill_seconds = t_decode - t0

        finished = False
        while not finished and len(new_tokens) < max_new_tokens:
            room = engine.max_seq_len - engine.seq_len - 1          
            if room < 0:
                break
            max_k = min(self.max_draft, engine.max_tokens_per_forward - 1, room, max_new_tokens - len(new_tokens) - 1)
            draft = cuda_engine.find_candidate_draft(history, self.ngram, max_k) if speculative and max_k > 0 else []

            logits = engine.forward([history[-1]] + draft, logits="all")
            predicted = engine.argmax(logits)

            accepted = 0
            while accepted < len(draft) and draft[accepted] == predicted[accepted]:
                accepted += 1
            step_tokens = draft[:accepted] + [predicted[accepted]] 
            engine.rollback(len(draft) - accepted)                    

            stats.forwards += 1
            stats.drafted += len(draft)
            stats.accepted += accepted
            for token in step_tokens:
                history.append(token)
                new_tokens.append(token)
                if token in eos or len(new_tokens) >= max_new_tokens:
                    finished = True
                    break

        _sync(engine.device)
        stats.seconds = time.perf_counter() - t_decode
        stats.new_tokens = len(new_tokens)
        return new_tokens, stats

PROMPTS = {
    "refactor_type_hints": '''class InventoryItem:
    def __init__(self, name, unit_price, quantity=0):
        self.name = name
        self.unit_price = unit_price
        self.quantity = quantity

    def total_cost(self):
        return self.unit_price * self.quantity

    def restock(self, amount):
        self.quantity += amount


# The same class, with type hints added to every method:
class InventoryItem:
''',
    "cpp_getters_setters": '''// C++ class User with id, name, email getters and setters
class User {
private:
    int id;
    std::string name;
    std::string email;

public:
''',
    "unit_tests": '''def slugify(text: str) -> str:
    """Lowercase, strip, and replace runs of non-alphanumerics with a single '-'."""
    import re
    return re.sub(r"[^a-z0-9]+", "-", text.lower().strip()).strip("-")


import unittest


class TestSlugify(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(slugify("Hello World"), "hello-world")

    def test_punctuation(self):
''',
    "open_ended_control": '''# Python: a small command-line todo application using argparse and a JSON file for storage
''',
}

def first_divergence(a: Sequence[int], b: Sequence[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--ngram", type=int, default=3)
    parser.add_argument("--max-draft", type=int, default=5)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    engine = CodeAlignEngine.from_pretrained(MODEL, max_seq_len=MAX_SEQ_LEN, quantize_lm_head=QUANTIZE_LM_HEAD)
    decoder = SpeculativeDecoder(engine, ngram=args.ngram, max_draft=args.max_draft)
    eos = [tokenizer.eos_token_id]

    warm = tokenizer("def f(x):\n    return x")["input_ids"]
    decoder.generate(warm, 16, eos, speculative=False)
    decoder.generate(warm, 16, eos, speculative=True)

    rows = []
    for name, prompt in PROMPTS.items():
        prompt_ids = tokenizer(prompt)["input_ids"]
        greedy_tokens, greedy = decoder.generate(prompt_ids, args.max_new_tokens, eos, speculative=False)
        pld_tokens, pld = decoder.generate(prompt_ids, args.max_new_tokens, eos, speculative=True)
        divergence = first_divergence(greedy_tokens, pld_tokens)
        speedup = pld.tokens_per_second / greedy.tokens_per_second if greedy.tokens_per_second else 0.0
        rows.append({"prompt": name, "prompt_tokens": len(prompt_ids), "greedy": greedy.as_dict(), "pld": pld.as_dict(),
                     "speedup": speedup, "identical_to_greedy": divergence is None, "first_divergence": divergence,
                     "pld_text": tokenizer.decode(pld_tokens)})
        print(f"{name:22s} greedy {greedy.tokens_per_second:7.1f} tok/s | PLD {pld.tokens_per_second:7.1f} tok/s "
              f"(x{speedup:.2f}) | acceptance {100 * pld.acceptance_rate:5.1f}% | "
              f"{pld.tokens_per_forward:.2f} tok/forward | identical: {divergence is None}"
              + ("" if divergence is None else f" (first divergence at token {divergence})"))

    total_greedy = sum(r["greedy"]["seconds"] for r in rows)
    total_pld = sum(r["pld"]["seconds"] for r in rows)
    print(f"\nOverall decode time (prefill excluded): greedy {total_greedy:.2f} s vs PLD {total_pld:.2f} s "
          f"(x{total_greedy / total_pld:.2f})")
    print(f"\n--- PLD output for '{rows[0]['prompt']}' ---\n{rows[0]['pld_text'][:800]}")

    save_results("level6_speculative", {
        "model": MODEL, "ngram": args.ngram, "max_draft": args.max_draft, "max_new_tokens": args.max_new_tokens,
        "prompts": rows, "overall_speedup": total_greedy / total_pld if total_pld else 0.0,
    })

if __name__ == "__main__":
    main()