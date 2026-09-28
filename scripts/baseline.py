import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.baseline_config import MAX_TOKENS, MODEL, NUM_ITERATIONS, PROMPT, WARMUP_ITERATIONS
from scripts.results_io import percentiles, save_results
from scripts.roofline import GPU_BANDWIDTH_GBPS, ceiling_ms

def model_bytes(model) -> int:
    seen, total = set(), 0
    for p in model.parameters():
        if p.data_ptr() in seen:
            continue
        seen.add(p.data_ptr())
        total += p.numel() * p.element_size()
    return total

def run_benchmark():
    precision = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16 # fp16 or bf16
    print(f"Loading {MODEL} using precision: {precision}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=MODEL,
        dtype=precision,
        device_map="cuda"
    )
    model.eval()
    inputs = tokenizer([PROMPT], return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    bytes_per_token = model_bytes(model)
    tpot_ceiling_ms = ceiling_ms(bytes_per_token)

    first_token_start = torch.cuda.Event(enable_timing=True)
    first_token_end = torch.cuda.Event(enable_timing=True)
    token_processed_start = torch.cuda.Event(enable_timing=True)
    token_processed_end = torch.cuda.Event(enable_timing=True)

    ttft_list = []
    tpot_list = []

    print(f"\nExecuting benchmark ({NUM_ITERATIONS} iterations + {WARMUP_ITERATIONS} warmup, prompt = {prompt_len} tokens)...")

    torch.cuda.reset_peak_memory_stats()
    for i in range(NUM_ITERATIONS + WARMUP_ITERATIONS):
        with torch.no_grad(): # TTFT & TPOT
            # --- 1. TTFT (Prefill) ---
            first_token_start.record()
            outputs = model(**inputs, use_cache=True)
            first_token_end.record()
            torch.cuda.synchronize()

            if i >= WARMUP_ITERATIONS: # Ignoring the first results -> warmup
                ttft_list.append(first_token_start.elapsed_time(first_token_end))

            # --- 2. TPOT (Decode) ---
            input_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
            past_key_values = outputs.past_key_values

            tokens_generated = 0

            while input_id.item() != tokenizer.eos_token_id and tokens_generated < MAX_TOKENS:
                token_processed_start.record()
                outputs = model(input_ids=input_id, past_key_values=past_key_values, use_cache=True)
                token_processed_end.record()
                torch.cuda.synchronize()

                if i >= WARMUP_ITERATIONS:
                    tpot_list.append(token_processed_start.elapsed_time(token_processed_end))

                input_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
                past_key_values = outputs.past_key_values
                tokens_generated += 1

    vram_peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print("\n=== Results ===")
    if not (ttft_list and tpot_list):
        print("Error: No valid metrics gathered. Check NUM_ITERATIONS or MAX_TOKENS.")
        return

    ttft = percentiles(ttft_list)
    tpot = percentiles(tpot_list)
    print(f"TTFT (ms): p50={ttft['p50_ms']:.2f}, p90={ttft['p90_ms']:.2f}, p99={ttft['p99_ms']:.2f}")
    print(f"TPOT (ms/token): p50={tpot['p50_ms']:.2f}, p90={tpot['p90_ms']:.2f}, p99={tpot['p99_ms']:.2f}")

    efficiency = tpot_ceiling_ms / tpot["p50_ms"] * 100
    print("\n=== Theoretical Roofline ===")
    print(f"Weights read per token: {bytes_per_token / 1e6:.0f} MB ({precision}) at {GPU_BANDWIDTH_GBPS} GB/s")
    print(f"Theoretical minimum TPOT: {tpot_ceiling_ms:.3f} ms/token")
    print(f"Bandwidth ceiling reached (p50): {efficiency:.2f}%")
    print("\n=== Memory Footprint ===")
    print(f"Peak VRAM allocated: {vram_peak_mb:.2f} MB")

    save_results("level0_baseline", {
        "model": MODEL,
        "precision": str(precision),
        "prompt_tokens": prompt_len,
        "max_new_tokens": MAX_TOKENS,
        "iterations": NUM_ITERATIONS,
        "warmup": WARMUP_ITERATIONS,
        "ttft": ttft,
        "tpot": tpot,
        "weights_bytes_per_token": bytes_per_token,
        "tpot_ceiling_ms": tpot_ceiling_ms,
        "pct_of_ceiling_p50": efficiency,
        "peak_vram_mb": vram_peak_mb,
    })

if __name__ == "__main__":
    run_benchmark()