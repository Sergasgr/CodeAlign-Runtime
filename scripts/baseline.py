import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.baseline_config import (
    MODEL, 
    PROMPT, 
    NUM_ITERATIONS,
    MAX_TOKENS,
    GPU_BANDWIDTH_GBPS, 
    MODEL_PARAMS_BILLIONS,
    MODEL_BYTES,
    THEORETICAL_MIN_TPOT_MS
)

def run_benchmark():
    precision = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16 # fp16 or bf16
    print(f"Loading {MODEL} using precision: {precision}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=MODEL,
        dtype=precision,
        device_map="cuda" 
    )
    chatml = [{"role": "user", "content": PROMPT}]
    text = tokenizer.apply_chat_template(chatml, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    first_token_start = torch.cuda.Event(enable_timing=True)
    first_token_end = torch.cuda.Event(enable_timing=True)
    token_processed_start = torch.cuda.Event(enable_timing=True)
    token_processed_end = torch.cuda.Event(enable_timing=True)
    
    ttft_list = []
    tpot_list = []

    print(f"\nExecuting benchmark ({NUM_ITERATIONS} iterations)...")
    
    warmup_steps = min(10, max(1, NUM_ITERATIONS // 10))
    
    torch.cuda.reset_peak_memory_stats()
    for i in range(NUM_ITERATIONS):
        with torch.no_grad(): # TTFT & TPOT
            # --- 1. TTFT (Prefill) ---
            first_token_start.record()
            outputs = model(**inputs, use_cache=True) 
            first_token_end.record()
            torch.cuda.synchronize()
            
            if i >= warmup_steps: # Ignoring the first results -> warmup
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
                
                if i >= warmup_steps:
                    tpot_list.append(token_processed_start.elapsed_time(token_processed_end))
                
                input_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
                past_key_values = outputs.past_key_values
                tokens_generated += 1
        
    print("\n=== Results ===")
    if ttft_list and tpot_list:
        print(f"TTFT (ms): p50={np.percentile(ttft_list, 50):.2f}, p90={np.percentile(ttft_list, 90):.2f}, p99={np.percentile(ttft_list, 99):.2f}")
        
        tpot_p50 = np.percentile(tpot_list, 50)
        print(f"TPOT (ms/token): p50={tpot_p50:.2f}, p90={np.percentile(tpot_list, 90):.2f}, p99={np.percentile(tpot_list, 99):.2f}")
        
        print("\n=== Theoretical Roofline ===")
        print(f"Hardware setup: {MODEL_PARAMS_BILLIONS}B model ({MODEL_BYTES / (1024**3):.2f} GB) on GPU with {GPU_BANDWIDTH_GBPS} GB/s BW")
        print(f"Theoretical minimum TPOT: {THEORETICAL_MIN_TPOT_MS:.2f} ms/token")
        
        efficiency = (THEORETICAL_MIN_TPOT_MS / tpot_p50) * 100
        print(f"Peak bandwidth ceiling reached: {efficiency:.2f}%")
    else:
        print("Error: No valid metrics gathered. Check NUM_ITERATIONS or MAX_TOKENS.")
    
    vram_peak_bytes = torch.cuda.max_memory_allocated()
    vram_peak_mb = vram_peak_bytes / (1024 ** 2)
    
    print("\n=== Memory Footprint ===")
    print(f"Peak VRAM allocated: {vram_peak_mb:.2f} MB")
        
if __name__ == "__main__":
    run_benchmark()