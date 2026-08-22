import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from scripts.baseline_config import MODEL
from src.quantization import replace_linear_layers
from bigcode_eval.evaluator import Evaluator
from bigcode_eval.tasks import get_task

def run_evaluation():
    precision = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16 
    print(f"Loading {MODEL} using precision: {precision}")
        
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=MODEL,
        torch_dtype=precision,
        device_map="cuda"
    )
    
    print("Quantizing the model to INT4 in memory...")
    replace_linear_layers(model)
    print("Model successfully quantized!")
    
    print("Starting HumanEval evaluation (pass@1)...")
    args = argparse.Namespace(
        model=MODEL,
        tasks="humaneval",
        do_sample=False, 
        temperature=0.2,
        top_p=0.95,
        top_k=0,
        n_samples=1, 
        batch_size=1,
        allow_code_execution=True, 
        limit=None,
        generation_only=False,
        postprocess=True,
        save_generations=False,
        save_references=False,
        mutate_method="edit",
        max_length_generation=512,
        instruction_tokens=None,
        check_references=False,
        metric_output_path="evaluation_results.json",
        prefix=""
    )

    accelerator = Accelerator()
    evaluator = Evaluator(accelerator, model, tokenizer, args)
    results = evaluator.evaluate(task_name="humaneval") 
    
    print("\n===RESULTS===")
    print(f"Pass@1 (Optimized INT4): {results['humaneval']['pass@1']}")

if __name__ == "__main__":
    run_evaluation()