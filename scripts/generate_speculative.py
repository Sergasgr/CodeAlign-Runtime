import time
import torch
import codealign_runtime_kernels 
from codealign_runtime_kernels import QwenBlock

class SpeculativeDecoder:
    def __init__(self, block: QwenBlock, embed_layer: torch.nn.Embedding, lm_head: torch.nn.Linear, max_seq_len: int):
        self.block = block
        self.embed_layer = embed_layer
        self.lm_head = lm_head
        self.max_seq_len = max_seq_len
        self.N = 3
        self.K = 5

    def generate(self, history: list) -> list:
        while len(history) < self.max_seq_len:
            draft = codealign_runtime_kernels.find_candidate_draft(history, self.N, self.K)
            current_tokens = [history[-1]] + draft
            
            input_tensor = torch.tensor(current_tokens, dtype=torch.int32, device="cuda")
            hidden_states = self.embed_layer(input_tensor)
            out_hidden = self.block.forward(hidden_states)
            logits = self.lm_head(out_hidden)
            
            predicted_tokens = codealign_runtime_kernels.fast_argmax(logits).tolist()
            
            accept_count = 0
            for d, p in zip(draft, predicted_tokens):
                if d != p: break
                accept_count += 1
        
            history.extend(draft[:accept_count])         
            history.append(predicted_tokens[accept_count])
            
            rejected = len(draft) - accept_count
            if rejected > 0:
                self.block.rollback_kv_cache(rejected)
                
        return history
    
if __name__ == "__main__":
    # Load model and tokenizer
    # block = QwenBlock(...)
    # block.load_weights(...)
    # embed_layer = ...
    # lm_head = ...
    
    decoder = SpeculativeDecoder(block, embed_layer, lm_head, max_seq_len=512)
    
    prompt = "// C++ class User with id, name, email getters and setters\nclass User {"
    history = [123, 456, 789] 
    initial_len = len(history)
    
    start_time = torch.cuda.Event(enable_timing=True) 
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()   
    final_history = decoder.generate(history)
    end_time.record()
    torch.cuda.synchronize()
    
    generated_tokens = len(final_history) - initial_len
    total_time = end_time - start_time
    tps = generated_tokens / total_time
    
    print(f"--- Level 6 Results ---")
    print(f"Tokens generated: {generated_tokens}")
    print(f"Total time: {total_time:.3f} s")
    print(f"Speed: {tps:.2f} Tokens/Second")
    
    # Also print the Acceptance Rate if you instrumented your class to count it.
    # print(f"Generated text:\n{tokenizer.decode(final_history)}")
    
    # Execute it and compare it with find_candidate_draft = [] always