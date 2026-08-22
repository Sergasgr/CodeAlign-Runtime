import torch
import torch.nn as nn
import codealign_runtime_kernels

class QuantizedLinearINT4(nn.Module):
    def __init__(self, q_weight, scales, bias = None):
        super().__init__()
        self.q_weight = q_weight
        self.scales = scales
        self.bias = bias
        
    def forward(self, x: torch.Tensor):
        batch_size, seq_len, in_features = x.shape
        if batch_size == 1 and seq_len == 1 and in_features:
            out = codealign_runtime_kernels.gemv_int4_forward(self.q_weight, self.scales, x.squeeze())
            if self.bias is not None: 
                out += self.bias
            return out.view(1, 1, -1)
        raise NotImplementedError("La fase prefill aún no está implementada") 
            
def quantize_to_int4(weight: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = weight.shape
    w_groups = weight.view(-1, group_size)
    max_vals = w_groups.abs().max(dim=1, keepdim=True)[0]
    scales = torch.clamp(max_vals, min=1e-9) / 7.0
    
    w_groups = torch.round(w_groups / scales).clamp(-8, 7).to(torch.int32)
    w_pack = w_groups.view(-1, 8)
    
    packed = torch.zeros(w_pack.shape[0], dtype=torch.int32, device=weight.device)
    for i in range(8):
        packed = packed | ((w_pack[:, i] & 0xF) << (4 * i))
    
    packed = packed.view(rows, cols // 8)
    scales = scales.view(rows, cols // group_size)
    
    return packed, scales

def replace_linear_layers(module: nn.Module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            q_weight, scales = quantize_to_int4(child.weight.data)
            new_layer = QuantizedLinearINT4(q_weight, scales, child.bias)
            setattr(module, name, new_layer)
        else:
            replace_linear_layers(child)