import torch
import torch.nn as nn
from cs336_basics.linear import Linear
def silu(x):
    in_dtype = x.dtype
    x = x.to(torch.float32)
    result = x * torch.sigmoid(x)
    return result.to(in_dtype)

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff , d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x):
        gate = silu(self.w1(x))*self.w3(x)    
        return self.w2(gate)