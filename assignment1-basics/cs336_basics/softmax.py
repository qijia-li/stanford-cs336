import torch
import torch.nn as nn

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    in_dtype = x.dtype
    x = x.to(torch.float32)

    # Numerically stable: subtract max along dim before exp to avoid overflow
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    result = exp_x / exp_x.sum(dim=dim, keepdim=True)
    return result.to(in_dtype)