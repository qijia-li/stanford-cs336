import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    in_dtype = x.dtype
    x = x.to(torch.float32)

    # Numerically stable: subtract max along dim before exp to avoid overflow
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    result = exp_x / exp_x.sum(dim=dim, keepdim=True)
    return result.to(in_dtype)

def cross_entropy_loss(
    inputs: Float[Tensor, "batch_size_vocab_size"],
    targets: Int[Tensor, "batch_size"],
) -> Float[Tensor, ""]:
    log_probs = inputs.float()    
    log_probs = log_probs - log_probs.logsumexp(dim=-1, keepdim=True)

    batch_idx = torch.arange(inputs.size(0), device=inputs.device)
    loss = -log_probs[batch_idx, targets].mean()
    
    return loss.to(inputs.dtype)