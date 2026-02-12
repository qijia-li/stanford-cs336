import math
import torch
import torch.nn as nn
from einops import einsum


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        std = 2.0 / math.sqrt(in_features + out_features)
        # PyTorch convention: weight (out_features, in_features)
        self.weight = nn.init.trunc_normal_(
            torch.Tensor(out_features, in_features).to(device=device, dtype=dtype), mean=0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_in), weight: (d_out, d_in) -> out: (..., d_out)
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")