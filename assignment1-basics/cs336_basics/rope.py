import torch
import torch.nn as nn

def apply_rotary_emb(x, cos, sin):
    # x: (..., seq_len, d_k); cos, sin: (..., seq_len, half_d_k)
    # 旋转 [x0,x1] -> [cos*x0 - sin*x1, sin*x0 + cos*x1]，用 einsum 做批量 2x2 旋转
    half_d_k = cos.shape[-1]
    x_flat = x.reshape(*x.shape[:-1], half_d_k, 2)  # (..., seq_len, half_d_k, 2)
    # R: (..., seq_len, half_d_k, 2, 2), R[...,0,:]=[cos,-sin], R[...,1,:]=[sin,cos]
    R = torch.stack(
        [torch.stack([cos, -sin], dim=-1), torch.stack([sin, cos], dim=-1)],
        dim=-2,
    )
    out_flat = torch.einsum("...pdij,...pdj->...pdi", R, x_flat)
    return out_flat.reshape(*x.shape)

class RoPE(nn.Module):
    """Rotary Position Embedding (RoPE). 将位置信息编码为旋转，应用于 Q/K。"""

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.half_d_k = d_k // 2

        # Step 1: Compute the inverse frequency
        #  theta^(-2i/d_k), i = 0, 1, ..., half_d_k-1
        inv_freq = 1.0 / (theta ** (torch.arange(self.half_d_k, device=device, dtype=torch.float32) * 2 / d_k))
        self.register_buffer("inv_freq", inv_freq.to(dtype or torch.get_default_dtype()))

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        # x: (..., seq_len, d_k)
        _, seq_len, _ = x.shape
        if positions is None:
            positions = torch.arange(seq_len, device=x.device, dtype=torch.int64)
        # positions 需能 broadcast 到 (..., seq_len)；再与 inv_freq 乘得 (..., seq_len, half_d_k)
        positions = positions.to(x.dtype)
        angles = positions.unsqueeze(-1) * self.inv_freq  # (..., seq_len, half_d_k)
        cos = angles.cos()
        sin = angles.sin()
        return apply_rotary_emb(x, cos, sin)