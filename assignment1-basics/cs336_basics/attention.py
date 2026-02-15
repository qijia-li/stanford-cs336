import torch
import torch.nn as nn
from cs336_basics.rope import RoPE
from jaxtyping import Float, Int
from torch import Tensor
from cs336_basics.softmax import softmax
from einops import einsum, rearrange
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    '''
    Q: (batch_size, ..., queries, d_k)
    K: (batch_size, ..., keys, d_k)
    V: (batch_size, ..., values, d_v)
    mask: same shape as scores. True = attend, False = masked (PyTorch convention).

    Returns:
        (batch_size, ..., d_v)
    '''
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        d_k = Q.shape[-1]
        # scores.shape: (..., queries, keys)
        score = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)
        if mask is not None:
            # PyTorch convention: True = attend, False = masked. Set masked positions to -inf.
            score = score.masked_fill(~mask, float('-inf'))
        attention = softmax(score, dim=-1)
        return einsum(attention, V, "... queries keys, ... keys d_v -> ... queries d_v")

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

    def forward(
        self, 
        q_proj_weight: Float[Tensor, " d_k d_in"],
        k_proj_weight: Float[Tensor, " d_k d_in"],
        v_proj_weight: Float[Tensor, " d_v d_in"],
        o_proj_weight: Float[Tensor, " d_model d_v"],
        in_features: Float[Tensor, " ... sequence_length d_in"]
    ) -> Float[Tensor, " ... sequence_length d_out"]:
        seq_len = in_features.shape[-2]

        # Project and reshape to (..., num_heads, seq_len, head_dim)
        Q = einsum(in_features, q_proj_weight, "... seq d_in, d_k d_in -> ... seq d_k")
        K = einsum(in_features, k_proj_weight, "... seq d_in, d_k d_in -> ... seq d_k")
        V = einsum(in_features, v_proj_weight, "... seq d_in, d_v d_in -> ... seq d_v")

        Q = rearrange(Q, "... seq (head d_k) -> ... head seq d_k", head=self.num_heads)
        K = rearrange(K, "... seq (head d_k) -> ... head seq d_k", head=self.num_heads)
        V = rearrange(V, "... seq (head d_v) -> ... head seq d_v", head=self.num_heads)

        # Causal mask: True = attend (past and current), False = masked (future)
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=Q.device), diagonal=0
        )
        # Broadcast for (..., head, query, key)
        # causal_mask.shape (seq_len, seq_len), 
        # unsqueeze(0): (seq_len, seq_len) → (1, seq_len, seq_len)
        # unsqueeze(0): (1, seq_len, seq_len) → (1, 1, seq_len, seq_len)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        attn_out = ScaledDotProductAttention()(Q, K, V, causal_mask)
        attn_out = rearrange(attn_out, "... head seq d_v -> ... seq (head d_v)")
        return einsum(attn_out, o_proj_weight, "... seq d_v, d_model d_v -> ... seq d_model")

class  MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.head_dim = d_model // num_heads
        self.rope = RoPE(theta=theta, d_k=self.head_dim, max_seq_len=max_seq_len, device=None, dtype=None)

    def forward(
        self,
        q_proj_weight: Float[Tensor, " d_k d_in"],
        k_proj_weight: Float[Tensor, " d_k d_in"],
        v_proj_weight: Float[Tensor, " d_v d_in"],
        o_proj_weight: Float[Tensor, " d_model d_v"],
        in_features: Float[Tensor, " ... sequence_length d_in"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, " ... sequence_length d_out"]:
        seq_len = in_features.shape[-2]

        # Project and reshape to (..., num_heads, seq_len, head_dim)
        Q = einsum(in_features, q_proj_weight, "... seq d_in, d_k d_in -> ... seq d_k")
        K = einsum(in_features, k_proj_weight, "... seq d_in, d_k d_in -> ... seq d_k")
        V = einsum(in_features, v_proj_weight, "... seq d_in, d_v d_in -> ... seq d_v")

        Q = rearrange(Q, "... seq (head d_k) -> ... head seq d_k", head=self.num_heads)
        K = rearrange(K, "... seq (head d_k) -> ... head seq d_k", head=self.num_heads)
        V = rearrange(V, "... seq (head d_v) -> ... head seq d_v", head=self.num_heads)

        # Apply RoPE to Q and K (not V). Positions must broadcast to (..., seq_len) for 4D Q/K.
        if token_positions is not None and Q.dim() == 4:
            positions = token_positions.unsqueeze(1)
        else:
            positions = token_positions
        Q = self.rope(Q, positions)
        K = self.rope(K, positions)

        # Causal mask: True = attend (past and current), False = masked (future)
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=Q.device), diagonal=0
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        attn_out = ScaledDotProductAttention()(Q, K, V, causal_mask)
        attn_out = rearrange(attn_out, "... head seq d_v -> ... seq (head d_v)")
        return einsum(attn_out, o_proj_weight, "... seq d_v, d_model d_v -> ... seq d_model")
