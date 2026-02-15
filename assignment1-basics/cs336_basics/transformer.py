"""
Pre-norm Transformer block: normalize input with RMSNorm, then apply attention/FFN, then add residual.
Flow: x -> ln1 -> MHA(RoPE) -> +x -> ln2 -> SwiGLU FFN -> +residual -> output
"""
from cs336_basics.attention import MultiHeadAttentionWithRoPE
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.swiglu import silu
from jaxtyping import Float
from torch import Tensor
import torch
import torch.nn as nn
from einops import einsum


class TransformerBlock(nn.Module):
    """Single pre-norm Transformer block with multi-head self-attention (RoPE) and SwiGLU FFN."""

    def __init__(self):
        super().__init__()

    def forward(self, d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, "batch sequence_length d_model"],
    ) -> Float[Tensor, "batch sequence_length d_model"]:
        """
        Given the weights of a pre-norm Transformer block and input features,
        return the output of running the Transformer block on the input features using RoPE.

        Args:
            d_model (int): The dimensionality of the Transformer block input.
            num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be evenly divisible by `num_heads`.
            d_ff (int): Dimensionality of the feed-forward inner layer.
            max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
            theta (float): RoPE parameter.
            weights (dict[str, Tensor]): State dict of our reference implementation.
            State dict of our reference implementation.
            The keys of this dictionary are:
            in_features (Float[Tensor, "batch sequence_length d_model"]): Tensor to run your implementation on.

        Returns:
            Float[Tensor, "batch sequence_length d_model"]: Tensor with the output of running the Transformer block on the input features while using RoPE.
        """
        # -------------------------------------------------------------------------
        # Load layer weights from state dict (stored as (out_dim, in_dim))
        # -------------------------------------------------------------------------
        q_proj_weight = weights["attn.q_proj.weight"]
        k_proj_weight = weights["attn.k_proj.weight"]
        v_proj_weight = weights["attn.v_proj.weight"]
        o_proj_weight = weights["attn.output_proj.weight"]
        ln1_weight = weights["ln1.weight"]
        ffn_w1_weight = weights["ffn.w1.weight"]
        ffn_w2_weight = weights["ffn.w2.weight"]
        ffn_w3_weight = weights["ffn.w3.weight"]
        ln2_weight = weights["ln2.weight"]

        # -------------------------------------------------------------------------
        # 1) Pre-norm attention: ln1(x) -> MHA(RoPE) -> residual x + attn_out
        # -------------------------------------------------------------------------
        ln1 = RMSNorm(d_model, device=in_features.device, dtype=in_features.dtype)
        ln1.weight.data.copy_(ln1_weight)
        normed = ln1(in_features)
        mhar = MultiHeadAttentionWithRoPE(d_model=d_model, num_heads=num_heads, theta=theta, max_seq_len=max_seq_len)
        attn_out = mhar(q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, normed)
        attn_out = attn_out + in_features  # residual connection

        # -------------------------------------------------------------------------
        # 2) Pre-norm FFN (SwiGLU): ln2(attn_out) -> gate*up -> w2 -> residual
        #    SwiGLU: out = w2( silu(w1(x)) * w3(x) ); weights are (out_dim, in_dim)
        # -------------------------------------------------------------------------
        ln2 = RMSNorm(d_model, device=attn_out.device, dtype=attn_out.dtype)
        ln2.weight.data.copy_(ln2_weight)
        ffn_input = ln2(attn_out)
        # w1, w3: (d_ff, d_model) -> (..., d_ff); w2: (d_model, d_ff) -> (..., d_model)
        gate = einsum(ffn_input, ffn_w1_weight, "... sequence_length d_model, d_ff d_model -> ... sequence_length d_ff")
        up = einsum(ffn_input, ffn_w3_weight, "... sequence_length d_model, d_ff d_model -> ... sequence_length d_ff")
        ffn_out = einsum(silu(gate) * up, ffn_w2_weight, "... sequence_length d_ff, d_model d_ff -> ... sequence_length d_model")
        return attn_out + ffn_out  # residual connection; no final post-norm