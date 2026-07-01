"""Multi-head causal self-attention, implemented from scratch.

The single :class:`MultiHeadAttention` module splits the projected queries, keys
and values into multiple heads, applies scaled dot-product attention with a
causal mask, and recombines the heads with an output projection.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention.

    Parameters
    ----------
    d_in:
        Input embedding dimension.
    d_out:
        Output dimension. Must be divisible by ``num_heads``.
    context_length:
        Maximum sequence length; used to size the causal mask buffer.
    dropout:
        Dropout probability applied to the attention weights.
    num_heads:
        Number of attention heads.
    qkv_bias:
        Whether the query/key/value projections include a bias term.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        # Each head operates on a slice of the output dimension.
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Combines the per-head outputs
        self.dropout = nn.Dropout(dropout)
        # Upper-triangular causal mask (registered so it moves with .to(device)).
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, _ = x.shape

        keys = self.W_key(x)  # (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Split the d_out dimension into (num_heads, head_dim).
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Scaled dot-product attention with a causal mask.
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # (b, num_tokens, num_heads, head_dim) -> combine heads -> (b, num_tokens, d_out)
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)
