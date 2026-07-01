"""The transformer block: pre-norm attention and feed-forward with residuals."""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .layers import FeedForward, LayerNorm


class TransformerBlock(nn.Module):
    """A single GPT transformer block.

    Uses the pre-LayerNorm variant: each sub-layer normalizes its input, applies
    the transformation, applies dropout, and adds the residual (shortcut)
    connection back.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention sub-layer with residual connection.
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # Feed-forward sub-layer with residual connection.
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
