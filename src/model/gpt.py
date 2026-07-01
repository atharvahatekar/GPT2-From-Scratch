"""The full GPT model: token + positional embeddings, transformer stack, head."""

from __future__ import annotations

import torch
import torch.nn as nn

from .layers import LayerNorm
from .transformer import TransformerBlock


class GPTModel(nn.Module):
    """A GPT-style decoder-only transformer language model.

    The forward pass maps a batch of token id sequences ``(batch, seq_len)`` to
    vocabulary logits ``(batch, seq_len, vocab_size)``.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        _, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # (batch, seq_len, emb_dim)
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)

    def num_params(self, trainable_only: bool = True) -> int:
        """Return the parameter count (trainable by default)."""
        params = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)
