"""Autoregressive text generation and token <-> text helpers."""

from __future__ import annotations

import torch


def text_to_token_ids(text: str, tokenizer) -> torch.Tensor:
    """Encode ``text`` into a ``(1, seq_len)`` batch of token ids."""
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids: torch.Tensor, tokenizer) -> str:
    """Decode a ``(1, seq_len)`` batch of token ids back into text."""
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def generate_text_simple(
    model, idx: torch.Tensor, max_new_tokens: int, context_size: int
) -> torch.Tensor:
    """Greedy decoding: repeatedly append the arg-max next token.

    ``idx`` is a ``(batch, n_tokens)`` tensor of the current context.
    """
    for _ in range(max_new_tokens):
        # Crop the context to the last ``context_size`` tokens.
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]  # focus on the last time step
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """Decoding with optional top-k filtering and temperature sampling.

    - ``temperature == 0`` (default) is greedy decoding.
    - ``temperature > 0`` samples from the (optionally top-k filtered) softmax.
    - ``eos_id`` stops generation early when that token is produced.
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # Top-k filtering: mask out all but the k most likely tokens.
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf")).to(logits.device),
                logits,
            )

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and idx_next.item() == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)
    return idx
