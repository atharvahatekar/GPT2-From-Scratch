"""Cross-entropy loss helpers for batches and full data loaders."""

from __future__ import annotations

import torch


def calc_loss_batch(input_batch, target_batch, model, device) -> torch.Tensor:
    """Cross-entropy loss for a single ``(input, target)`` batch.

    Logits ``(batch, seq_len, vocab)`` and targets ``(batch, seq_len)`` are
    flattened so the loss is computed over every predicted token.
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )


def calc_loss_loader(data_loader, model, device, num_batches: int | None = None) -> float:
    """Average loss over (up to ``num_batches`` of) a data loader."""
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / num_batches
