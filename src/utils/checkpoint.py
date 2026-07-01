"""Save and load model (and optimizer) checkpoints."""

from __future__ import annotations

import torch


def save_checkpoint(path, model, optimizer=None) -> None:
    """Save ``model`` (and optionally ``optimizer``) state to ``path``."""
    checkpoint = {"model_state_dict": model.state_dict()}
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    """Load state from ``path`` into ``model`` (and ``optimizer`` if given)."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer
