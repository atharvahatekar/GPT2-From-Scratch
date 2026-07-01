"""LLM From Scratch — a modular GPT-style language model built with PyTorch.

The package is organised into focused submodules that mirror the build stages:

- :mod:`src.config`      — model hyperparameter presets (GPT-2 sizes)
- :mod:`src.tokenizer`   — rule-based and BPE tokenizers
- :mod:`src.data`        — sliding-window dataset and dataloader
- :mod:`src.model`       — attention, layers, transformer block, GPT model
- :mod:`src.generation`  — autoregressive text generation
- :mod:`src.training`    — loss functions and the pretraining loop
- :mod:`src.utils`       — checkpointing, plotting, pretrained weight loading

Common entry points are re-exported here for convenience::

    from src import GPTModel, get_config, BPETokenizer, create_dataloader
"""

from __future__ import annotations

from .config import GPT_CONFIG_124M, GPT2_MODEL_CONFIGS, get_config
from .data import GPTDataset, create_dataloader
from .generation import (
    generate,
    generate_text_simple,
    text_to_token_ids,
    token_ids_to_text,
)
from .model import (
    FeedForward,
    GELU,
    GPTModel,
    LayerNorm,
    MultiHeadAttention,
    TransformerBlock,
)
from .tokenizer import BPETokenizer, SimpleTokenizerV1, SimpleTokenizerV2
from .training import (
    calc_loss_batch,
    calc_loss_loader,
    evaluate_model,
    train_model_simple,
)

__version__ = "0.1.0"

__all__ = [
    # config
    "GPT_CONFIG_124M",
    "GPT2_MODEL_CONFIGS",
    "get_config",
    # tokenizers
    "SimpleTokenizerV1",
    "SimpleTokenizerV2",
    "BPETokenizer",
    # data
    "GPTDataset",
    "create_dataloader",
    # model
    "MultiHeadAttention",
    "LayerNorm",
    "GELU",
    "FeedForward",
    "TransformerBlock",
    "GPTModel",
    # generation
    "generate",
    "generate_text_simple",
    "text_to_token_ids",
    "token_ids_to_text",
    # training
    "calc_loss_batch",
    "calc_loss_loader",
    "evaluate_model",
    "train_model_simple",
]
