"""Model configuration presets for the GPT models built in this project.

A configuration is a plain ``dict`` describing the model hyperparameters. Keeping
it as a dict (instead of a dataclass) mirrors the layer signatures used across
``src.model`` and keeps the code close to the reference implementation.
"""

from __future__ import annotations

from copy import deepcopy

# Base configuration: GPT-2 small (124M parameters).
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Number of tokens in the GPT-2 BPE vocabulary
    "context_length": 1024,  # Maximum sequence length the model can attend over
    "emb_dim": 768,          # Embedding / hidden dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of transformer blocks
    "drop_rate": 0.1,        # Dropout probability
    "qkv_bias": False,       # Whether Q/K/V projections use a bias term
}

# Size-specific overrides for the four public GPT-2 checkpoints. Combine these
# with ``GPT_CONFIG_124M`` via :func:`get_config`.
GPT2_MODEL_CONFIGS = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


def get_config(model_name: str = "gpt2-small (124M)", **overrides) -> dict:
    """Return a full model config for ``model_name``.

    Starts from :data:`GPT_CONFIG_124M`, applies the size-specific overrides from
    :data:`GPT2_MODEL_CONFIGS`, then applies any explicit keyword ``overrides``.

    Example
    -------
    >>> cfg = get_config("gpt2-medium (355M)", qkv_bias=True)
    >>> cfg["emb_dim"], cfg["qkv_bias"]
    (1024, True)
    """
    if model_name not in GPT2_MODEL_CONFIGS:
        raise KeyError(
            f"Unknown model '{model_name}'. Choose from {list(GPT2_MODEL_CONFIGS)}."
        )
    cfg = deepcopy(GPT_CONFIG_124M)
    cfg.update(GPT2_MODEL_CONFIGS[model_name])
    cfg.update(overrides)
    return cfg
