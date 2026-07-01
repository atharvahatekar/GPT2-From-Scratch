"""Byte Pair Encoding tokenizer that wraps OpenAI's :mod:`tiktoken`.

This is the production tokenizer used for the rest of the pipeline. It exposes a
thin, project-consistent ``encode`` / ``decode`` interface over the GPT-2 BPE
vocabulary so it can be swapped in wherever a tokenizer is expected.
"""

from __future__ import annotations

import tiktoken

ENDOFTEXT_TOKEN = "<|endoftext|>"


class BPETokenizer:
    """Reusable BPE tokenizer wrapper using :mod:`tiktoken`."""

    def __init__(self, encoding_name: str = "gpt2", allowed_special=None):
        self.tokenizer = tiktoken.get_encoding(encoding_name)
        self.allowed_special = set(allowed_special or {ENDOFTEXT_TOKEN})
        self.vocab_size = self.tokenizer.max_token_value + 1

    def encode(self, text: str, allowed_special=None) -> list[int]:
        allowed = allowed_special if allowed_special is not None else self.allowed_special
        return self.tokenizer.encode(text, allowed_special=allowed)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def encode_batch(self, texts, allowed_special=None) -> list[list[int]]:
        allowed = allowed_special if allowed_special is not None else self.allowed_special
        return [self.tokenizer.encode(t, allowed_special=allowed) for t in texts]
