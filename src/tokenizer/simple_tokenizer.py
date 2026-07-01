"""Rule-based tokenizers built from scratch with regular expressions.

These are the pedagogical tokenizers from the first stage of the project. They
split text on whitespace and common punctuation and map tokens to integer ids.
:class:`SimpleTokenizerV2` additionally handles out-of-vocabulary tokens with a
``<|unk|>`` placeholder.
"""

from __future__ import annotations

import re

# Splits on punctuation, the em-dash ("--"), and whitespace, keeping the
# delimiters so punctuation becomes its own token.
_SPLIT_PATTERN = r'([,.:;?_!"()\']|--|\s)'

UNK_TOKEN = "<|unk|>"
ENDOFTEXT_TOKEN = "<|endoftext|>"


def _tokenize(text: str) -> list[str]:
    """Split ``text`` into a list of non-empty, stripped tokens."""
    return [tok.strip() for tok in re.split(_SPLIT_PATTERN, text) if tok.strip()]


def build_vocab(text: str, special_tokens: tuple[str, ...] = ()) -> dict[str, int]:
    """Build a ``token -> id`` vocabulary from ``text``.

    ``special_tokens`` are appended after the sorted corpus tokens so they always
    receive the highest ids.
    """
    tokens = sorted(set(_tokenize(text)))
    tokens.extend(t for t in special_tokens if t not in tokens)
    return {tok: idx for idx, tok in enumerate(tokens)}


class SimpleTokenizerV1:
    """Minimal regex tokenizer with ``encode`` / ``decode`` support.

    Accepts either a prebuilt ``token -> id`` vocabulary or a raw text string
    (from which a vocabulary is built automatically).
    """

    def __init__(self, vocab_or_text: dict[str, int] | str):
        if isinstance(vocab_or_text, str):
            self.str_to_int = build_vocab(vocab_or_text)
        else:
            self.str_to_int = dict(vocab_or_text)
        self.int_to_str = {i: s for s, i in self.str_to_int.items()}

    def encode(self, text: str) -> list[int]:
        return [self.str_to_int[s] for s in _tokenize(text)]

    def decode(self, ids: list[int]) -> str:
        text = " ".join(self.int_to_str[i] for i in ids)
        # Remove the space that " ".join inserts before punctuation.
        return re.sub(r'\s+([,.?!"()\'])', r"\1", text)


class SimpleTokenizerV2(SimpleTokenizerV1):
    """Like :class:`SimpleTokenizerV1` but maps unknown tokens to ``<|unk|>``.

    Ensures ``<|unk|>`` exists in the vocabulary so that any token outside the
    training corpus can still be encoded.
    """

    def __init__(self, vocab_or_text: dict[str, int] | str):
        super().__init__(vocab_or_text)
        if UNK_TOKEN not in self.str_to_int:
            self.str_to_int[UNK_TOKEN] = max(self.str_to_int.values(), default=-1) + 1
            self.int_to_str = {i: s for s, i in self.str_to_int.items()}

    def encode(self, text: str) -> list[int]:
        tokens = [
            tok if tok in self.str_to_int else UNK_TOKEN for tok in _tokenize(text)
        ]
        return [self.str_to_int[s] for s in tokens]
