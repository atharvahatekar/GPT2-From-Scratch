"""Dataset and DataLoader for next-token language-model pretraining.

Text is tokenized once, then chunked with a sliding window into overlapping
``(input, target)`` pairs where the target is the input shifted right by one
token. This is the standard GPT-style pretraining data format.
"""

from __future__ import annotations

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset


class GPTDataset(Dataset):
    """Sliding-window dataset of ``(input_ids, target_ids)`` token pairs.

    Parameters
    ----------
    txt:
        The raw text corpus.
    tokenizer:
        Any object exposing ``encode(text, allowed_special=...)`` — e.g. a
        :class:`~src.tokenizer.BPETokenizer` or a raw ``tiktoken`` encoding.
    max_length:
        Length of each input sequence (the context window).
    stride:
        Step size between consecutive windows. ``stride == max_length`` gives
        non-overlapping chunks; a smaller stride yields overlap.
    """

    def __init__(self, txt: str, tokenizer, max_length: int, stride: int):
        self.input_ids: list[torch.Tensor] = []
        self.target_ids: list[torch.Tensor] = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    txt: str,
    tokenizer=None,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Build a :class:`~torch.utils.data.DataLoader` over a :class:`GPTDataset`.

    If ``tokenizer`` is ``None``, the GPT-2 BPE encoding is used by default.
    """
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
