"""Generate text from a trained checkpoint.

Usage
-----
    python scripts/generate.py --checkpoint model.pth --prompt "Every effort" \
        --max-new-tokens 50 --temperature 1.0 --top-k 25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tiktoken
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import GPTModel, get_config  # noqa: E402
from src.generation import generate, text_to_token_ids, token_ids_to_text  # noqa: E402
from src.utils import load_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="model.pth")
    parser.add_argument("--model", type=str, default="gpt2-small (124M)")
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--prompt", type=str, default="Every effort moves you")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")

    cfg = get_config(args.model, context_length=args.context_length)
    model = GPTModel(cfg).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    idx = text_to_token_ids(args.prompt, tokenizer).to(device)
    token_ids = generate(
        model=model,
        idx=idx,
        max_new_tokens=args.max_new_tokens,
        context_size=cfg["context_length"],
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(token_ids_to_text(token_ids, tokenizer))


if __name__ == "__main__":
    main()
