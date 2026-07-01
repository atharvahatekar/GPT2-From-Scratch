"""End-to-end pretraining script for the from-scratch GPT model.

Reads a text corpus, splits it into train/validation dataloaders, trains a
``GPTModel``, plots the losses, and saves a checkpoint.

Usage
-----
    python scripts/train.py --data data/the-verdict.txt --epochs 10

Run with ``--help`` to see all options.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import tiktoken
import torch

# Allow running as a plain script (``python scripts/train.py``) by making the
# repository root importable.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import GPTModel, create_dataloader, get_config  # noqa: E402
from src.training import train_model_simple  # noqa: E402
from src.utils import plot_losses, save_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain a GPT model from scratch.")
    parser.add_argument("--data", type=str, default="data/the-verdict.txt",
                        help="Path to the training text file.")
    parser.add_argument("--model", type=str, default="gpt2-small (124M)",
                        help="Model size preset (see src.config.GPT2_MODEL_CONFIGS).")
    parser.add_argument("--context-length", type=int, default=256,
                        help="Context window length (overrides the preset).")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--stride", type=int, default=None,
                        help="Sliding-window stride (defaults to context length).")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--train-frac", type=float, default=0.9)
    parser.add_argument("--eval-freq", type=int, default=5)
    parser.add_argument("--eval-iter", type=int, default=5)
    parser.add_argument("--start-context", type=str, default="Every effort moves you")
    parser.add_argument("--checkpoint", type=str, default="model.pth")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    text = Path(args.data).read_text(encoding="utf-8")
    tokenizer = tiktoken.get_encoding("gpt2")

    # Shrink the context length so the demo trains quickly on small corpora.
    cfg = get_config(args.model, context_length=args.context_length)
    stride = args.stride or args.context_length

    # Train / validation split on the raw text.
    split_idx = int(args.train_frac * len(text))
    train_text, val_text = text[:split_idx], text[split_idx:]

    train_loader = create_dataloader(
        train_text, tokenizer=tokenizer, batch_size=args.batch_size,
        max_length=cfg["context_length"], stride=stride,
        shuffle=True, drop_last=True,
    )
    val_loader = create_dataloader(
        val_text, tokenizer=tokenizer, batch_size=args.batch_size,
        max_length=cfg["context_length"], stride=stride,
        shuffle=False, drop_last=False,
    )

    model = GPTModel(cfg).to(device)
    print(f"Model parameters: {model.num_params():,}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    start = time.time()
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=args.epochs, eval_freq=args.eval_freq, eval_iter=args.eval_iter,
        start_context=args.start_context, tokenizer=tokenizer,
    )
    print(f"Training completed in {(time.time() - start) / 60:.2f} minutes.")

    save_checkpoint(args.checkpoint, model, optimizer)
    print(f"Saved checkpoint to {args.checkpoint}")

    if train_losses:
        epochs_tensor = torch.linspace(0, args.epochs, len(train_losses))
        plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses,
                    save_path="loss-plot.pdf")


if __name__ == "__main__":
    main()
