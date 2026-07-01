# LLM From Scratch

A ground-up, modular implementation of a **GPT-style Large Language Model** in
PyTorch — no high-level ML frameworks for the core model logic, only PyTorch
primitives and NumPy. Every component (tokenizer, attention, transformer blocks,
training loop, text generation) is written from scratch and organized as a clean,
importable Python package under [`src/`](src/).

The model is designed to be trained on the Harry Potter book series, but works on
any plain-text corpus.

---

## Project Structure

```
src/
├── config.py                 # GPT-2 hyperparameter presets + get_config()
├── tokenizer/
│   ├── simple_tokenizer.py   # Rule-based tokenizers (V1, V2) built from scratch
│   └── bpe_tokenizer.py      # BPE tokenizer wrapping tiktoken (GPT-2)
├── data/
│   └── dataset.py            # Sliding-window GPTDataset + create_dataloader()
├── model/
│   ├── attention.py          # Multi-head causal self-attention
│   ├── layers.py             # LayerNorm, GELU, FeedForward
│   ├── transformer.py        # TransformerBlock (pre-norm + residuals)
│   └── gpt.py                # GPTModel (full architecture)
├── generation/
│   └── generate.py           # Greedy + top-k/temperature sampling
├── training/
│   ├── losses.py             # Cross-entropy loss helpers
│   └── trainer.py            # Pretraining loop + evaluation
└── utils/
    ├── checkpoint.py         # Save/load model + optimizer state
    ├── plotting.py           # Loss curve plotting
    └── gpt2_weights.py       # Load pretrained OpenAI GPT-2 weights

scripts/
├── train.py                  # End-to-end pretraining CLI
└── generate.py               # Text generation CLI

learning-code-files/          # Original step-by-step learning notebooks
Notebooks/                    # Cleaned-up notebooks per component
```

---

## Setup

```bash
git clone https://github.com/your-username/LLM-From-Scratch.git
cd LLM-From-Scratch
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

---

## Quick Start

### Build and run the model

```python
import torch
from src import GPTModel, get_config, BPETokenizer, create_dataloader
from src import generate_text_simple, text_to_token_ids, token_ids_to_text

# 1. Configure a model (GPT-2 small by default; override any hyperparameter)
cfg = get_config("gpt2-small (124M)", context_length=256)
model = GPTModel(cfg)
print(f"Parameters: {model.num_params():,}")

# 2. Tokenize + build a dataloader with a sliding window
tokenizer = BPETokenizer()
text = open("data/the-verdict.txt", encoding="utf-8").read()
loader = create_dataloader(text, tokenizer=tokenizer,
                           batch_size=4, max_length=256, stride=128)

# 3. Generate text
idx = text_to_token_ids("Every effort moves you", tokenizer)
out = generate_text_simple(model, idx, max_new_tokens=10,
                           context_size=cfg["context_length"])
print(token_ids_to_text(out, tokenizer))
```

### Train from the command line

```bash
python scripts/train.py --data data/the-verdict.txt --epochs 10 --context-length 256
```

### Generate from a checkpoint

```bash
python scripts/generate.py --checkpoint model.pth --prompt "Every effort" \
    --max-new-tokens 50 --temperature 1.0 --top-k 25
```

---

## How the Pipeline Works

```
Raw text  →  BPETokenizer.encode()  →  token IDs
                                           ↓
                          GPTDataset (sliding window)
                       [tok_0 … tok_N]     ← input
                       [tok_1 … tok_N+1]   ← target (shifted by 1)
                                           ↓
                          DataLoader → GPTModel → logits
                                           ↓
                          cross-entropy loss → AdamW → repeat
```

The **GPTModel** stacks token + positional embeddings, `n_layers`
`TransformerBlock`s (each = pre-norm multi-head causal attention + feed-forward
with residual connections), a final LayerNorm, and an output projection to
vocabulary logits.

---

## Using Pretrained GPT-2 Weights

The architecture is weight-compatible with OpenAI's GPT-2. Given the `params`
dictionary from the reference `download_and_load_gpt2` loader:

```python
from src import GPTModel, get_config
from src.utils import load_weights_into_gpt

cfg = get_config("gpt2-small (124M)", context_length=1024, qkv_bias=True)
model = GPTModel(cfg)
load_weights_into_gpt(model, params)   # copies pretrained weights in place
```

---

## Project Roadmap

| Stage | Topic | Status |
|-------|-------|--------|
| 1 | Tokenization, Embeddings & Data Pipeline | ✅ |
| 2 | Attention Mechanism (Scaled Dot-Product, Multi-Head) | ✅ |
| 3 | GPT Architecture (Transformer Blocks, LayerNorm, Feed-Forward) | ✅ |
| 4 | Pre-training & Text Generation | ✅ |

---

## Dependencies

| Library | Purpose |
|---------|---------|
| `torch` | Tensor ops, Dataset/DataLoader, model layers |
| `tiktoken` | GPT-2 BPE tokenizer (OpenAI) |
| `numpy` | Numerical utilities, weight loading |
| `matplotlib` | Training loss visualization |
| `regex` | Advanced pattern matching for rule-based tokenizer |
