

import os
import urllib.request
import torch
import tiktoken
from pathlib import Path



device = (
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)



DATA_PATH = DATA_DIR / "shakespeare.txt"

def download_dataset():
    """Download TinyShakespeare if not already present."""
    DATA_DIR.mkdir(exist_ok=True)
    
    if os.path.exists(DATA_PATH):
        print(f"Dataset already exists at {DATA_PATH}")
        return
    
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print("Downloading TinyShakespeare...")
    urllib.request.urlretrieve(url, DATA_PATH)
    print(f" Downloaded to {DATA_PATH}")

download_dataset()

# Read it
with open(DATA_PATH, "r") as f:
    raw_text = f.read()

print(f"\nDataset loaded: {len(raw_text):,} characters")
print(f" Preview: {repr(raw_text[:100])}")


# ── STEP 2: TOKENIZER ───────────────────────────────────────
# CONCEPT: A tokenizer converts text ↔ integers.
#
# Why not just use characters (a=0, b=1, ...)?
#   → Vocabulary would be tiny (~65 chars). Model learns slowly.
#
# Why not use whole words?
#   → Vocabulary would be 500,000+. Too large.
#
# BPE (Byte Pair Encoding) finds the sweet spot:
#   → Common words = 1 token ("the" → 1234)
#   → Rare words = split into subwords ("unbelievable" → ["un","believ","able"])
#   → Vocabulary size = ~50,000–100,000 tokens
#
# tiktoken's "gpt2" encoding = same tokenizer as GPT-2.
# Vocab size = 50,257 tokens.

enc = tiktoken.get_encoding("gpt2")

print(f"\n Tokenizer: tiktoken GPT-2 BPE")
print(f"   Vocab size: {enc.n_vocab:,} tokens")

# Let's see it in action
sample = "What is gravity?"
tokens = enc.encode("Prathibha is learning about Transformers")
decoded = enc.decode(tokens)

print(f"\n   Text:    {repr(sample)}")
print(f"   Tokens:  {tokens}")
print(f"   Decoded: {repr(decoded)}")

# KEY INSIGHT: each token maps back to a string chunk
print(f"\n   Token breakdown:")
for tok_id in tokens:
    chunk = enc.decode([tok_id])
    print(f"     {tok_id:6d} → {repr(chunk)}")


# ── STEP 3: ENCODE THE ENTIRE DATASET ──────────────────────
# Convert all of Shakespeare into a 1D tensor of integers.
# This is literally what gets fed into the model eventually.

print(f"\n Encoding full dataset...")
all_tokens = enc.encode(raw_text)
data = torch.tensor(all_tokens, dtype=torch.long)

print(f" Encoded: {len(data):,} tokens")
print(f"   Shape: {data.shape}")
print(f"   First 10 tokens: {data[:10].tolist()}")
print(f"   Decoded back: {repr(enc.decode(data[:10].tolist()))}")


# ── STEP 4: TRAIN / VAL SPLIT ──────────────────────────────
# Standard practice: 90% train, 10% validation.
# Validation = data the model never trains on.
# We use it to check if the model is actually learning
# or just memorizing (overfitting).

split = int(0.9 * len(data))
train_data = data[:split]
val_data   = data[split:]

print(f"\nData split:")
print(f"   Train: {len(train_data):,} tokens ({len(train_data)/len(data)*100:.0f}%)")
print(f"   Val:   {len(val_data):,} tokens  ({len(val_data)/len(data)*100:.0f}%)")


# ── STEP 5: BATCH LOADER ────────────────────────────────────
# CONCEPT: What does the model actually receive?
#
# We feed it chunks of tokens called "context windows".
# Block size = how many tokens the model sees at once.
#
# Example with block_size=8:
#   Input:  [t1, t2, t3, t4, t5, t6, t7, t8]
#   Target: [t2, t3, t4, t5, t6, t7, t8, t9]
#
# The target is shifted by 1. At every position,
# the model predicts the NEXT token.
# This is "autoregressive" training.
#
# WHY THIS SHAPE:
#   One sequence actually contains 8 training examples!
#   Given [t1] → predict t2
#   Given [t1,t2] → predict t3
#   ...
#   Given [t1..t8] → predict t9
# This is extremely data-efficient.

BLOCK_SIZE = 256   # context window length (tokens seen at once)
BATCH_SIZE = 32    # how many sequences we process in parallel

def get_batch(split_name: str):
    """
    Returns a random batch of (input, target) tensors.
    
    Args:
        split_name: "train" or "val"
    
    Returns:
        x: (BATCH_SIZE, BLOCK_SIZE) — input token sequences
        y: (BATCH_SIZE, BLOCK_SIZE) — target token sequences (x shifted by 1)
    """
    source = train_data if split_name == "train" else val_data
    
    # Pick BATCH_SIZE random starting positions
    # Each position gives us a sequence of BLOCK_SIZE tokens
    ix = torch.randint(len(source) - BLOCK_SIZE, (BATCH_SIZE,))
    
    # Stack sequences into a batch matrix
    x = torch.stack([source[i : i + BLOCK_SIZE]     for i in ix])
    y = torch.stack([source[i+1 : i + BLOCK_SIZE+1] for i in ix])
    
    # Move to GPU (MPS on your MacBook)
    return x.to(device), y.to(device)


# ── TEST THE BATCH LOADER ───────────────────────────────────
print(f"\nTesting batch loader...")
x, y = get_batch("train")

print(f"   Input shape:  {x.shape}  -> (batch={BATCH_SIZE}, seq_len={BLOCK_SIZE})")
print(f"   Target shape: {y.shape}")
print(f"\n   First sequence (input tokens):")
print(f"   {x[0, :10].tolist()} ...")
print(f"\n   First sequence (target tokens — shifted by 1):")
print(f"   {y[0, :10].tolist()} ...")
print(f"\n   Decoded input:  {repr(enc.decode(x[0, :20].tolist()))}")
print(f"   Decoded target: {repr(enc.decode(y[0, :20].tolist()))}")
print(f"\n   Notice: target = input shifted right by 1 token")
