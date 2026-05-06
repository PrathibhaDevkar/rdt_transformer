

import torch
import torch.nn as nn
import tiktoken
import urllib.request
import os
from pathlib import Path

# ── DEVICE ──────────────────────────────────────────────────
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ── HYPERPARAMETERS ─────────────────────────────────────────
# These are the knobs that define your model's size.
# We'll use small values now so everything runs fast.
# In Phase 2 (RDT) we'll scale these up.

VOCAB_SIZE  = 50_257   # fixed — tiktoken GPT-2 vocabulary
BLOCK_SIZE  = 256      # context window (tokens seen at once)
BATCH_SIZE  = 32       # sequences per batch
N_EMBED     = 384      #  NEW: size of each embedding vector

# N_EMBED is the most important new number today.
# Every token will become a vector of 384 floats.
# Larger = more expressive, but slower.
# GPT-2 small uses 768. We use 384 for speed.

# ── QUICK DATA SETUP (from Lesson 1) ────────────────────────
DATA_PATH = DATA_DIR / "shakespeare.txt"

def download_dataset():
    DATA_DIR.mkdir(exist_ok=True)
    if not os.path.exists(DATA_PATH):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, DATA_PATH)
    
download_dataset()

enc = tiktoken.get_encoding("gpt2")
with open(DATA_PATH) as f:
    raw_text = f.read()

data  = torch.tensor(enc.encode(raw_text), dtype=torch.long)
split = int(0.9 * len(data))
train_data = data[:split]

def get_batch():
    ix = torch.randint(len(train_data) - BLOCK_SIZE, (BATCH_SIZE,))
    x  = torch.stack([train_data[i   : i+BLOCK_SIZE  ] for i in ix])
    y  = torch.stack([train_data[i+1 : i+BLOCK_SIZE+1] for i in ix])
    return x.to(device), y.to(device)

print("Data ready\n")


# ════════════════════════════════════════════════════════════
# PART 1 — TOKEN EMBEDDING
# ════════════════════════════════════════════════════════════
# CONCEPT:
# nn.Embedding is just a lookup table.
# It's a matrix of shape (VOCAB_SIZE, N_EMBED).
# Each row = one token's vector representation.
#
# When you pass token ID 5962, it returns row 5962.
# That's it. No math — just a table lookup.
#
#   token_id=5962 → embedding_table[5962] → [0.23, -0.87, ...]
#
# The magic: these vectors are LEARNED via backprop.
# They start random. After training, similar tokens
# (e.g. "king" and "queen") end up with similar vectors.
# That's the model learning semantic meaning.

print("=" * 55)
print("PART 1: Token Embedding")
print("=" * 55)

# Create the embedding layer
# This creates a (50257, 384) matrix of learnable parameters
token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBED).to(device)

print(f"\nEmbedding table shape: {token_embedding_table.weight.shape}")
print(f"  -> {VOCAB_SIZE:,} tokens × {N_EMBED} dimensions each")
print(f"  -> Total parameters: {VOCAB_SIZE * N_EMBED:,}")

# Get a batch of token IDs
x, y = get_batch()
print(f"\nInput x shape:  {x.shape}  -> (batch={BATCH_SIZE}, seq_len={BLOCK_SIZE})")

# Pass through embedding layer
# Each integer in x gets replaced by its N_EMBED-dim vector
token_emb = token_embedding_table(x)
print(f"After embedding: {token_emb.shape}  -> (batch={BATCH_SIZE}, seq_len={BLOCK_SIZE}, n_embed={N_EMBED})")

# ── Let's see what one token looks like ─────────────────────
first_token_id = x[0, 0].item()
first_token_vec = token_emb[0, 0]
print(f"\nFirst token in batch:")
print(f"  ID:     {first_token_id}")
print(f"  Text:   {repr(enc.decode([first_token_id]))}")
print(f"  Vector: [{first_token_vec[0]:.4f}, {first_token_vec[1]:.4f}, {first_token_vec[2]:.4f}, ... (384 values)]")
print(f"\n KEY INSIGHT: The model will adjust these 384 values")
print(f"   during training until they encode meaning.")


# ════════════════════════════════════════════════════════════
# PART 2 — POSITIONAL EMBEDDING
# ════════════════════════════════════════════════════════════
# CONCEPT:
# Here's a problem with pure token embeddings:
#
#   "dog bites man"  -> [dog_vec, bites_vec, man_vec]
#   "man bites dog"  -> [man_vec, bites_vec, dog_vec]
#
# The TOKEN embeddings are identical in both sentences!
# The model can't tell which word came first.
#
# Transformers process all tokens simultaneously (in parallel).
# Unlike RNNs, there's no inherent sense of order.
# So we need to INJECT position information.
#
# Solution: learn a separate embedding for each POSITION.
# Position 0 gets its own vector. Position 1 gets its own.
# All the way up to BLOCK_SIZE-1.
#
# These position vectors get ADDED to token vectors.
# The model learns to use this positional signal to
# understand word order and syntax.

print("\n" + "=" * 55)
print("PART 2: Positional Embedding")
print("=" * 55)

# Create positional embedding table
# Shape: (BLOCK_SIZE, N_EMBED)
# One vector per possible position (0 to 255)
position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED).to(device)

print(f"\nPositional embedding table shape: {position_embedding_table.weight.shape}")
print(f"  -> {BLOCK_SIZE} positions × {N_EMBED} dimensions each")

# Create a tensor of position indices: [0, 1, 2, ..., 255]
positions = torch.arange(BLOCK_SIZE, device=device)  # shape: (256,)
print(f"\nPosition indices shape: {positions.shape}")
print(f"Position indices: [{positions[0].item()}, {positions[1].item()}, ..., {positions[-1].item()}]")

# Look up position embeddings
pos_emb = position_embedding_table(positions)   # (BLOCK_SIZE, N_EMBED)
print(f"Position embeddings shape: {pos_emb.shape}  → (seq_len={BLOCK_SIZE}, n_embed={N_EMBED})")


# ════════════════════════════════════════════════════════════
# PART 3 — COMBINE THEM
# ════════════════════════════════════════════════════════════
# CONCEPT:
# The final input to the transformer is simply:
#
#   x = token_embedding + positional_embedding
#
# We ADD them together elementwise.
# This works because both are N_EMBED-dimensional.
# The result encodes BOTH what the token is AND where it is.
#
# This is a design choice (not the only option),
# but it's what GPT-2 and most transformers use.
# The model learns to disentangle the two signals.

print("\n" + "=" * 55)
print("PART 3: Combine Token + Position Embeddings")
print("=" * 55)

# pos_emb is (256, 384) — same for every sequence in the batch
# token_emb is (32, 256, 384) — different per sequence
# PyTorch broadcasting automatically handles the addition:
# pos_emb gets "broadcast" across the batch dimension

x_combined = token_emb + pos_emb   # shape: (32, 256, 384)

print(f"\nToken embedding shape:    {token_emb.shape}")
print(f"Position embedding shape: {pos_emb.shape}  <- broadcast across batch")
print(f"Combined shape: {x_combined.shape}")
print(f"\n  (batch=32, seq_len=256, n_embed=384)")
print(f"\n This (32, 256, 384) tensor is what feeds into")
print(f"   the attention layer in Lesson 3.")


# ════════════════════════════════════════════════════════════
# PART 4 — BUILD IT AS A PROPER nn.MODULE
# ════════════════════════════════════════════════════════════
# CONCEPT:
# In PyTorch, models are built as classes that inherit nn.Module.
# This gives you:
#   - Automatic parameter tracking (for backprop)
#   - .parameters() to pass to optimizer
#   - .to(device) to move everything to GPU at once
#   - Clean, reusable structure
#
# Every component of our transformer will be an nn.Module.
# Here we package the embedding logic into one clean class.

print("\n" + "=" * 55)
print("PART 4: Clean nn.Module Implementation")
print("=" * 55)

class EmbeddingLayer(nn.Module):
    """
    Converts token IDs into combined token+position embeddings.
    
    Input:  x of shape (batch, seq_len) — integer token IDs
    Output: embeddings of shape (batch, seq_len, n_embed)
    """
    
    def __init__(self, vocab_size, block_size, n_embed):
        super().__init__()
        # Two learned lookup tables
        self.token_emb_table = nn.Embedding(vocab_size, n_embed)
        self.pos_emb_table   = nn.Embedding(block_size, n_embed)
        self.block_size      = block_size
    
    def forward(self, x):
        # x: (batch, seq_len)
        batch, seq_len = x.shape
        
        # Token embeddings — different per token ID
        tok_emb = self.token_emb_table(x)                              # (batch, seq_len, n_embed)
        
        # Position embeddings — same for every sequence in batch
        positions = torch.arange(seq_len, device=x.device)            # (seq_len,)
        pos_emb   = self.pos_emb_table(positions)                      # (seq_len, n_embed)
        
        # Add them — broadcasting handles the batch dimension
        return tok_emb + pos_emb                                       # (batch, seq_len, n_embed)

# Instantiate and test
embedding_layer = EmbeddingLayer(VOCAB_SIZE, BLOCK_SIZE, N_EMBED).to(device)

x, y = get_batch()
out  = embedding_layer(x)

print(f"\nInput shape:  {x.shape}")
print(f"Output shape: {out.shape}")
print(f"\nLearnable parameters in EmbeddingLayer:")
for name, param in embedding_layer.named_parameters():
    print(f"  {name:30s} → {param.shape}  ({param.numel():,} params)")

total_params = sum(p.numel() for p in embedding_layer.parameters())
print(f"\nTotal embedding parameters: {total_params:,}")


