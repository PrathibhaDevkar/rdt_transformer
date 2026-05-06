

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import urllib.request
import os
from pathlib import Path

# ── DEVICE + HYPERPARAMETERS ────────────────────────────────
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"✅ Device: {device}\n")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

VOCAB_SIZE  = 50_257
BLOCK_SIZE  = 256
BATCH_SIZE  = 32
N_EMBED     = 384
N_HEADS     = 6
HEAD_SIZE   = N_EMBED // N_HEADS   # 64
N_LAYERS    = 4       #  NEW: how many transformer blocks to stack
DROPOUT     = 0.1
FF_MULT     = 4       # NEW: FFN expands to 4× N_EMBED internally

# ── QUICK DATA SETUP ────────────────────────────────────────
DATA_PATH = DATA_DIR / "shakespeare.txt"
def download_dataset():
    DATA_DIR.mkdir(exist_ok=True)
    if not os.path.exists(DATA_PATH):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, DATA_PATH)

download_dataset()
enc = tiktoken.get_encoding("gpt2")
with open(DATA_PATH) as f:
    raw_text = f.read()
data = torch.tensor(enc.encode(raw_text), dtype=torch.long)
split = int(0.9 * len(data))
train_data = data[:split]

def get_batch():
    ix = torch.randint(len(train_data) - BLOCK_SIZE, (BATCH_SIZE,))
    x  = torch.stack([train_data[i   : i+BLOCK_SIZE  ] for i in ix])
    y  = torch.stack([train_data[i+1 : i+BLOCK_SIZE+1] for i in ix])
    return x.to(device), y.to(device)

# ── COMPONENTS FROM PREVIOUS LESSONS ────────────────────────
class EmbeddingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb_table = nn.Embedding(VOCAB_SIZE, N_EMBED)
        self.pos_emb_table   = nn.Embedding(BLOCK_SIZE, N_EMBED)
    def forward(self, x):
        B, T = x.shape
        tok_emb = self.token_emb_table(x)
        pos_emb = self.pos_emb_table(torch.arange(T, device=x.device))
        return tok_emb + pos_emb

class SingleHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query   = nn.Linear(N_EMBED, head_size, bias=False)
        self.key     = nn.Linear(N_EMBED, head_size, bias=False)
        self.value   = nn.Linear(N_EMBED, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)
    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores  = q @ k.transpose(-2, -1) * (HEAD_SIZE ** -0.5)
        scores  = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        return weights @ v

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads   = nn.ModuleList([SingleHead(HEAD_SIZE) for _ in range(N_HEADS)])
        self.proj    = nn.Linear(N_EMBED, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

print("Previous components loaded\n")


# ════════════════════════════════════════════════════════════
# PART 1 — FEEDFORWARD NETWORK
# ════════════════════════════════════════════════════════════
# CONCEPT:
# After attention, each token has gathered context from others.
# Now each token needs to individually PROCESS that context.
#
# The FFN does this independently per token — it's the same
# network applied to each of the 256 positions separately.
# Tokens don't communicate here — they think alone.
#
# Structure:
#   Linear(384 → 1536)   ← expand to 4× width
#   GELU()               ← nonlinearity
#   Linear(1536 → 384)   ← compress back
#   Dropout()
#
# WHY 4× expansion?
# A wider middle layer can represent more complex functions.
# This is where most factual knowledge is believed to be stored.
# Specific facts ("Paris is the capital of France") are encoded
# in these FFN weights as key-value memories.
# The 4× ratio comes from the original "Attention Is All You Need"
# paper and has been used in almost every transformer since.
#
# WHY GELU and not ReLU?
# ReLU: f(x) = max(0, x)  — hard zero cutoff
# GELU: smoother, probabilistic version of ReLU
# GELU allows small negative values to pass through,
# giving better gradient flow and empirically better results
# in language models.

print("=" * 55)
print("PART 1: FeedForward Network")
print("=" * 55)

class FeedForward(nn.Module):
    """
    Position-wise FeedForward Network.
    Applied independently to each token position.
    
    Input:  (B, T, N_EMBED)
    Output: (B, T, N_EMBED)   ← same shape
    """
    
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            # Expand: 384 → 1536
            nn.Linear(n_embed, FF_MULT * n_embed),
            
            # GELU nonlinearity — introduces non-linear
            # transformation so the network can learn
            # complex functions, not just linear ones
            nn.GELU(),
            
            # Compress: 1536 → 384
            nn.Linear(FF_MULT * n_embed, n_embed),
            
            # Dropout for regularization
            nn.Dropout(DROPOUT),
        )
    
    def forward(self, x):
        # x: (B, T, N_EMBED)
        # self.net applied to last dimension only
        # Each token processed independently
        return self.net(x)   # (B, T, N_EMBED)

# ── Test FFN ─────────────────────────────────────────────────
ffn      = FeedForward(N_EMBED).to(device)
emb_layer = EmbeddingLayer().to(device)
mha      = MultiHeadAttention().to(device)

x, y   = get_batch()
emb    = emb_layer(x)          # (32, 256, 384)
attn   = mha(emb)              # (32, 256, 384)
ff_out = ffn(attn)             # (32, 256, 384)

print(f"\nInput to FFN:  {attn.shape}")
print(f"Output of FFN: {ff_out.shape}  ← same shape")

ffn_params = sum(p.numel() for p in ffn.parameters())
print(f"\nFFN parameters: {ffn_params:,}")
print(f"  Layer 1: {N_EMBED} × {FF_MULT*N_EMBED} = {N_EMBED * FF_MULT*N_EMBED:,}")
print(f"  Layer 2: {FF_MULT*N_EMBED} × {N_EMBED} = {FF_MULT*N_EMBED * N_EMBED:,}")

# ── Visualize GELU vs ReLU ───────────────────────────────────
print(f"\nGELU vs ReLU comparison:")
test_vals = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
gelu_out  = F.gelu(test_vals)
relu_out  = F.relu(test_vals)
print(f"  Input: {[f'{v:.1f}' for v in test_vals.tolist()]}")
print(f"  GELU:  {[f'{v:.3f}' for v in gelu_out.tolist()]}")
print(f"  ReLU:  {[f'{v:.3f}' for v in relu_out.tolist()]}")
print(f"\n  Notice: GELU allows small negative values through")
print(f"  (-0.5 → -0.154 in GELU vs 0.0 in ReLU)")


# ════════════════════════════════════════════════════════════
# PART 2 — LAYER NORMALIZATION
# ════════════════════════════════════════════════════════════
# CONCEPT:
# As data flows through many layers, activations can drift —
# becoming very large or very small. This makes training
# unstable and slow.
#
# Layer Norm fixes this by normalizing each token's vector
# to have mean=0 and std=1, then applying learned scale (γ)
# and shift (β) parameters.
#
# For each token vector of 384 values:
#   1. Compute mean μ and std σ across those 384 values
#   2. Normalize: x̂ = (x - μ) / σ
#   3. Scale and shift: output = γ × x̂ + β
#
# γ and β are learned — the model decides how much to
# rescale after normalization.
#
# WHY LAYER NORM and not BATCH NORM?
# Batch Norm normalizes across the batch dimension.
# That causes problems at inference when batch size = 1.
# Layer Norm normalizes across the feature dimension (384),
# so it works identically regardless of batch size.
#
# WHERE WE APPLY IT:
# We use "Pre-LN" — normalize BEFORE attention and FFN.
# This is more stable than the original "Post-LN" design.

print("\n" + "=" * 55)
print("PART 2: Layer Normalization")
print("=" * 55)

# PyTorch's built-in LayerNorm
# normalizes over the last dimension (N_EMBED = 384)
ln = nn.LayerNorm(N_EMBED).to(device)

# Before normalization
print(f"\nBefore LayerNorm:")
print(f"  Mean: {emb[0,0].mean().item():.4f}")
print(f"  Std:  {emb[0,0].std().item():.4f}")

# After normalization
normed = ln(emb)
print(f"\nAfter LayerNorm:")
print(f"  Mean: {normed[0,0].mean().item():.4f}  ← close to 0")
print(f"  Std:  {normed[0,0].std().item():.4f}   ← close to 1")
print(f"\nShape unchanged: {normed.shape}")
print(f"LayerNorm parameters: γ={ln.weight.shape}, β={ln.bias.shape}")
print(f"  (just 384 scale + 384 shift = 768 learned params)")


# ════════════════════════════════════════════════════════════
# PART 3 — TRANSFORMER BLOCK
# ════════════════════════════════════════════════════════════
# CONCEPT:
# One complete transformer block = Attention + FFN
# with Layer Norm and Residual Connections.
#
# RESIDUAL CONNECTIONS — the most important design choice:
#   x = x + attention(layernorm(x))
#   x = x + ffn(layernorm(x))
#
# We ADD the output back to the input instead of replacing it.
# WHY? Two reasons:
#
# 1. Gradient flow: during backprop, gradients can flow
#    directly through the + sign without passing through
#    attention or FFN. This prevents vanishing gradients
#    in deep networks (many layers).
#
# 2. Information preservation: the original token information
#    is never destroyed — attention and FFN only ADD to it.
#    If a layer learns nothing useful, the residual ensures
#    the signal passes through unchanged.
#
# This is why deep transformers (100+ layers) can be trained
# at all. Without residuals, gradients would vanish.

print("\n" + "=" * 55)
print("PART 3: Transformer Block = Attention + FFN + Residuals")
print("=" * 55)

class TransformerBlock(nn.Module):
    """
    One complete transformer block.
    
    Contains:
      - Layer Norm before attention (Pre-LN)
      - Multi-Head Self-Attention with residual
      - Layer Norm before FFN (Pre-LN)  
      - FeedForward Network with residual
    
    Input:  (B, T, N_EMBED)
    Output: (B, T, N_EMBED)   ← same shape always
    """
    
    def __init__(self):
        super().__init__()
        self.ln1  = nn.LayerNorm(N_EMBED)   # before attention
        self.attn = MultiHeadAttention()
        self.ln2  = nn.LayerNorm(N_EMBED)   # before FFN
        self.ffn  = FeedForward(N_EMBED)
    
    def forward(self, x):
        # Attention with residual connection
        # x + ... means we ADD output back to input
        x = x + self.attn(self.ln1(x))
        
        # FFN with residual connection
        x = x + self.ffn(self.ln2(x))
        
        return x   # (B, T, N_EMBED)

# ── Test one block ───────────────────────────────────────────
block     = TransformerBlock().to(device)
block_out = block(emb)

print(f"\nInput:  {emb.shape}")
print(f"Output: {block_out.shape}   ← same shape")

block_params = sum(p.numel() for p in block.parameters())
print(f"\nParameters in one TransformerBlock: {block_params:,}")
print(f"  Attention:    {sum(p.numel() for p in block.attn.parameters()):,}")
print(f"  FFN:          {sum(p.numel() for p in block.ffn.parameters()):,}")
print(f"  Layer Norms:  {sum(p.numel() for p in block.ln1.parameters()) + sum(p.numel() for p in block.ln2.parameters()):,}")


# ════════════════════════════════════════════════════════════
# PART 4 — STACK N_LAYERS BLOCKS
# ════════════════════════════════════════════════════════════
# CONCEPT:
# The full transformer backbone is just N_LAYERS blocks
# stacked one after another.
#
# Each block refines token representations further:
#   Block 1: low-level patterns (punctuation, common words)
#   Block 2: grammar and syntax
#   Block 3: semantic relationships
#   Block 4: higher-level reasoning
#
# The output of each block feeds into the next.
# All blocks have the same shape (B, T, N_EMBED) in and out.
# This is why stacking works cleanly.

print("\n" + "=" * 55)
print("PART 4: Stacking N_LAYERS Blocks")
print("=" * 55)

class TransformerBackbone(nn.Module):
    """
    Full transformer backbone: embedding + N stacked blocks.
    
    Input:  (B, T) integer token IDs
    Output: (B, T, N_EMBED) rich token representations
    """
    
    def __init__(self):
        super().__init__()
        self.embedding = EmbeddingLayer()
        
        # Stack N_LAYERS transformer blocks
        self.blocks = nn.Sequential(
            *[TransformerBlock() for _ in range(N_LAYERS)]
        )
        
        # Final layer norm after all blocks
        self.ln_final = nn.LayerNorm(N_EMBED)
    
    def forward(self, x):
        # x: (B, T)
        out = self.embedding(x)    # (B, T, N_EMBED)
        out = self.blocks(out)     # (B, T, N_EMBED) through N layers
        out = self.ln_final(out)   # (B, T, N_EMBED)
        return out

# ── Test backbone ────────────────────────────────────────────
backbone      = TransformerBackbone().to(device)
backbone_out  = backbone(x)

print(f"\nInput:  {x.shape}           ← integer token IDs")
print(f"Output: {backbone_out.shape}  ← rich representations")

total_params = sum(p.numel() for p in backbone.parameters())
print(f"\nTotal backbone parameters: {total_params:,}")
print(f"  Embedding:  {sum(p.numel() for p in backbone.embedding.parameters()):,}")
print(f"  {N_LAYERS} Blocks:     {sum(p.numel() for p in backbone.blocks.parameters()):,}")
print(f"  Final LN:   {sum(p.numel() for p in backbone.ln_final.parameters()):,}")

# Show how representations change through layers
print(f"\nHow representations evolve through layers:")
with torch.no_grad():
    h = backbone.embedding(x)
    print(f"  After embedding: mean={h.mean().item():.4f}, std={h.std().item():.4f}")
    for i, block in enumerate(backbone.blocks):
        h = block(h)
        print(f"  After block {i+1}:   mean={h.mean().item():.4f}, std={h.std().item():.4f}")


