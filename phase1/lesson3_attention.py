

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import urllib.request
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ── DEVICE + HYPERPARAMETERS ────────────────────────────────
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"✅ Device: {device}")

VOCAB_SIZE  = 50_257
BLOCK_SIZE  = 256
BATCH_SIZE  = 32
N_EMBED     = 384
N_HEADS     = 6       # ← NEW: number of parallel attention heads
HEAD_SIZE   = N_EMBED // N_HEADS  # 384 // 6 = 64 per head
DROPOUT     = 0.1     # ← NEW: randomly zero out 10% of activations

# HEAD_SIZE is important:
# We split N_EMBED evenly across all heads.
# Each head works in a 64-dim subspace independently.
# All heads run in parallel, outputs get concatenated.
print(f"\nN_EMBED={N_EMBED}, N_HEADS={N_HEADS}, HEAD_SIZE={HEAD_SIZE}")
print(f"Each head works in a {HEAD_SIZE}-dim subspace\n")

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

# ── EMBEDDING LAYER (from Lesson 2) ─────────────────────────
class EmbeddingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb_table = nn.Embedding(VOCAB_SIZE, N_EMBED)
        self.pos_emb_table   = nn.Embedding(BLOCK_SIZE, N_EMBED)
    def forward(self, x):
        B, T = x.shape
        tok_emb = self.token_emb_table(x)
        pos_emb = self.pos_emb_table(torch.arange(T, device=x.device))
        return tok_emb + pos_emb   # (B, T, N_EMBED)

print("✅ Data + Embedding ready\n")


# ════════════════════════════════════════════════════════════
# PART 1 — SINGLE ATTENTION HEAD
# ════════════════════════════════════════════════════════════
# CONCEPT RECAP:
# Every token asks "what am I looking for?" (Query)
# Every token says "here's what I contain" (Key)
# Every token offers "here's my actual info" (Value)
#
# Attention score between token i and token j:
#   score(i,j) = Q_i · K_j / sqrt(HEAD_SIZE)
#
# After softmax: these become attention weights (sum to 1)
# Final output: weighted sum of all Value vectors

print("=" * 55)
print("PART 1: Single Attention Head")
print("=" * 55)

class SingleHead(nn.Module):
    """
    One attention head.
    
    Takes embeddings of shape (B, T, N_EMBED)
    Returns attended output of shape (B, T, HEAD_SIZE)
    """
    
    def __init__(self, head_size):
        super().__init__()
        # Three linear projections — no bias needed here
        # Each takes N_EMBED → head_size
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.key   = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        
        # Causal mask — upper triangle of -inf
        # register_buffer = not a parameter (not learned),
        # but moves to GPU with the model automatically
        self.register_buffer(
            'tril',
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        )
        
        self.dropout = nn.Dropout(DROPOUT)
    
    def forward(self, x):
        B, T, C = x.shape   # C = N_EMBED = 384
        
        # ── Step 1: Project to Q, K, V ──────────────────────
        # Each token's 384-dim vector gets projected to
        # three different head_size-dim vectors
        q = self.query(x)   # (B, T, head_size)
        k = self.key(x)     # (B, T, head_size)
        v = self.value(x)   # (B, T, head_size)
        
        # ── Step 2: Compute attention scores ────────────────
        # QK^T: for every pair (i,j), dot product of Q_i and K_j
        # @ is matrix multiply in PyTorch
        # k.transpose(-2,-1) flips the last two dims: (B,T,hs)→(B,hs,T)
        scores = q @ k.transpose(-2, -1)   # (B, T, T)
        
        # Scale by 1/sqrt(head_size)
        # WHY: without scaling, dot products grow large as head_size
        # increases, pushing softmax into regions of tiny gradients.
        # Dividing by sqrt(d) keeps variance stable.
        scores = scores * (HEAD_SIZE ** -0.5)  # (B, T, T)
        
        # ── Step 3: Causal mask ─────────────────────────────
        # Set future positions to -inf so softmax makes them 0
        # tril is lower triangular: 1s where we CAN attend, 0s where we can't
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # ── Step 4: Softmax → attention weights ─────────────
        # Each row sums to 1.0
        # Each row = "how much does token i attend to token j?"
        weights = F.softmax(scores, dim=-1)   # (B, T, T)
        weights = self.dropout(weights)
        
        # ── Step 5: Weighted sum of Values ──────────────────
        out = weights @ v   # (B, T, T) @ (B, T, head_size) = (B, T, head_size)
        
        return out

# ── Test single head ────────────────────────────────────────
embedding_layer = EmbeddingLayer().to(device)
single_head     = SingleHead(HEAD_SIZE).to(device)

x, y  = get_batch()
emb   = embedding_layer(x)           # (32, 256, 384)
h_out = single_head(emb)             # (32, 256, 64)

print(f"\nInput to head:  {emb.shape}   → (B, T, N_EMBED)")
print(f"Output of head: {h_out.shape}  → (B, T, HEAD_SIZE)")
print(f"\n💡 Each token now contains a weighted blend of")
print(f"   information from all PREVIOUS tokens")

# ── Visualize attention weights for one head ─────────────────
with torch.no_grad():
    q = single_head.query(emb[0:1])      # (1, T, head_size)
    k = single_head.key(emb[0:1])        # (1, T, head_size)
    scores  = (q @ k.transpose(-2,-1)) * (HEAD_SIZE**-0.5)
    scores  = scores.masked_fill(single_head.tril[:256,:256]==0, float('-inf'))
    weights = F.softmax(scores, dim=-1)  # (1, T, T)

print(f"\nAttention weights shape: {weights.shape}  → (1, T, T)")
print(f"\nFor token at position 10 — attention distribution:")
print(f"  (how much it attends to positions 0-10)")
top_weights = weights[0, 10, :11]
for pos, w in enumerate(top_weights.tolist()):
    bar = "█" * int(w * 40)
    tok = enc.decode([x[0, pos].item()])
    print(f"  pos {pos:2d} {repr(tok):12s} {bar} {w:.3f}")


# ════════════════════════════════════════════════════════════
# PART 2 — MULTI-HEAD ATTENTION
# ════════════════════════════════════════════════════════════
# CONCEPT:
# One head learns ONE type of relationship.
# But language has many types simultaneously:
#   - grammatical (subject → verb)
#   - semantic (pronoun → noun)  
#   - positional (nearby words)
#   - syntactic (modifier → noun)
#
# Solution: run N_HEADS attention heads IN PARALLEL.
# Each head gets its own Q, K, V weight matrices.
# Each head learns different relationships independently.
#
# Outputs: (B, T, HEAD_SIZE) per head
# Concatenated: (B, T, N_HEADS * HEAD_SIZE) = (B, T, N_EMBED)
# Final projection back to N_EMBED.

print("\n" + "=" * 55)
print("PART 2: Multi-Head Attention")
print("=" * 55)

class MultiHeadAttention(nn.Module):
    """
    N_HEADS attention heads running in parallel.
    
    Input:  (B, T, N_EMBED)
    Output: (B, T, N_EMBED)   ← same shape in and out
    """
    
    def __init__(self, n_heads, head_size):
        super().__init__()
        # Create n_heads independent SingleHead modules
        self.heads = nn.ModuleList([
            SingleHead(head_size) for _ in range(n_heads)
        ])
        
        # Final linear projection after concatenation
        # Takes (B, T, N_EMBED) → (B, T, N_EMBED)
        # Mixes information across heads
        self.proj    = nn.Linear(N_EMBED, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT)
    
    def forward(self, x):
        # Run all heads in parallel, each returns (B, T, HEAD_SIZE)
        head_outputs = [head(x) for head in self.heads]
        
        # Concatenate along last dim:
        # 6 × (B, T, 64) → (B, T, 384)
        out = torch.cat(head_outputs, dim=-1)
        
        # Final projection + dropout
        out = self.dropout(self.proj(out))
        return out   # (B, T, N_EMBED)

# ── Test multi-head attention ────────────────────────────────
mha = MultiHeadAttention(N_HEADS, HEAD_SIZE).to(device)
mha_out = mha(emb)

print(f"\nInput:  {emb.shape}     → (B, T, N_EMBED)")
print(f"Output: {mha_out.shape}  → (B, T, N_EMBED)  ← same shape!")
print(f"\n{N_HEADS} heads × {HEAD_SIZE} dims = {N_HEADS * HEAD_SIZE} = N_EMBED ✅")

# ── Count parameters ────────────────────────────────────────
mha_params = sum(p.numel() for p in mha.parameters())
print(f"\nMulti-Head Attention parameters: {mha_params:,}")
print(f"  Per head Q,K,V: 3 × ({N_EMBED}×{HEAD_SIZE}) = {3*N_EMBED*HEAD_SIZE:,}")
print(f"  {N_HEADS} heads: {N_HEADS} × {3*N_EMBED*HEAD_SIZE:,} = {N_HEADS*3*N_EMBED*HEAD_SIZE:,}")
print(f"  Final projection: {N_EMBED}×{N_EMBED} = {N_EMBED*N_EMBED:,}")


# ════════════════════════════════════════════════════════════
# PART 3 — WHY THE SHAPE STAYS (B, T, N_EMBED)
# ════════════════════════════════════════════════════════════
# This is a critical design property.
# Input and output of attention are the same shape.
#
# WHY: Because we stack many attention layers on top of each other.
# The output of layer 1 feeds into layer 2.
# For this to work cleanly, shape must be preserved.
#
# Each layer REFINES the token representations.
# Early layers: low-level patterns (punctuation, syntax)
# Middle layers: semantic relationships
# Deep layers: abstract reasoning, task-specific behavior

print("\n" + "=" * 55)
print("PART 3: Shape Flow Summary")
print("=" * 55)

print(f"""
Token IDs x:          {x.shape}
      ↓ EmbeddingLayer
Embeddings:           {emb.shape}
      ↓ MultiHeadAttention (head 1)
After attention:      {mha_out.shape}  ← same shape
      ↓ MultiHeadAttention (head 2)  [next lesson]
After attention:      (32, 256, 384)  ← still same
      ↓ ... × N_LAYERS
      ↓ Final linear projection
Logits:               (32, 256, 50257)  ← one score per vocab token
      ↓ softmax → sample
Next token prediction ✅
""")


