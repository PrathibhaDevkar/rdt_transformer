

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import urllib.request
import os
import time
from pathlib import Path

# ── DEVICE + HYPERPARAMETERS ────────────────────────────────
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"✅ Device: {device}\n")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

VOCAB_SIZE    = 50_257
BLOCK_SIZE    = 128
BATCH_SIZE    = 16
N_EMBED       = 384
N_HEADS       = 6
HEAD_SIZE     = N_EMBED // N_HEADS
DROPOUT       = 0.2
FF_MULT       = 4

# ── RDT-SPECIFIC HYPERPARAMETERS ────────────────────────────
N_LOOP_STEPS  = 8     # how many times to loop the recurrent block
                       # this is T in h(t+1) = A·h(t) + B·e + Transformer(h(t), e)
N_PRELUDE     = 1     # standard blocks before the loop
N_CODA        = 1     # standard blocks after the loop
SPECTRAL_CAP  = 0.9   # enforce spectral radius of A < this value

LEARNING_RATE = 1e-4  # lower than Phase 1 — more stable for RDT
MAX_ITERS     = 5000
EVAL_INTERVAL = 500
EVAL_ITERS    = 50

print(f"RDT config:")
print(f"  Prelude blocks:   {N_PRELUDE}")
print(f"  Loop steps:       {N_LOOP_STEPS}")
print(f"  Coda blocks:      {N_CODA}")
print(f"  Spectral cap:     {SPECTRAL_CAP}")
print(f"  Effective depth:  {N_PRELUDE + N_LOOP_STEPS + N_CODA} equivalent layers\n")


# ── DATA SETUP ───────────────────────────────────────────────
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

data       = torch.tensor(enc.encode(raw_text), dtype=torch.long)
split      = int(0.9 * len(data))
train_data = data[:split]
val_data   = data[split:]

def get_batch(split_name="train"):
    source = train_data if split_name == "train" else val_data
    ix = torch.randint(len(source) - BLOCK_SIZE, (BATCH_SIZE,))
    x  = torch.stack([source[i   : i+BLOCK_SIZE  ] for i in ix])
    y  = torch.stack([source[i+1 : i+BLOCK_SIZE+1] for i in ix])
    return x.to(device), y.to(device)

print("✅ Data ready\n")


# ── SHARED COMPONENTS FROM PHASE 1 ──────────────────────────
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
    def __init__(self):
        super().__init__()
        self.query   = nn.Linear(N_EMBED, HEAD_SIZE, bias=False)
        self.key     = nn.Linear(N_EMBED, HEAD_SIZE, bias=False)
        self.value   = nn.Linear(N_EMBED, HEAD_SIZE, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)
    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores  = q @ k.transpose(-2,-1) * (HEAD_SIZE ** -0.5)
        scores  = scores.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        return weights @ v

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads   = nn.ModuleList([SingleHead() for _ in range(N_HEADS)])
        self.proj    = nn.Linear(N_EMBED, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBED, FF_MULT * N_EMBED),
            nn.GELU(),
            nn.Linear(FF_MULT * N_EMBED, N_EMBED),
            nn.Dropout(DROPOUT),
        )
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """Standard transformer block from Phase 1 — used for Prelude and Coda."""
    def __init__(self):
        super().__init__()
        self.ln1  = nn.LayerNorm(N_EMBED)
        self.attn = MultiHeadAttention()
        self.ln2  = nn.LayerNorm(N_EMBED)
        self.ffn  = FeedForward()
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ════════════════════════════════════════════════════════════
# PART 1 — THE RECURRENT BLOCK
# ════════════════════════════════════════════════════════════
# CONCEPT:
# This is the core new component. It's a standard transformer
# block PLUS the hidden state injection mechanism.
#
# The update rule at each loop step t:
#   h(t+1) = A·h(t) + B·e + Transformer(h(t), e)
#
# Where:
#   h(t) = hidden state at loop step t  (what we've computed so far)
#   e    = encoded input from Prelude   (re-injected every step)
#   A    = learned (N_EMBED × N_EMBED) matrix — how much h(t) carries forward
#   B    = learned (N_EMBED × N_EMBED) matrix — how much input e carries forward
#
# WHY RE-INJECT e AT EVERY STEP?
# Without re-injection, after many loop steps the hidden state
# drifts away from the original input. The model "forgets"
# what the original question was.
# Re-injecting e at every step acts as an anchor — the model
# always has access to what was originally asked.
#
# ANALOGY:
# Imagine solving a math problem by writing drafts.
# h(t) = your current draft answer
# e    = the original problem statement (you re-read it each draft)
# A·h(t) = how much of your last draft you keep
# B·e    = how much the problem statement directly influences you
# Transformer(h(t), e) = your new reasoning on this draft

print("=" * 55)
print("PART 1: Recurrent Block")
print("=" * 55)

class RecurrentBlock(nn.Module):
    """
    The looped core of the RDT.
    
    Contains one transformer block that gets applied
    N_LOOP_STEPS times with the same weights.
    
    Each loop step refines h using:
        h = A·h + B·e + Transformer(h, e)
    
    Input:
        e: (B, T, N_EMBED) — encoded input from Prelude (fixed)
    
    Output:
        h: (B, T, N_EMBED) — final hidden state after all loops
    """
    
    def __init__(self):
        super().__init__()
        
        # The transformer block — shared across ALL loop steps
        # This is the key difference from Phase 1
        # One set of weights, used N_LOOP_STEPS times
        self.ln1  = nn.LayerNorm(N_EMBED)
        self.attn = MultiHeadAttention()
        self.ln2  = nn.LayerNorm(N_EMBED)
        self.ffn  = FeedForward()
        
        # ── LTI Injection Matrices ──────────────────────────
        # A: how much of previous hidden state carries forward
        # B: how much of original input carries forward
        # Both initialized as near-identity matrices
        # (small perturbation from identity = stable start)
        self.A = nn.Parameter(torch.eye(N_EMBED) + 0.01 * torch.randn(N_EMBED, N_EMBED))
        self.B = nn.Parameter(torch.eye(N_EMBED) + 0.01 * torch.randn(N_EMBED, N_EMBED))
        
        # Layer norm for hidden state — stabilizes h across loops
        self.ln_h = nn.LayerNorm(N_EMBED)
        
        self.dropout = nn.Dropout(DROPOUT)
    
    def _transformer_forward(self, h):
        """Standard transformer block forward — attention + FFN."""
        h = h + self.attn(self.ln1(h))
        h = h + self.ffn(self.ln2(h))
        return h
    
    def enforce_stability(self):
        with torch.no_grad():
            # Move to CPU for SVD — not yet supported on MPS
            A_cpu = self.A.data.cpu()
            S = torch.linalg.svdvals(A_cpu)
            spectral_radius = S.max().item()
            
            if spectral_radius > SPECTRAL_CAP:
                self.A.data *= (SPECTRAL_CAP / spectral_radius)
    
    def forward(self, e):
        # e: (B, T, N_EMBED) — original encoded input, fixed
        B, T, C = e.shape
        
        # Initialize hidden state = encoded input
        # The first "draft" starts from the input itself
        h = e.clone()
        
        # ── Loop N_LOOP_STEPS times ──────────────────────────
        # Same weights used every iteration
        # Each iteration refines h further
        for step in range(N_LOOP_STEPS):
            
            # Transformer refinement on current hidden state
            delta = self._transformer_forward(h)   # (B, T, N_EMBED)
            
            # LTI injection:
            # h(t+1) = A·h(t) + B·e + Transformer(h(t), e)
            #
            # Matrix multiply: (B,T,C) @ (C,C) → (B,T,C)
            # @ with einsum for batched matrix multiply
            Ah = torch.einsum('btc,cd->btd', h, self.A)      # A·h(t)
            Be = torch.einsum('btc,cd->btd', e, self.B)      # B·e
            
            h = Ah + Be + delta                               # update
            h = self.ln_h(h)                                  # normalize
            h = self.dropout(h)
        
        return h   # (B, T, N_EMBED)


# ── Test RecurrentBlock ──────────────────────────────────────
emb_layer = EmbeddingLayer().to(device)
rec_block  = RecurrentBlock().to(device)

x, y  = get_batch()
e     = emb_layer(x)           # (16, 128, 384) — encoded input
h_out = rec_block(e)           # (16, 128, 384) — after 8 loops

print(f"\nInput  shape: {e.shape}")
print(f"Output shape: {h_out.shape}   ← same shape after {N_LOOP_STEPS} loops")

rec_params = sum(p.numel() for p in rec_block.parameters())
print(f"\nRecurrentBlock parameters: {rec_params:,}")
print(f"  Transformer weights: {sum(p.numel() for p in list(rec_block.attn.parameters()) + list(rec_block.ffn.parameters())):,}")
print(f"  A matrix:  {rec_block.A.numel():,}  ({N_EMBED}×{N_EMBED})")
print(f"  B matrix:  {rec_block.B.numel():,}  ({N_EMBED}×{N_EMBED})")

# Check initial spectral radius
S = torch.linalg.svdvals(rec_block.A.data.cpu())
print(f"\nInitial spectral radius of A: {S.max().item():.4f}")
rec_block.enforce_stability()
S_after = torch.linalg.svdvals(rec_block.A.data.cpu())
print(f"After enforce_stability():    {S_after.max().item():.4f}")
print(f"  (capped at {SPECTRAL_CAP})")


# ════════════════════════════════════════════════════════════
# PART 2 — FULL RDT MODEL
# ════════════════════════════════════════════════════════════
# CONCEPT:
# Three-part structure:
#
#   Prelude  → N_PRELUDE standard blocks, run ONCE
#              Processes raw embeddings into a rich
#              starting state for the recurrent loop
#
#   Recurrent Block → looped N_LOOP_STEPS times
#                     Iterative refinement
#
#   Coda     → N_CODA standard blocks, run ONCE
#              Post-processes final hidden state
#              before LM Head prediction
#
# The Prelude and Coda use standard TransformerBlocks
# from Phase 1 — nothing new there.
# The magic is entirely in the RecurrentBlock.

print("\n" + "=" * 55)
print("PART 2: Full RDT Model")
print("=" * 55)

class RDT(nn.Module):
    """
    Recurrent Depth Transformer.
    
    Architecture:
        Embedding → Prelude → RecurrentBlock (×T) → Coda → LM Head
    
    Key difference from NanoGPT:
        The RecurrentBlock uses ONE set of weights
        applied T times instead of T unique blocks.
    """
    
    def __init__(self):
        super().__init__()
        
        # Input embedding (same as Phase 1)
        self.embedding = EmbeddingLayer()
        
        # Prelude — standard blocks, run once
        # Converts raw embeddings into good starting state
        self.prelude = nn.Sequential(
            *[TransformerBlock() for _ in range(N_PRELUDE)]
        )
        
        # Recurrent Block — the unique part
        # ONE block, looped N_LOOP_STEPS times
        self.recurrent = RecurrentBlock()
        
        # Coda — standard blocks, run once
        # Post-processes before prediction
        self.coda = nn.Sequential(
            *[TransformerBlock() for _ in range(N_CODA)]
        )
        
        # Final layer norm
        self.ln_final = nn.LayerNorm(N_EMBED)
        
        # LM Head: 384 → 50,257
        self.lm_head  = nn.Linear(N_EMBED, VOCAB_SIZE, bias=False)
        
        # Weight tying — same as Phase 1
        self.lm_head.weight = self.embedding.token_emb_table.weight
        
        # Weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def enforce_stability(self):
        """Call after each optimizer step to keep A stable."""
        self.recurrent.enforce_stability()
    
    def forward(self, x, targets=None):
        # ── Embedding ────────────────────────────────────────
        e = self.embedding(x)      # (B, T, N_EMBED)
        
        # ── Prelude ──────────────────────────────────────────
        # Standard transformer processing — run once
        e = self.prelude(e)        # (B, T, N_EMBED)
        
        # ── Recurrent Block ──────────────────────────────────
        # The looped core — same weights, N_LOOP_STEPS iterations
        h = self.recurrent(e)      # (B, T, N_EMBED)
        
        # ── Coda ─────────────────────────────────────────────
        # Post-processing — run once
        h = self.coda(h)           # (B, T, N_EMBED)
        
        # ── Final LN + LM Head ───────────────────────────────
        h      = self.ln_final(h)       # (B, T, N_EMBED)
        logits = self.lm_head(h)        # (B, T, VOCAB_SIZE)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.view(B*T, V),
                targets.view(B*T)
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8):
        """Same generation logic as Phase 1."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs  = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


# ── Instantiate and test ─────────────────────────────────────
rdt_model = RDT().to(device)

x, y = get_batch()
logits, loss = rdt_model(x, y)

print(f"\nForward pass:")
print(f"  Input:   {x.shape}")
print(f"  Logits:  {logits.shape}")
print(f"  Loss:    {loss.item():.4f}")

expected = torch.log(torch.tensor(VOCAB_SIZE, dtype=torch.float)).item()
print(f"  Expected random loss: {expected:.4f}")
print(f"  {'✅ Good initialization!' if abs(loss.item() - expected) < 1.5 else '⚠️ Check initialization'}")


# ════════════════════════════════════════════════════════════
# PART 3 — PARAMETER COMPARISON: GPT vs RDT
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("PART 3: Parameter Comparison — GPT vs RDT")
print("=" * 55)

# Count RDT parameters
rdt_total   = sum(p.numel() for p in rdt_model.parameters())
rdt_tied    = rdt_model.embedding.token_emb_table.weight.numel()
rdt_unique  = rdt_total - rdt_tied

print(f"\nRDT breakdown:")
print(f"  Embedding:       {sum(p.numel() for p in rdt_model.embedding.parameters()):>12,}")
print(f"  Prelude ({N_PRELUDE} block): {sum(p.numel() for p in rdt_model.prelude.parameters()):>12,}")
print(f"  RecurrentBlock:  {sum(p.numel() for p in rdt_model.recurrent.parameters()):>12,}  (used {N_LOOP_STEPS}× but counted once)")
print(f"  Coda ({N_CODA} block):    {sum(p.numel() for p in rdt_model.coda.parameters()):>12,}")
print(f"  LM Head:         {'(tied)':>12}")
print(f"  ─────────────────────────────")
print(f"  Total unique:    {rdt_unique:>12,}")

# Compare to equivalent GPT (same effective depth)
# GPT with N_PRELUDE + N_LOOP_STEPS + N_CODA unique blocks
equivalent_gpt_blocks = N_PRELUDE + N_LOOP_STEPS + N_CODA
block_params = sum(p.numel() for p in rdt_model.prelude.parameters()) // N_PRELUDE
gpt_equivalent_params = block_params * equivalent_gpt_blocks

print(f"\nComparison:")
print(f"  Effective depth:          {equivalent_gpt_blocks} layers")
print(f"  GPT with {equivalent_gpt_blocks} unique blocks:  ~{gpt_equivalent_params:>10,} params (estimated)")
print(f"  RDT with {N_LOOP_STEPS} loops:          {rdt_unique:>10,} params (actual)")
print(f"  Parameter savings:         ~{(1 - rdt_unique/gpt_equivalent_params)*100:.0f}% fewer parameters")
print(f"\n💡 Same effective reasoning depth, fewer parameters.")
print(f"   Harder problems → run more loops at inference.")
print(f"   No retraining needed.")


# ════════════════════════════════════════════════════════════
# PART 4 — TRAINING LOOP WITH STABILITY ENFORCEMENT
# ════════════════════════════════════════════════════════════
# KEY DIFFERENCE FROM PHASE 1:
# After every optimizer step, we call enforce_stability()
# to keep spectral radius of A below SPECTRAL_CAP.
# This is the LTI constraint that prevents h from exploding.

print("\n" + "=" * 55)
print("PART 4: Training")
print("=" * 55)

@torch.no_grad()
def estimate_loss():
    out = {}
    rdt_model.eval()
    for split_name in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split_name)
            _, loss = rdt_model(X, Y)
            losses[k] = loss.item()
        out[split_name] = losses.mean().item()
    rdt_model.train()
    return out

optimizer = torch.optim.AdamW(rdt_model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler — reduce LR over time
# This directly addresses the overfitting we saw in Phase 1
# Early steps: higher LR → fast learning
# Later steps: lower LR → gentle refinement, less memorization
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=MAX_ITERS, eta_min=1e-5
)

print(f"\nTraining config:")
print(f"  Steps:           {MAX_ITERS:,}")
print(f"  Learning rate:   {LEARNING_RATE} → 1e-5 (cosine decay)")
print(f"  Loop steps:      {N_LOOP_STEPS}")
print(f"  Stability cap:   {SPECTRAL_CAP}")
print(f"\nStarting training...\n")

start_time    = time.time()
best_val_loss = float('inf')

for step in range(MAX_ITERS):
    
    # Evaluate periodically
    if step % EVAL_INTERVAL == 0 or step == MAX_ITERS - 1:
        losses  = estimate_loss()
        elapsed = time.time() - start_time
        
        # Check spectral radius during training
        A_cpu = rdt_model.recurrent.A.data.cpu()
        S = torch.linalg.svdvals(A_cpu)
        sr = S.max().item()
        
        saved = ""
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save(rdt_model.state_dict(), RESULTS_DIR / "best_rdt_model.pt")
            saved = " ← saved"
        
        print(f"  step {step:4d} | "
              f"train: {losses['train']:.4f} | "
              f"val: {losses['val']:.4f} | "
              f"sr(A): {sr:.3f} | "
              f"t: {elapsed:.0f}s{saved}")
    
    # Forward + backward
    x, y = get_batch("train")
    logits, loss = rdt_model(x, y)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(rdt_model.parameters(), 1.0)
    optimizer.step()
    
    # ── STABILITY ENFORCEMENT ────────────────────────────────
    # This is the key addition vs Phase 1 training loop.
    # After every weight update, re-enforce spectral radius < 0.9
    rdt_model.enforce_stability()
    
    # Step the learning rate scheduler
    scheduler.step()

total_time = time.time() - start_time
print(f"\n✅ Training complete in {total_time/60:.1f} minutes")
print(f"   Best validation loss: {best_val_loss:.4f}")


# ── TEXT GENERATION ──────────────────────────────────────────
print("\n" + "=" * 55)
print("PART 5: Generation — RDT vs GPT comparison")
print("=" * 55)

rdt_model.load_state_dict(torch.load(RESULTS_DIR / "best_rdt_model.pt", map_location=device))
rdt_model.eval()

def generate_text(prompt, max_tokens=200, temperature=0.8):
    tokens = enc.encode(prompt)
    idx    = torch.tensor([tokens], dtype=torch.long, device=device)
    out    = rdt_model.generate(idx, max_new_tokens=max_tokens, temperature=temperature)
    return enc.decode(out[0].tolist())

prompts = ["First Citizen:", "ROMEO:", "To be, or not to be"]
for prompt in prompts:
    print(f"\n{'─'*50}")
    print(f"Prompt: '{prompt}'")
    print(f"{'─'*50}")
    print(generate_text(prompt))


