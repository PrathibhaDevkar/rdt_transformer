
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

VOCAB_SIZE   = 50_257
BLOCK_SIZE   = 128      # reduced from 256 -> saves 4× attention memory
BATCH_SIZE   = 16       # reduced from 32  -> halves batch memory
N_EMBED      = 384
N_HEADS      = 6
HEAD_SIZE    = N_EMBED // N_HEADS
N_LAYERS     = 4
DROPOUT      = 0.1
FF_MULT      = 4
LEARNING_RATE = 3e-4
MAX_ITERS     = 5000
EVAL_INTERVAL = 500
EVAL_ITERS    = 50

# ── DATA SETUP ───────────────────────────────────────────────
DATA_PATH = DATA_DIR / "shakespeare.txt"
def download_dataset():
    DATA_DIR.mkdir(exist_ok=True)
    if not os.path.exists(DATA_PATH):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print("⬇️  Downloading dataset...")
        urllib.request.urlretrieve(url, DATA_PATH)

download_dataset()
enc = tiktoken.get_encoding("gpt2")
with open(DATA_PATH) as f:
    raw_text = f.read()

data       = torch.tensor(enc.encode(raw_text), dtype=torch.long)
split      = int(0.9 * len(data))
train_data = data[:split]
val_data   = data[split:]

def get_batch(split_name):
    source = train_data if split_name == "train" else val_data
    ix = torch.randint(len(source) - BLOCK_SIZE, (BATCH_SIZE,))
    x  = torch.stack([source[i   : i+BLOCK_SIZE  ] for i in ix])
    y  = torch.stack([source[i+1 : i+BLOCK_SIZE+1] for i in ix])
    return x.to(device), y.to(device)

print("✅ Data ready\n")


# ── ALL COMPONENTS FROM PREVIOUS LESSONS ────────────────────
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
        scores  = q @ k.transpose(-2,-1) * (HEAD_SIZE ** -0.5)
        scores  = scores.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
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

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, FF_MULT * n_embed),
            nn.GELU(),
            nn.Linear(FF_MULT * n_embed, n_embed),
            nn.Dropout(DROPOUT),
        )
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1  = nn.LayerNorm(N_EMBED)
        self.attn = MultiHeadAttention()
        self.ln2  = nn.LayerNorm(N_EMBED)
        self.ffn  = FeedForward(N_EMBED)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ════════════════════════════════════════════════════════════
# PART 1 — LANGUAGE MODEL HEAD
# ════════════════════════════════════════════════════════════
# CONCEPT:
# The backbone outputs (B, T, 384) — rich token representations.
# But we need to predict the next token from 50,257 options.
#
# The LM Head is one linear layer: 384 → 50,257
# It produces "logits" — one raw score per vocabulary token.
# Higher score = model thinks that token is more likely next.
#
# These are NOT probabilities yet — just raw scores.
# We use softmax to convert to probabilities at generation time.
# During training we use cross-entropy loss directly on logits.
#
# WEIGHT TYING:
# We reuse the token embedding weights for the LM head.
# The embedding maps token_id → vector.
# The LM head maps vector → token_id scores.
# They're inverses of each other — sharing weights works well
# and saves ~19M parameters. This is standard practice.

print("=" * 55)
print("PART 1: Complete GPT Model")
print("=" * 55)

class NanoGPT(nn.Module):
    """
    Complete language model: Embedding + Transformer + LM Head.
    
    Forward pass:
      Input:  (B, T) integer token IDs
      Output: (B, T, VOCAB_SIZE) logits  [training mode]
              + scalar loss if targets provided
    
    Generation:
      Autoregressively samples next tokens.
    """
    
    def __init__(self):
        super().__init__()
        self.embedding = EmbeddingLayer()
        self.blocks    = nn.Sequential(*[TransformerBlock() for _ in range(N_LAYERS)])
        self.ln_final  = nn.LayerNorm(N_EMBED)
        
        # LM Head: 384 → 50,257
        # Maps final token representations to vocab scores
        self.lm_head   = nn.Linear(N_EMBED, VOCAB_SIZE, bias=False)
        
        # Weight tying: share embedding and lm_head weights
        # token_emb_table: (50257, 384)
        # lm_head:         (50257, 384) — same matrix, reused
        self.lm_head.weight = self.embedding.token_emb_table.weight
        
        # Initialize weights properly
        # Small initial weights = stable training start
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize linear and embedding weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, targets=None):
        # x: (B, T) token IDs
        # targets: (B, T) next token IDs — only provided during training
        
        # Pass through backbone
        h = self.embedding(x)      # (B, T, N_EMBED)
        h = self.blocks(h)         # (B, T, N_EMBED)
        h = self.ln_final(h)       # (B, T, N_EMBED)
        
        # Project to vocabulary
        logits = self.lm_head(h)   # (B, T, VOCAB_SIZE)
        
        # If targets provided, compute loss
        loss = None
        if targets is not None:
            # Cross-entropy expects (N, C) and (N,)
            # We reshape: (B*T, VOCAB_SIZE) and (B*T,)
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.view(B*T, V),
                targets.view(B*T)
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8):
        """
        Generate text autoregressively.
        
        idx: (1, T) starting token IDs (the "prompt")
        max_new_tokens: how many tokens to generate
        temperature: controls randomness
            < 1.0 = more focused/deterministic
            > 1.0 = more random/creative
        """
        for _ in range(max_new_tokens):
            # Crop context to BLOCK_SIZE if too long
            idx_cond = idx[:, -BLOCK_SIZE:]
            
            # Forward pass — no targets needed
            logits, _ = self(idx_cond)
            
            # Take logits at the LAST position (next token prediction)
            logits = logits[:, -1, :]   # (1, VOCAB_SIZE)
            
            # Apply temperature — scale before softmax
            # Low temp → sharper distribution → more predictable
            # High temp → flatter distribution → more creative
            logits = logits / temperature
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)   # (1, VOCAB_SIZE)
            
            # Sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)   # (1, 1)
            
            # Append to sequence and continue
            idx = torch.cat([idx, idx_next], dim=1)   # (1, T+1)
        
        return idx

# Instantiate model
model = NanoGPT().to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
# Subtract tied weights (counted twice)
tied_params = model.embedding.token_emb_table.weight.numel()
unique_params = total_params - tied_params

print(f"\nModel architecture:")
print(f"  Embedding:    {sum(p.numel() for p in model.embedding.parameters()):>12,}")
print(f"  {N_LAYERS} Blocks:      {sum(p.numel() for p in model.blocks.parameters()):>12,}")
print(f"  LM Head:      {sum(p.numel() for p in model.lm_head.parameters()):>12,}  (tied - not extra)")
print(f"  Total unique: {unique_params:>12,} parameters")

# Test forward pass
x, y = get_batch("train")
logits, loss = model(x, y)
print(f"\nForward pass test:")
print(f"  Input:   {x.shape}")
print(f"  Logits:  {logits.shape}   → (B, T, VOCAB_SIZE)")
print(f"  Loss:    {loss.item():.4f}")

# What should initial loss be?
# With random weights, the model has no preference.
# It should assign equal probability to all 50,257 tokens.
# Expected loss = -log(1/50257) = log(50257) ≈ 10.82
expected_loss = torch.log(torch.tensor(VOCAB_SIZE, dtype=torch.float)).item()
print(f"\n  Expected random loss: {expected_loss:.4f}")
print(f"  Actual initial loss:  {loss.item():.4f}")
print(f"  {'✅ Close to random — good initialization!' if abs(loss.item() - expected_loss) < 1.5 else '⚠️ Unusual initial loss'}")


# ════════════════════════════════════════════════════════════
# PART 2 — CROSS-ENTROPY LOSS EXPLAINED
# ════════════════════════════════════════════════════════════
# CONCEPT:
# Cross-entropy loss measures how wrong the model's predictions are.
#
# For each token position, the model outputs 50,257 logit scores.
# After softmax, these become probabilities.
# The loss = -log(probability assigned to the CORRECT next token)
#
# Example:
#   Correct next token: "cat" (id=4839)
#   Model assigns probability 0.001 to "cat"
#   Loss = -log(0.001) = 6.9  ← very wrong
#
#   After training:
#   Model assigns probability 0.6 to "cat"
#   Loss = -log(0.6) = 0.51   ← much better
#
# The lower the loss, the better the model is at predicting
# the next token. We want to minimize this number.
#
# Why -log? Because log(p) is negative for p < 1.
# Negating makes it positive. log is also nice mathematically —
# it turns products into sums and has great gradient properties.

print("\n" + "=" * 55)
print("PART 2: Cross-Entropy Loss — What it Means")
print("=" * 55)

print(f"\nLoss interpretation guide:")
losses = [10.82, 7.0, 5.0, 3.5, 2.5, 2.0]
meanings = [
    "Random guessing — untrained model",
    "Learned basic token frequencies",
    "Learning word patterns",
    "Decent language model",
    "Good — coherent sentences forming",
    "Strong — Shakespeare-like output",
]
for l, m in zip(losses, meanings):
    bar = "█" * int((11 - l) * 3)
    print(f"  {l:.2f}  {bar:<20} {m}")


# ════════════════════════════════════════════════════════════
# PART 3 — TRAINING LOOP
# ════════════════════════════════════════════════════════════
# CONCEPT: The training loop is the same 4 steps, repeated:
#
#   1. get_batch()      → sample random data
#   2. model(x, y)      → forward pass, compute loss
#   3. loss.backward()  → backprop: compute gradients
#   4. optimizer.step() → update weights using gradients
#
# OPTIMIZER: AdamW
# SGD (Stochastic Gradient Descent) is the simplest optimizer.
# AdamW improves on it by:
#   - Maintaining a running average of gradients (momentum)
#   - Adapting learning rate per parameter (adaptive)
#   - Weight decay for regularization
# It's the standard optimizer for transformer training.
#
# GRADIENT CLIPPING:
# Sometimes gradients spike to very large values, causing
# the optimizer to take a huge step and destabilize training.
# Clipping caps gradient norm at 1.0 — a safety measure.

print("\n" + "=" * 55)
print("PART 3: Training Loop")
print("=" * 55)

# Loss estimation function
# We average over EVAL_ITERS batches for a stable estimate
@torch.no_grad()
def estimate_loss():
    """Compute average loss over multiple batches for stability."""
    out = {}
    model.eval()   # disable dropout during evaluation
    for split_name in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split_name)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split_name] = losses.mean().item()
    model.train()  # re-enable dropout
    return out

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print(f"\nTraining config:")
print(f"  Steps:         {MAX_ITERS:,}")
print(f"  Batch size:    {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Eval every:    {EVAL_INTERVAL} steps")
print(f"\nStarting training... (est. 10-15 min on MPS)\n")

# ── Training loop ────────────────────────────────────────────
start_time = time.time()
best_val_loss = float('inf')

for step in range(MAX_ITERS):
    
    # ── Evaluate periodically ────────────────────────────────
    if step % EVAL_INTERVAL == 0 or step == MAX_ITERS - 1:
        losses   = estimate_loss()
        elapsed  = time.time() - start_time
        train_l  = losses['train']
        val_l    = losses['val']
        
        # Track best validation loss
        if val_l < best_val_loss:
            best_val_loss = val_l
            torch.save(model.state_dict(), RESULTS_DIR / "best_model.pt")
            saved = " ← saved"
        else:
            saved = ""
        
        print(f"  step {step:4d}/{MAX_ITERS} | "
              f"train loss: {train_l:.4f} | "
              f"val loss: {val_l:.4f} | "
              f"time: {elapsed:.0f}s{saved}")
    
    # ── Forward pass ─────────────────────────────────────────
    x, y = get_batch("train")
    logits, loss = model(x, y)
    
    # ── Backward pass ────────────────────────────────────────
    optimizer.zero_grad(set_to_none=True)   # clear previous gradients
    loss.backward()                          # compute new gradients
    
    # Gradient clipping — prevents training instability
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # ── Update weights ───────────────────────────────────────
    optimizer.step()

total_time = time.time() - start_time
print(f"\n✅ Training complete in {total_time/60:.1f} minutes")
print(f"   Best validation loss: {best_val_loss:.4f}")


# ════════════════════════════════════════════════════════════
# PART 4 — GENERATE TEXT
# ════════════════════════════════════════════════════════════
# Now the exciting part — sample from the trained model.
# We give it a short prompt and let it continue.

print("\n" + "=" * 55)
print("PART 4: Text Generation")
print("=" * 55)

# Load best saved model
model.load_state_dict(torch.load(RESULTS_DIR / "best_model.pt", map_location=device))
model.eval()

def generate_text(prompt, max_tokens=300, temperature=0.8):
    """Generate text from a prompt string."""
    # Encode prompt to token IDs
    prompt_tokens = enc.encode(prompt)
    idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    
    # Generate
    output_ids = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature)
    
    # Decode back to text
    return enc.decode(output_ids[0].tolist())

prompts = [
    "First Citizen:",
    "ROMEO:",
    "To be, or not to be",
]

for prompt in prompts:
    print(f"\n{'─'*50}")
    print(f"Prompt: '{prompt}'")
    print(f"{'─'*50}")
    generated = generate_text(prompt, max_tokens=200)
    print(generated)

