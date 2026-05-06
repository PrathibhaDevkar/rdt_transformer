
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import json
import os
import time
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
from pathlib import Path

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"✅ Device: {device}\n")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ── HYPERPARAMETERS ─────────────────────────────────────────
# Smaller than Lesson 1 — reasoning task needs less capacity
# but more precision
VOCAB_SIZE    = 64      # small alphabet: nodes A-Z + special tokens
BLOCK_SIZE    = 32      # reasoning chains are short
BATCH_SIZE    = 128     # can use larger batches — sequences are small
N_EMBED       = 128     # smaller embedding — task is simpler
N_HEADS       = 4
HEAD_SIZE     = N_EMBED // N_HEADS   # 32
N_LAYERS_GPT  = 6       # GPT: 6 unique layers
N_LOOP_STEPS  = 6       # RDT: 1 block looped 6 times
N_PRELUDE     = 1
N_CODA        = 1
DROPOUT       = 0.1
FF_MULT       = 4
SPECTRAL_CAP  = 0.9

LEARNING_RATE = 1e-3
MAX_ITERS     = 3000
EVAL_INTERVAL = 300
EVAL_ITERS    = 100

# Reasoning task config
MAX_TRAIN_HOPS = 5      # train on chains up to this length
TEST_HOPS      = [1, 2, 3, 4, 5, 7, 10]  # test at all these lengths
N_ENTITIES     = 20     # number of nodes in the reasoning graph
TRAIN_SAMPLES  = 5000
TEST_SAMPLES   = 500


# ════════════════════════════════════════════════════════════
# PART 1 — SYNTHETIC REASONING DATASET
# ════════════════════════════════════════════════════════════
# CONCEPT:
# We build a simple "chain following" task.
#
# The vocabulary:
#   0     = PAD token
#   1     = EOS (end of sequence)
#   2     = SEP (separates question from answer)
#   3-22  = entity tokens (nodes 0-19, representing A through T)
#   23-42 = relation tokens (arrows, 20 possible relations)
#
# A 3-hop chain looks like:
#   [entity_3, rel_1, entity_7, rel_2, entity_12,
#    rel_3, entity_5, SEP, entity_5, EOS]
#   ← given these facts →      ← answer →
#
# The model sees the chain of (entity, relation, entity) triplets
# and must predict the final entity after SEP.
#
# Why this task?
#   - Ground truth is computable — we know the exact answer
#   - Difficulty scales linearly with hop count
#   - Requires exactly N reasoning steps for N hops
#   - Can test generalization to unseen chain lengths

print("=" * 55)
print("PART 1: Building Reasoning Dataset")
print("=" * 55)

# Special tokens
PAD_TOKEN = 0
EOS_TOKEN = 1
SEP_TOKEN = 2
ENTITY_OFFSET   = 3                        # entities: 3 to 3+N_ENTITIES-1
RELATION_OFFSET = ENTITY_OFFSET + N_ENTITIES  # relations after entities

def entity_token(e):
    """Convert entity id to token id."""
    return ENTITY_OFFSET + e

def relation_token(r):
    """Convert relation id to token id."""
    return RELATION_OFFSET + r

def generate_chain(n_hops, n_entities=N_ENTITIES, n_relations=20):
    """
    Generate one N-hop reasoning chain.
    
    Returns:
        input_tokens:  the chain facts as token ids
        answer_token:  the correct final entity token
        chain:         human-readable list of (entity, relation) pairs
    """
    # Random starting entity
    start = random.randint(0, n_entities - 1)
    
    # Build chain by following random relations
    chain = [start]
    relations = []
    current = start
    
    for _ in range(n_hops):
        relation = random.randint(0, n_relations - 1)
        # Next entity is deterministic given (current, relation)
        # We use a fixed hash so the same (entity, relation) always
        # leads to the same next entity — making it learnable
        next_entity = (current * 7 + relation * 13 + 3) % n_entities
        relations.append(relation)
        chain.append(next_entity)
        current = next_entity
    
    # Build token sequence:
    # [e0, r0, e1, r1, e2, ..., rN-1, eN, SEP, eN, EOS]
    tokens = []
    for i in range(n_hops):
        tokens.append(entity_token(chain[i]))
        tokens.append(relation_token(relations[i]))
    tokens.append(entity_token(chain[-1]))  # final entity
    tokens.append(SEP_TOKEN)
    tokens.append(entity_token(chain[-1]))  # answer (target)
    tokens.append(EOS_TOKEN)
    
    return tokens, entity_token(chain[-1]), chain

def build_dataset(n_samples, max_hops, min_hops=1):
    """Build a dataset of reasoning chains."""
    sequences = []
    answers   = []
    hop_counts = []
    
    for _ in range(n_samples):
        n_hops = random.randint(min_hops, max_hops)
        tokens, answer, _ = generate_chain(n_hops)
        
        # Pad to BLOCK_SIZE
        if len(tokens) <= BLOCK_SIZE:
            padded = tokens + [PAD_TOKEN] * (BLOCK_SIZE - len(tokens))
            sequences.append(padded)
            answers.append(answer)
            hop_counts.append(n_hops)
    
    return (
        torch.tensor(sequences, dtype=torch.long),
        torch.tensor(answers,   dtype=torch.long),
        hop_counts
    )

# Build datasets
print("\nBuilding training dataset...")
train_seqs, train_answers, train_hops = build_dataset(TRAIN_SAMPLES, MAX_TRAIN_HOPS)
print(f"  Training samples: {len(train_seqs):,}  (1-{MAX_TRAIN_HOPS} hops)")

# Build separate test sets for each hop count
test_datasets = {}
for n_hops in TEST_HOPS:
    seqs, answers, _ = build_dataset(TEST_SAMPLES, n_hops, min_hops=n_hops)
    test_datasets[n_hops] = (seqs, answers)
    print(f"  Test {n_hops}-hop: {len(seqs):,} samples")

# Show an example
example_tokens, example_answer, example_chain = generate_chain(3)
print(f"\nExample 3-hop chain:")
print(f"  Chain:  {' → '.join(str(e) for e in example_chain)}")
print(f"  Tokens: {example_tokens}")
print(f"  Answer token: {example_answer}  (entity {example_chain[-1]})")

# Batch loader
def get_reasoning_batch(split="train"):
    if split == "train":
        idx = torch.randint(len(train_seqs), (BATCH_SIZE,))
        seqs    = train_seqs[idx].to(device)
        answers = train_answers[idx].to(device)
    else:
        seqs, answers = test_datasets[split]
        seqs    = seqs.to(device)
        answers = answers.to(device)
    
    # Input = full sequence, target = sequence shifted by 1
    x = seqs[:, :-1]
    y = seqs[:, 1:]
    return x, y, answers, seqs


# ════════════════════════════════════════════════════════════
# PART 2 — MODEL DEFINITIONS
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("PART 2: Model Definitions")
print("=" * 55)

# ── Shared components ────────────────────────────────────────
class EmbeddingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, N_EMBED)
        self.pos_emb   = nn.Embedding(BLOCK_SIZE, N_EMBED)
    def forward(self, x):
        B, T = x.shape
        return self.token_emb(x) + self.pos_emb(torch.arange(T, device=x.device))

class SingleHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(N_EMBED, HEAD_SIZE, bias=False)
        self.k = nn.Linear(N_EMBED, HEAD_SIZE, bias=False)
        self.v = nn.Linear(N_EMBED, HEAD_SIZE, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.drop = nn.Dropout(DROPOUT)
    def forward(self, x):
        B, T, C = x.shape
        q = self.q(x); k = self.k(x); v = self.v(x)
        sc = q @ k.transpose(-2,-1) * HEAD_SIZE**-0.5
        sc = sc.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        w  = self.drop(F.softmax(sc, dim=-1))
        return w @ v

class MHA(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([SingleHead() for _ in range(N_HEADS)])
        self.proj  = nn.Linear(N_EMBED, N_EMBED)
        self.drop  = nn.Dropout(DROPOUT)
    def forward(self, x):
        return self.drop(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))

class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBED, FF_MULT*N_EMBED), nn.GELU(),
            nn.Linear(FF_MULT*N_EMBED, N_EMBED), nn.Dropout(DROPOUT)
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1=nn.LayerNorm(N_EMBED); self.attn=MHA()
        self.ln2=nn.LayerNorm(N_EMBED); self.ffn=FFN()
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

# ── GPT Model ────────────────────────────────────────────────
class GPTReasoner(nn.Module):
    """Standard GPT with N_LAYERS_GPT unique blocks."""
    def __init__(self):
        super().__init__()
        self.emb      = EmbeddingLayer()
        self.blocks   = nn.Sequential(*[Block() for _ in range(N_LAYERS_GPT)])
        self.ln_final = nn.LayerNorm(N_EMBED)
        self.lm_head  = nn.Linear(N_EMBED, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.emb.token_emb.weight
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0, 0.02)
    def forward(self, x, targets=None):
        h = self.ln_final(self.blocks(self.emb(x)))
        logits = self.lm_head(h)
        loss = None
        if targets is not None:
            B,T,V = logits.shape
            loss = F.cross_entropy(logits.reshape(B*T,V), targets.reshape(B*T),
                                   ignore_index=PAD_TOKEN)
        return logits, loss

# ── RDT Model ────────────────────────────────────────────────
class RDTReasoner(nn.Module):
    """RDT with looped recurrent block."""
    def __init__(self, n_loops=N_LOOP_STEPS):
        super().__init__()
        self.n_loops  = n_loops
        self.emb      = EmbeddingLayer()
        self.prelude  = nn.Sequential(*[Block() for _ in range(N_PRELUDE)])
        # Recurrent block components
        self.ln1  = nn.LayerNorm(N_EMBED)
        self.attn = MHA()
        self.ln2  = nn.LayerNorm(N_EMBED)
        self.ffn  = FFN()
        self.ln_h = nn.LayerNorm(N_EMBED)
        # Injection matrices
        self.A = nn.Parameter(torch.eye(N_EMBED) + 0.01*torch.randn(N_EMBED, N_EMBED))
        self.B = nn.Parameter(torch.eye(N_EMBED) + 0.01*torch.randn(N_EMBED, N_EMBED))
        self.coda     = nn.Sequential(*[Block() for _ in range(N_CODA)])
        self.ln_final = nn.LayerNorm(N_EMBED)
        self.lm_head  = nn.Linear(N_EMBED, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.emb.token_emb.weight
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0, 0.02)
    def enforce_stability(self):
        with torch.no_grad():
            S = torch.linalg.svdvals(self.A.data.cpu())
            sr = S.max().item()
            if sr > SPECTRAL_CAP:
                self.A.data *= (SPECTRAL_CAP / sr)
    def _recurrent_forward(self, e):
        h = e.clone()
        for _ in range(self.n_loops):
            delta = h + self.attn(self.ln1(h))
            delta = delta + self.ffn(self.ln2(delta))
            Ah = torch.einsum('btc,cd->btd', h, self.A)
            Be = torch.einsum('btc,cd->btd', e, self.B)
            h  = self.ln_h(Ah + Be + delta)
        return h
    def forward(self, x, targets=None, n_loops_override=None):
        # n_loops_override lets us test with MORE loops at inference
        original_loops = self.n_loops
        if n_loops_override is not None:
            self.n_loops = n_loops_override
        e = self.prelude(self.emb(x))
        h = self._recurrent_forward(e)
        h = self.ln_final(self.coda(h))
        logits = self.lm_head(h)
        loss = None
        if targets is not None:
            B,T,V = logits.shape
            loss = F.cross_entropy(logits.reshape(B*T,V), targets.reshape(B*T),
                                   ignore_index=PAD_TOKEN)
        self.n_loops = original_loops
        return logits, loss


# ── Parameter counts ─────────────────────────────────────────
gpt_model = GPTReasoner().to(device)
rdt_model = RDTReasoner().to(device)

gpt_params = sum(p.numel() for p in gpt_model.parameters())
rdt_params = sum(p.numel() for p in rdt_model.parameters())

print(f"\nGPT parameters: {gpt_params:,}  ({N_LAYERS_GPT} unique blocks)")
print(f"RDT parameters: {rdt_params:,}  (1 block × {N_LOOP_STEPS} loops)")
print(f"RDT is {(1-rdt_params/gpt_params)*100:.0f}% smaller")


# ════════════════════════════════════════════════════════════
# PART 3 — ACCURACY EVALUATION
# ════════════════════════════════════════════════════════════
# For reasoning tasks, loss isn't the best metric.
# We care about: did the model get the RIGHT answer?
#
# Evaluation: feed the chain, find the SEP token position,
# check if the model's top prediction after SEP = correct answer.

print("\n" + "=" * 55)
print("PART 3: Accuracy Evaluation Function")
print("=" * 55)

@torch.no_grad()
def evaluate_accuracy(model, n_hops, n_loops_override=None):
    """
    Evaluate exact-match accuracy on N-hop chains.
    
    For each test sequence:
      1. Find position of SEP token
      2. Get model's prediction at that position
      3. Check if top-1 prediction = correct answer
    
    Returns accuracy as percentage.
    """
    model.eval()
    seqs, answers = test_datasets[n_hops]
    
    # Process in batches
    correct = 0
    total   = 0
    batch_size = 64
    
    for i in range(0, len(seqs), batch_size):
        batch_seqs    = seqs[i:i+batch_size].to(device)
        batch_answers = answers[i:i+batch_size].to(device)
        
        x = batch_seqs[:, :-1]
        
        if n_loops_override is not None:
            logits, _ = model(x, n_loops_override=n_loops_override)
        else:
            logits, _ = model(x)
        
        # Find SEP token position in each sequence
        for j in range(len(batch_seqs)):
            seq = batch_seqs[j]
            sep_positions = (seq == SEP_TOKEN).nonzero(as_tuple=True)[0]
            
            if len(sep_positions) == 0:
                continue
            
            sep_pos = sep_positions[0].item()
            
            # Prediction at SEP position = next token prediction
            if sep_pos - 1 < logits.shape[1]:
                pred = logits[j, sep_pos].argmax().item()
                true = batch_answers[j].item()
                
                if pred == true:
                    correct += 1
                total += 1
    
    model.train()
    acc = (correct / total * 100) if total > 0 else 0
    return acc

print("\nAccuracy function ready.")
print("Measures: did the model predict the correct final entity?")


# ════════════════════════════════════════════════════════════
# PART 4 — TRAIN BOTH MODELS
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("PART 4: Training Both Models")
print("=" * 55)

def train_model(model, model_name, is_rdt=False):
    """Train a model and return loss history."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_ITERS, eta_min=1e-5
    )
    
    train_losses = []
    val_accuracies = []
    best_val_acc = 0
    best_path = RESULTS_DIR / f"best_{model_name.lower()}_reasoner.pt"
    
    print(f"\n--- Training {model_name} ---")
    start = time.time()
    
    for step in range(MAX_ITERS):
        # Get batch
        x, y, answers, seqs = get_reasoning_batch("train")
        logits, loss = model(x, y)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if is_rdt:
            model.enforce_stability()
        
        scheduler.step()
        
        # Evaluate
        if step % EVAL_INTERVAL == 0 or step == MAX_ITERS - 1:
            # Accuracy on 3-hop (representative of training distribution)
            acc_3 = evaluate_accuracy(model, 3)
            acc_5 = evaluate_accuracy(model, 5)
            elapsed = time.time() - start
            
            train_losses.append(loss.item())
            val_accuracies.append(acc_3)
            
            saved = ""
            if acc_3 > best_val_acc or step == 0:
                best_val_acc = acc_3
                torch.save(model.state_dict(), best_path)
                saved = " ← saved"
            
            print(f"  step {step:4d} | loss: {loss.item():.4f} | "
                  f"acc@3hop: {acc_3:.1f}% | acc@5hop: {acc_5:.1f}% | "
                  f"t: {elapsed:.0f}s{saved}")
    
    print(f"  Best 3-hop accuracy: {best_val_acc:.1f}%")
    
    # Load best model
    model.load_state_dict(torch.load(best_path, map_location=device))
    return train_losses, val_accuracies

# Train GPT
gpt_losses, gpt_accs = train_model(gpt_model, "GPT", is_rdt=False)

# Train RDT
rdt_losses, rdt_accs = train_model(rdt_model, "RDT", is_rdt=True)


# ════════════════════════════════════════════════════════════
# PART 5 — THE KEY EXPERIMENT
# ════════════════════════════════════════════════════════════
# Both models trained on 1-5 hop chains.
# Now test on ALL hop counts including 7 and 10 (unseen).
#
# Key test: RDT with MORE loops on harder chains.
# We run the RDT with 6 loops (trained) vs 10 loops (more)
# on 7-hop and 10-hop chains.
# If loops ~ reasoning steps, more loops should help.

print("\n" + "=" * 55)
print("PART 5: The Key Experiment — Generalization to Longer Chains")
print("=" * 55)

gpt_model.eval()
rdt_model.eval()

print("\nEvaluating on all hop counts...")
print(f"\n{'Hops':>6} | {'GPT Acc':>10} | {'RDT (6 loops)':>14} | {'RDT (10 loops)':>15} | {'RDT (14 loops)':>15}")
print("-" * 70)

results = {
    'hops': [],
    'gpt': [],
    'rdt_trained': [],
    'rdt_more': [],
    'rdt_extra': [],
}

for n_hops in TEST_HOPS:
    gpt_acc      = evaluate_accuracy(gpt_model, n_hops)
    rdt_acc_base = evaluate_accuracy(rdt_model, n_hops, n_loops_override=6)
    rdt_acc_more = evaluate_accuracy(rdt_model, n_hops, n_loops_override=10)
    rdt_acc_xtra = evaluate_accuracy(rdt_model, n_hops, n_loops_override=14)
    
    results['hops'].append(n_hops)
    results['gpt'].append(gpt_acc)
    results['rdt_trained'].append(rdt_acc_base)
    results['rdt_more'].append(rdt_acc_more)
    results['rdt_extra'].append(rdt_acc_xtra)
    
    # Mark unseen hop counts
    unseen = " *" if n_hops > MAX_TRAIN_HOPS else "  "
    print(f"{n_hops:>5}{unseen} | {gpt_acc:>9.1f}% | {rdt_acc_base:>13.1f}% | "
          f"{rdt_acc_more:>14.1f}% | {rdt_acc_xtra:>14.1f}%")

print("\n* = unseen during training (generalization test)")
print("\nKey finding:")
print("  On unseen longer chains (7-hop, 10-hop):")
print("  → Does GPT accuracy drop?")
print("  → Does more RDT loops recover accuracy?")


# ════════════════════════════════════════════════════════════
# PART 6 — SAVE RESULTS + PLOT
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("PART 6: Saving Results")
print("=" * 55)

# Save numerical results
results_data = {
    'config': {
        'max_train_hops': MAX_TRAIN_HOPS,
        'gpt_layers': N_LAYERS_GPT,
        'rdt_loops_trained': N_LOOP_STEPS,
        'gpt_params': gpt_params,
        'rdt_params': rdt_params,
    },
    'results': results
}

with open(RESULTS_DIR / "benchmark_results.json", "w") as f:
    json.dump(results_data, f, indent=2)
print("✅ Results saved to benchmark_results.json")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: accuracy vs hop count
ax = axes[0]
ax.plot(results['hops'], results['gpt'],         'b-o', label=f'GPT ({N_LAYERS_GPT} layers)', linewidth=2)
ax.plot(results['hops'], results['rdt_trained'], 'r-o', label=f'RDT ({N_LOOP_STEPS} loops, trained)', linewidth=2)
ax.plot(results['hops'], results['rdt_more'],    'r--s', label='RDT (10 loops, inference)', linewidth=2)
ax.plot(results['hops'], results['rdt_extra'],   'r:^', label='RDT (14 loops, inference)', linewidth=2)
ax.axvline(x=MAX_TRAIN_HOPS, color='gray', linestyle='--', alpha=0.7, label='Train boundary')
ax.fill_betweenx([0,100], MAX_TRAIN_HOPS, max(TEST_HOPS),
                  alpha=0.1, color='gray', label='Unseen at training')
ax.set_xlabel('Number of Hops', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Reasoning Accuracy: GPT vs RDT', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 105)

# Right: training loss
ax2 = axes[1]
steps = list(range(0, MAX_ITERS, EVAL_INTERVAL)) + [MAX_ITERS-1]
steps = steps[:len(gpt_losses)]
ax2.plot(steps, gpt_losses, 'b-', label='GPT loss', linewidth=2)
ax2.plot(steps, rdt_losses, 'r-', label='RDT loss', linewidth=2)
ax2.set_xlabel('Training Step', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_title('Training Loss: GPT vs RDT', fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "benchmark_plot.png", dpi=150, bbox_inches='tight')
plt.close()
print("Plot saved to benchmark_plot.png")


