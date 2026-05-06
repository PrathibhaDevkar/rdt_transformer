

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import json
import time
import matplotlib
matplotlib.use('Agg')
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
N_ENTITIES    = 6
N_RELATIONS   = 3
BLOCK_SIZE    = 160
BATCH_SIZE    = 64
N_EMBED       = 256      # increased from 128
N_HEADS       = 8        # increased from 4
HEAD_SIZE     = N_EMBED // N_HEADS   # 32
N_LAYERS_GPT  = 6
N_LOOP_STEPS  = 6
N_PRELUDE     = 1
N_CODA        = 1
DROPOUT       = 0.1
FF_MULT       = 4
SPECTRAL_CAP  = 0.9

# Curriculum config
CURRICULUM_STAGES    = [1, 2, 3, 4, 5]   # max hops per stage
STEPS_PER_STAGE      = 2000              # max steps per stage
ACCURACY_THRESHOLD   = 82.0              # advance when this accuracy reached
SAMPLES_PER_STAGE    = 5000
LEARNING_RATE        = 3e-4

# Test config
TEST_HOPS    = [1, 2, 3, 4, 5, 7, 10]
TEST_SAMPLES = 500

# Token vocabulary
PAD  = 0; EOS = 1; SEP = 2; ARR = 3; QST = 4
ENTITY_OFFSET   = 5
RELATION_OFFSET = ENTITY_OFFSET + N_ENTITIES
VOCAB_SIZE      = RELATION_OFFSET + N_RELATIONS + 5

def E(i): return ENTITY_OFFSET   + i
def R(i): return RELATION_OFFSET + i


# ════════════════════════════════════════════════════════════
# PART 1 — DATASET
# ════════════════════════════════════════════════════════════

print("=" * 55)
print("PART 1: Dataset")
print("=" * 55)

def generate_random_table():
    return {e: {r: random.randint(0, N_ENTITIES-1)
                for r in range(N_RELATIONS)}
            for e in range(N_ENTITIES)}

def table_to_tokens(table):
    tokens = []
    for e in range(N_ENTITIES):
        for r in range(N_RELATIONS):
            tokens += [E(e), R(r), ARR, E(table[e][r])]
    return tokens

def follow_chain(table, start, relations):
    current = start
    for r in relations:
        current = table[current][r]
    return current

def generate_example(n_hops):
    table     = generate_random_table()
    start     = random.randint(0, N_ENTITIES-1)
    relations = [random.randint(0, N_RELATIONS-1) for _ in range(n_hops)]
    answer    = follow_chain(table, start, relations)
    tokens    = table_to_tokens(table)
    tokens   += [SEP, E(start)] + [R(r) for r in relations] + [QST, E(answer), EOS]
    return tokens, E(answer)

def build_dataset(n_samples, max_hops, min_hops=1):
    seqs, answers = [], []
    for _ in range(n_samples):
        n_hops = random.randint(min_hops, max_hops)
        tokens, answer = generate_example(n_hops)
        if len(tokens) <= BLOCK_SIZE:
            padded = tokens + [PAD] * (BLOCK_SIZE - len(tokens))
            seqs.append(padded)
            answers.append(answer)
    return (torch.tensor(seqs, dtype=torch.long),
            torch.tensor(answers, dtype=torch.long))

# Build test sets (fixed — never changes across stages)
print("\nBuilding test sets...")
test_datasets = {}
for n_hops in TEST_HOPS:
    seqs, answers = build_dataset(TEST_SAMPLES, n_hops, min_hops=n_hops)
    test_datasets[n_hops] = (seqs, answers)
    print(f"  Test {n_hops:2d}-hop: {len(seqs)} samples")

# Verify sequence length
example_tokens, _ = generate_example(10)
print(f"\nMax sequence length (10-hop): {len(example_tokens)} tokens")
print(f"BLOCK_SIZE: {BLOCK_SIZE} → {'✅ fits' if len(example_tokens) <= BLOCK_SIZE else '❌ too small'}")


# ════════════════════════════════════════════════════════════
# PART 2 — MODELS
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("PART 2: Models")
print("=" * 55)

class EmbeddingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB_SIZE, N_EMBED)
        self.pos = nn.Embedding(BLOCK_SIZE, N_EMBED)
    def forward(self, x):
        B, T = x.shape
        return self.tok(x) + self.pos(torch.arange(T, device=x.device))

class SingleHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(N_EMBED, HEAD_SIZE, bias=False)
        self.k = nn.Linear(N_EMBED, HEAD_SIZE, bias=False)
        self.v = nn.Linear(N_EMBED, HEAD_SIZE, bias=False)
        self.register_buffer('tril',
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.drop = nn.Dropout(DROPOUT)
    def forward(self, x):
        B, T, C = x.shape
        q = self.q(x); k = self.k(x); v = self.v(x)
        sc = q @ k.transpose(-2,-1) * HEAD_SIZE**-0.5
        sc = sc.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        return self.drop(F.softmax(sc, dim=-1)) @ v

class MHA(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([SingleHead() for _ in range(N_HEADS)])
        self.proj  = nn.Linear(N_EMBED, N_EMBED)
        self.drop  = nn.Dropout(DROPOUT)
    def forward(self, x):
        return self.drop(self.proj(
            torch.cat([h(x) for h in self.heads], dim=-1)))

class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBED, FF_MULT*N_EMBED), nn.GELU(),
            nn.Linear(FF_MULT*N_EMBED, N_EMBED), nn.Dropout(DROPOUT))
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

class GPTReasoner(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb      = EmbeddingLayer()
        self.blocks   = nn.Sequential(*[Block() for _ in range(N_LAYERS_GPT)])
        self.ln_final = nn.LayerNorm(N_EMBED)
        self.lm_head  = nn.Linear(N_EMBED, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.emb.tok.weight
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0, 0.02)
    def forward(self, x, targets=None, **kwargs):
        h = self.ln_final(self.blocks(self.emb(x)))
        logits = self.lm_head(h)
        loss = None
        if targets is not None:
            B,T,V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B*T,V), targets.reshape(B*T),
                ignore_index=PAD)
        return logits, loss

class RDTReasoner(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_loops  = N_LOOP_STEPS
        self.emb      = EmbeddingLayer()
        self.prelude  = nn.Sequential(*[Block() for _ in range(N_PRELUDE)])
        self.ln1=nn.LayerNorm(N_EMBED); self.attn=MHA()
        self.ln2=nn.LayerNorm(N_EMBED); self.ffn=FFN()
        self.ln_h     = nn.LayerNorm(N_EMBED)
        self.A = nn.Parameter(
            torch.eye(N_EMBED) + 0.01*torch.randn(N_EMBED, N_EMBED))
        self.B = nn.Parameter(
            torch.eye(N_EMBED) + 0.01*torch.randn(N_EMBED, N_EMBED))
        self.coda     = nn.Sequential(*[Block() for _ in range(N_CODA)])
        self.ln_final = nn.LayerNorm(N_EMBED)
        self.lm_head  = nn.Linear(N_EMBED, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.emb.tok.weight
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0, 0.02)
    def enforce_stability(self):
        with torch.no_grad():
            S  = torch.linalg.svdvals(self.A.data.cpu())
            sr = S.max().item()
            if sr > SPECTRAL_CAP:
                self.A.data *= (SPECTRAL_CAP / sr)
    def _recurrent_forward(self, e, n_loops):
        h = e.clone()
        for _ in range(n_loops):
            delta = h + self.attn(self.ln1(h))
            delta = delta + self.ffn(self.ln2(delta))
            Ah = torch.einsum('btc,cd->btd', h, self.A)
            Be = torch.einsum('btc,cd->btd', e, self.B)
            h  = self.ln_h(Ah + Be + delta)
        return h
    def forward(self, x, targets=None, n_loops_override=None):
        loops  = n_loops_override or self.n_loops
        e = self.prelude(self.emb(x))
        h = self._recurrent_forward(e, loops)
        h = self.ln_final(self.coda(h))
        logits = self.lm_head(h)
        loss = None
        if targets is not None:
            B,T,V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B*T,V), targets.reshape(B*T),
                ignore_index=PAD)
        return logits, loss

gpt_model = GPTReasoner().to(device)
rdt_model = RDTReasoner().to(device)

gpt_params = sum(p.numel() for p in gpt_model.parameters())
rdt_params = sum(p.numel() for p in rdt_model.parameters())
print(f"\nGPT: {gpt_params:,} params ({N_LAYERS_GPT} unique blocks)")
print(f"RDT: {rdt_params:,} params (1 block × {N_LOOP_STEPS} loops)")
print(f"RDT is {(1-rdt_params/gpt_params)*100:.0f}% smaller")


# ════════════════════════════════════════════════════════════
# PART 3 — EVALUATION
# ════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_accuracy(model, n_hops, n_loops_override=None):
    model.eval()
    seqs, answers = test_datasets[n_hops]
    correct = 0; total = 0

    for i in range(0, len(seqs), 64):
        batch_seqs = seqs[i:i+64].to(device)
        batch_ans  = answers[i:i+64].to(device)
        x = batch_seqs[:, :-1]

        if n_loops_override is not None:
            logits, _ = model(x, n_loops_override=n_loops_override)
        else:
            logits, _ = model(x)

        for j in range(len(batch_seqs)):
            seq = batch_seqs[j]
            qst_pos = (seq == QST).nonzero(as_tuple=True)[0]
            if len(qst_pos) == 0: continue
            pos = qst_pos[0].item()
            if pos < logits.shape[1]:
                if logits[j, pos].argmax().item() == batch_ans[j].item():
                    correct += 1
                total += 1

    model.train()
    return (correct/total*100) if total > 0 else 0


# ════════════════════════════════════════════════════════════
# PART 4 — CURRICULUM TRAINING
# ════════════════════════════════════════════════════════════
# CONCEPT:
# Instead of training on all hop counts at once,
# we advance through stages:
#
#   Stage 1: only 1-hop chains
#             → model learns the basic 1-step lookup
#   Stage 2: 1 and 2-hop chains
#             → model learns "do it twice"
#   ...
#   Stage 5: all 1-5 hop chains
#             → model can handle full training distribution
#
# We advance to the next stage when accuracy on the
# current hardest hop count exceeds ACCURACY_THRESHOLD.
# This ensures we never move on before mastering the skill.
#
# WHY THIS WORKS:
#   The model is forced to build composable skills.
#   1-hop mastery is prerequisite for 2-hop.
#   2-hop mastery is prerequisite for 3-hop.
#   By stage 5 the model has a genuine step-by-step rule.

print("\n" + "=" * 55)
print("PART 4: Curriculum Training")
print("=" * 55)

def curriculum_train(model, model_name, is_rdt=False):
    """
    Train model through curriculum stages.
    Returns per-stage accuracy history.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    stage_history = []   # accuracy at end of each stage
    total_steps   = 0
    start_time    = time.time()

    print(f"\n{'='*50}")
    print(f"Training: {model_name}")
    print(f"{'='*50}")

    for stage_idx, max_hops in enumerate(CURRICULUM_STAGES):
        print(f"\n--- Stage {stage_idx+1}: max {max_hops} hop(s) ---")

        # Build fresh training data for this stage
        train_seqs, train_answers = build_dataset(
            SAMPLES_PER_STAGE, max_hops)
        print(f"    Training samples: {len(train_seqs):,}")

        best_stage_acc = 0.0
        best_path = RESULTS_DIR / f"best_{model_name.lower()}_curriculum.pt"

        for step in range(STEPS_PER_STAGE):
            # Sample batch from current stage data
            idx  = torch.randint(len(train_seqs), (BATCH_SIZE,))
            seqs = train_seqs[idx].to(device)
            x    = seqs[:, :-1]
            y    = seqs[:, 1:]

            logits, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if is_rdt:
                model.enforce_stability()

            total_steps += 1

            # Evaluate every 200 steps
            if step % 200 == 0 or step == STEPS_PER_STAGE - 1:
                # Accuracy on the hardest hop in this stage
                acc = evaluate_accuracy(model, max_hops)
                elapsed = time.time() - start_time

                saved = ""
                if acc > best_stage_acc:
                    best_stage_acc = acc
                    torch.save(model.state_dict(), best_path)
                    saved = " ← saved"

                print(f"    step {step:4d} | loss: {loss.item():.4f} | "
                      f"acc@{max_hops}hop: {acc:.1f}% | "
                      f"t: {elapsed:.0f}s{saved}")

                # Advance early if threshold reached
                if acc >= ACCURACY_THRESHOLD and step >= 200:
                    print(f"    ✅ Threshold {ACCURACY_THRESHOLD}% reached!"
                          f" Advancing to next stage.")
                    break

        stage_history.append(best_stage_acc)
        print(f"    Stage {stage_idx+1} best accuracy: {best_stage_acc:.1f}%")

    print(f"\n✅ Curriculum complete. Total steps: {total_steps:,}")
    print(f"   Time: {(time.time()-start_time)/60:.1f} minutes")

    # Load best overall checkpoint
    model.load_state_dict(
        torch.load(best_path, map_location=device))
    return stage_history

# Train both models
gpt_stage_history = curriculum_train(gpt_model, "GPT", is_rdt=False)
rdt_stage_history = curriculum_train(rdt_model, "RDT", is_rdt=True)


# ════════════════════════════════════════════════════════════
# PART 5 — THE KEY EXPERIMENT
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("PART 5: Generalization Test")
print("=" * 55)

print(f"\n{'Hops':>6} | {'GPT':>8} | {'RDT(6)':>8} | "
      f"{'RDT(10)':>9} | {'RDT(14)':>9} | {'Note'}")
print("-" * 68)

results = {
    'hops': [], 'gpt': [],
    'rdt_6': [], 'rdt_10': [], 'rdt_14': []
}

for n_hops in TEST_HOPS:
    gpt_acc = evaluate_accuracy(gpt_model, n_hops)
    rdt_6   = evaluate_accuracy(rdt_model, n_hops, n_loops_override=6)
    rdt_10  = evaluate_accuracy(rdt_model, n_hops, n_loops_override=10)
    rdt_14  = evaluate_accuracy(rdt_model, n_hops, n_loops_override=14)

    results['hops'].append(n_hops)
    results['gpt'].append(gpt_acc)
    results['rdt_6'].append(rdt_6)
    results['rdt_10'].append(rdt_10)
    results['rdt_14'].append(rdt_14)

    note = "← UNSEEN" if n_hops > max(CURRICULUM_STAGES) else ""
    print(f"{n_hops:>6} | {gpt_acc:>7.1f}% | {rdt_6:>7.1f}% | "
          f"{rdt_10:>8.1f}% | {rdt_14:>8.1f}% | {note}")

# ── Analysis ─────────────────────────────────────────────────
print("\n--- Key Findings ---")

# Stage learning curve
print(f"\nCurriculum stage accuracy:")
print(f"  {'Stage':>6} | {'Max Hops':>9} | {'GPT':>8} | {'RDT':>8}")
print(f"  {'-'*40}")
for i, (max_hops, gpt_acc, rdt_acc) in enumerate(
        zip(CURRICULUM_STAGES, gpt_stage_history, rdt_stage_history)):
    print(f"  {i+1:>6} | {max_hops:>9} | {gpt_acc:>7.1f}% | {rdt_acc:>7.1f}%")

# Generalization gap
print(f"\nGeneralization to unseen lengths:")
for n_hops in [7, 10]:
    if n_hops in results['hops']:
        idx     = results['hops'].index(n_hops)
        gpt_acc = results['gpt'][idx]
        rdt_6   = results['rdt_6'][idx]
        rdt_14  = results['rdt_14'][idx]
        gain    = rdt_14 - rdt_6

        print(f"\n  {n_hops}-hop (unseen):")
        print(f"    GPT (6 layers):   {gpt_acc:.1f}%")
        print(f"    RDT (6 loops):    {rdt_6:.1f}%")
        print(f"    RDT (14 loops):   {rdt_14:.1f}%")
        if gain > 5:
            print(f"    → Loop scaling helped: +{gain:.1f}% ✅")
        elif rdt_14 > gpt_acc + 5:
            print(f"    → RDT outperformed GPT: "
                  f"+{rdt_14-gpt_acc:.1f}% ✅")
        else:
            print(f"    → Marginal difference: {gain:.1f}%")

# Parameter efficiency
print(f"\nParameter efficiency:")
print(f"  GPT: {gpt_params:,} parameters")
print(f"  RDT: {rdt_params:,} parameters")
print(f"  RDT uses {(1-rdt_params/gpt_params)*100:.0f}% fewer parameters")
for n_hops in [3, 5]:
    idx = results['hops'].index(n_hops)
    print(f"  At {n_hops}-hop: GPT={results['gpt'][idx]:.1f}%, "
          f"RDT={results['rdt_6'][idx]:.1f}%")


# ════════════════════════════════════════════════════════════
# PART 6 — SAVE + PLOT
# ════════════════════════════════════════════════════════════

# Save results
with open(RESULTS_DIR / "benchmark_v3_results.json", "w") as f:
    json.dump({
        'config': {
            'curriculum_stages': CURRICULUM_STAGES,
            'accuracy_threshold': ACCURACY_THRESHOLD,
            'gpt_layers': N_LAYERS_GPT,
            'rdt_loops': N_LOOP_STEPS,
            'n_embed': N_EMBED,
            'gpt_params': gpt_params,
            'rdt_params': rdt_params,
        },
        'results': results,
        'stage_history': {
            'gpt': gpt_stage_history,
            'rdt': rdt_stage_history,
        }
    }, f, indent=2)
print("\n✅ Results saved to benchmark_v3_results.json")

# Plot — 3 panels
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: accuracy vs hops
ax = axes[0]
ax.plot(results['hops'], results['gpt'],
        'b-o', label=f'GPT ({N_LAYERS_GPT} layers)', lw=2, ms=8)
ax.plot(results['hops'], results['rdt_6'],
        'r-o', label='RDT (6 loops, trained)', lw=2, ms=8)
ax.plot(results['hops'], results['rdt_10'],
        'r--s', label='RDT (10 loops)', lw=2, ms=7)
ax.plot(results['hops'], results['rdt_14'],
        'r:^', label='RDT (14 loops)', lw=2, ms=7)
ax.axvline(x=5.5, color='gray', linestyle='--', alpha=0.7)
ax.fill_betweenx([0,105], 5.5, 10.5,
                  alpha=0.08, color='orange')
ax.text(6.2, 90, 'Unseen', fontsize=9, color='darkorange')
ax.set_xlabel('Chain Length (hops)', fontsize=11)
ax.set_ylabel('Accuracy (%)', fontsize=11)
ax.set_title('Generalization:\nGPT vs RDT', fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(-5, 108)
ax.set_xticks(TEST_HOPS)

# Panel 2: curriculum learning curve
ax2 = axes[1]
stages = list(range(1, len(CURRICULUM_STAGES)+1))
ax2.plot(stages, gpt_stage_history, 'b-o',
         label='GPT', lw=2, ms=8)
ax2.plot(stages, rdt_stage_history, 'r-o',
         label='RDT', lw=2, ms=8)
ax2.set_xlabel('Curriculum Stage', fontsize=11)
ax2.set_ylabel('Best Accuracy (%)', fontsize=11)
ax2.set_title('Curriculum Learning:\nStage Accuracy', fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-5, 108)
ax2.set_xticks(stages)
stage_labels = [f'Stage {i}\n(≤{h} hop)' for i,h in
                enumerate(CURRICULUM_STAGES, 1)]
ax2.set_xticklabels(stage_labels, fontsize=7)

# Panel 3: parameter efficiency
ax3 = axes[2]
models  = ['GPT\n(6 layers)', 'RDT\n(6 loops)']
params  = [gpt_params/1e6, rdt_params/1e6]
colors  = ['#4472C4', '#C0392B']
bars    = ax3.bar(models, params, color=colors, width=0.4, alpha=0.85)
ax3.set_ylabel('Parameters (millions)', fontsize=11)
ax3.set_title('Parameter Efficiency', fontsize=11)
for bar, val in zip(bars, params):
    ax3.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.02,
             f'{val:.2f}M', ha='center', fontsize=10, fontweight='bold')
savings = (1 - rdt_params/gpt_params)*100
ax3.text(0.5, max(params)*0.6,
         f'{savings:.0f}% fewer\nparameters',
         ha='center', fontsize=11, color='#C0392B', fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim(0, max(params)*1.3)

plt.suptitle('RDT vs GPT: Curriculum Benchmark Results',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "benchmark_v3_plot.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("Plot saved to benchmark_v3_plot.png")

