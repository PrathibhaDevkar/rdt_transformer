

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import random
import os
import time
import tiktoken
from pathlib import Path

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"  Device: {device}\n")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ── HYPERPARAMETERS ─────────────────────────────────────────
VOCAB_SIZE    = 50_257   # GPT-2 BPE tokenizer
BLOCK_SIZE    = 128
BATCH_SIZE    = 8
N_EMBED       = 384
N_HEADS       = 6
HEAD_SIZE     = N_EMBED // N_HEADS
N_LAYERS      = 4
N_LOOP_STEPS  = 6
N_PRELUDE     = 1
N_CODA        = 1
DROPOUT       = 0.1
FF_MULT       = 4
SPECTRAL_CAP  = 0.9

# Fine-tuning config
FINETUNE_LR     = 1e-4    # lower than pretraining — gentle updates
FINETUNE_ITERS  = 3000
EVAL_INTERVAL   = 300
TRAIN_SAMPLES   = 3000
TEST_SAMPLES    = 300

enc = tiktoken.get_encoding("gpt2")


# ════════════════════════════════════════════════════════════
# PART 1 — TOOL DEFINITIONS
# ════════════════════════════════════════════════════════════
# CONCEPT:
# Tools are just Python functions.
# The agent calls them by name with arguments.
# Results come back as strings.
#
# Three tools:
#   calculator  → safe math evaluation
#   search      → fact lookup from knowledge base
#   transform   → text operations

print("=" * 55)
print("PART 1: Tool Definitions")
print("=" * 55)

# ── Knowledge base for search tool ──────────────────────────
KNOWLEDGE_BASE = {
    "capital of france":        "Paris",
    "capital of japan":         "Tokyo",
    "capital of germany":       "Berlin",
    "capital of australia":     "Canberra",
    "capital of brazil":        "Brasília",
    "capital of india":         "New Delhi",
    "capital of canada":        "Ottawa",
    "capital of argentina":     "Buenos Aires",
    "largest planet":           "Jupiter",
    "closest planet to sun":    "Mercury",
    "speed of light":           "299,792,458 meters per second",
    "boiling point of water":   "100 degrees Celsius",
    "freezing point of water":  "0 degrees Celsius",
    "number of continents":     "7",
    "number of oceans":         "5",
    "inventor of telephone":    "Alexander Graham Bell",
    "author of hamlet":         "William Shakespeare",
    "year of moon landing":     "1969",
    "tallest mountain":         "Mount Everest",
    "longest river":            "Nile River",
}

def tool_calculator(expression: str) -> str:
    """Safely evaluate a math expression."""
    try:
        # Only allow safe math operations
        allowed = set('0123456789+-*/()., ')
        if not all(c in allowed for c in expression):
            return "Error: invalid characters in expression"
        result = eval(expression, {"__builtins__": {}}, {})
        return str(round(result, 6))
    except Exception as e:
        return f"Error: {str(e)}"

def tool_search(query: str) -> str:
    """Look up a fact from the knowledge base."""
    key = query.lower().strip()
    if key in KNOWLEDGE_BASE:
        return KNOWLEDGE_BASE[key]
    # Fuzzy match — find closest key
    for kb_key, value in KNOWLEDGE_BASE.items():
        if key in kb_key or kb_key in key:
            return value
    return f"Not found: '{query}'"

def tool_transform(op: str, text: str) -> str:
    """Apply a text transformation."""
    op = op.lower().strip()
    if op == "upper":
        return text.upper()
    elif op == "lower":
        return text.lower()
    elif op == "reverse":
        return text[::-1]
    elif op == "count_words":
        return str(len(text.split()))
    elif op == "count_chars":
        return str(len(text))
    elif op == "title":
        return text.title()
    else:
        return f"Unknown operation: {op}"

# Tool registry — maps name to function
TOOLS = {
    "calculator": tool_calculator,
    "search":     tool_search,
    "transform":  tool_transform,
}

def execute_tool(tool_call: dict) -> str:
    """Execute a tool call and return the result."""
    tool_name = tool_call.get("tool")
    if tool_name not in TOOLS:
        return f"Unknown tool: {tool_name}"
    if tool_name == "calculator":
        return tool_calculator(tool_call.get("input", ""))
    elif tool_name == "search":
        return tool_search(tool_call.get("input", ""))
    elif tool_name == "transform":
        return tool_transform(
            tool_call.get("op", ""),
            tool_call.get("input", "")
        )
    return "Error: could not execute tool"

# Test tools
print("\nTesting tools:")
print(f"  calculator('15 * 4 + 7')     → {tool_calculator('15 * 4 + 7')}")
print(f"  calculator('(100 - 32) / 1.8') → {tool_calculator('(100 - 32) / 1.8')}")
print(f"  search('capital of japan')   → {tool_search('capital of japan')}")
print(f"  search('tallest mountain')   → {tool_search('tallest mountain')}")
print(f"  transform('upper', 'hello')  → {tool_transform('upper', 'hello')}")
print(f"  transform('reverse', 'abcd') → {tool_transform('reverse', 'abcd')}")
print(f"  transform('count_words', 'hello world foo') → {tool_transform('count_words', 'hello world foo')}")


# ════════════════════════════════════════════════════════════
# PART 2 — TOOL CALLING DATASET
# ════════════════════════════════════════════════════════════
# CONCEPT:
# Each training example is a (prompt, completion) pair.
#
# Format:
#   <task>What is 25 multiplied by 4?</task>
#   <tool>{"tool": "calculator", "input": "25 * 4"}</tool>
#   <result>100</result>
#   <answer>25 multiplied by 4 is 100.</answer>
#
# We train the model to predict everything after <task>...
# The model learns: given a task, produce the right tool call.
#
# We use XML-like tags so the agent loop can easily parse
# what the model is outputting at each step.

print("\n" + "=" * 55)
print("PART 2: Tool Calling Dataset")
print("=" * 55)

def make_calculator_example():
    """Generate a math task requiring calculator."""
    templates = [
        (lambda a,b: f"What is {a} plus {b}?",
         lambda a,b: f"{a} + {b}",
         lambda a,b: f"{a} plus {b} equals {tool_calculator(f'{a} + {b}')}.",),

        (lambda a,b: f"Calculate {a} times {b}.",
         lambda a,b: f"{a} * {b}",
         lambda a,b: f"{a} times {b} is {tool_calculator(f'{a} * {b}')}.",),

        (lambda a,b: f"What is {a} minus {b}?",
         lambda a,b: f"{a} - {b}",
         lambda a,b: f"{a} minus {b} equals {tool_calculator(f'{a} - {b}')}.",),

        (lambda a,b: f"Divide {a} by {b}.",
         lambda a,b: f"{a} / {b}",
         lambda a,b: f"{a} divided by {b} is {tool_calculator(f'{a} / {b}')}.",),

        (lambda a,b: f"What is {a} squared plus {b}?",
         lambda a,b: f"{a}**2 + {b}",
         lambda a,b: f"{a} squared plus {b} equals {tool_calculator(f'{a}**2 + {b}')}.",),
    ]
    a = random.randint(1, 50)
    b = random.randint(1, 50)
    if b == 0: b = 1
    task_fn, expr_fn, ans_fn = random.choice(templates)
    task   = task_fn(a, b)
    expr   = expr_fn(a, b)
    result = tool_calculator(expr)
    answer = ans_fn(a, b)
    tool_json = json.dumps({"tool": "calculator", "input": expr})
    return task, tool_json, result, answer

def make_search_example():
    """Generate a lookup task requiring search."""
    key = random.choice(list(KNOWLEDGE_BASE.keys()))
    value = KNOWLEDGE_BASE[key]
    templates = [
        f"What is the {key}?",
        f"Tell me the {key}.",
        f"I need to know the {key}.",
        f"Look up: {key}.",
        f"Find the {key} for me.",
    ]
    task      = random.choice(templates)
    tool_json = json.dumps({"tool": "search", "input": key})
    result    = value
    answer    = f"The {key} is {value}."
    return task, tool_json, result, answer

def make_transform_example():
    """Generate a text transformation task."""
    words = ["hello world", "python programming", "machine learning",
             "recurrent transformer", "deep neural network",
             "artificial intelligence", "gradient descent"]
    text = random.choice(words)
    ops = {
        "upper":       (f"Convert '{text}' to uppercase.",
                        f"'{text}' in uppercase is '{tool_transform('upper', text)}'."),
        "lower":       (f"Convert '{text.upper()}' to lowercase.",
                        f"'{text.upper()}' in lowercase is '{tool_transform('lower', text.upper())}'."),
        "reverse":     (f"Reverse the string '{text}'.",
                        f"'{text}' reversed is '{tool_transform('reverse', text)}'."),
        "count_words": (f"How many words are in '{text}'?",
                        f"'{text}' has {tool_transform('count_words', text)} words."),
        "title":       (f"Convert '{text}' to title case.",
                        f"'{text}' in title case is '{tool_transform('title', text)}'."),
    }
    op = random.choice(list(ops.keys()))
    task, answer = ops[op]
    if op == "lower":
        input_text = text.upper()
    else:
        input_text = text
    tool_json = json.dumps({"tool": "transform", "op": op, "input": input_text})
    result    = tool_transform(op, input_text)
    return task, tool_json, result, answer

def make_example():
    """Generate one training example randomly."""
    generators = [make_calculator_example,
                  make_search_example,
                  make_transform_example]
    return random.choice(generators)()

def format_example(task, tool_json, result, answer):
    """Format as training text."""
    return (f"<task>{task}</task>"
            f"<tool>{tool_json}</tool>"
            f"<result>{result}</result>"
            f"<answer>{answer}</answer>")

def build_finetune_dataset(n_samples):
    """Build tokenized dataset for fine-tuning."""
    sequences = []
    for _ in range(n_samples):
        task, tool_json, result, answer = make_example()
        text   = format_example(task, tool_json, result, answer)
        tokens = enc.encode(text)
        if len(tokens) < BLOCK_SIZE:
            padded = tokens + [0] * (BLOCK_SIZE - len(tokens))
            sequences.append(padded[:BLOCK_SIZE])
    return torch.tensor(sequences, dtype=torch.long)

# Build datasets
print("\nBuilding fine-tuning dataset...")
train_data = build_finetune_dataset(TRAIN_SAMPLES)
test_data  = build_finetune_dataset(TEST_SAMPLES)
print(f"  Train: {len(train_data):,} examples")
print(f"  Test:  {len(test_data):,}  examples")

# Show examples
print("\nSample training examples:")
for i in range(3):
    task, tool_json, result, answer = make_example()
    print(f"\n  Example {i+1}:")
    print(f"    Task:   {task}")
    print(f"    Tool:   {tool_json}")
    print(f"    Result: {result}")
    print(f"    Answer: {answer}")

def get_finetune_batch(split="train"):
    source = train_data if split == "train" else test_data
    idx = torch.randint(len(source), (BATCH_SIZE,))
    seqs = source[idx].to(device)
    x = seqs[:, :-1]
    y = seqs[:, 1:]
    return x, y


# ════════════════════════════════════════════════════════════
# PART 3 — RDT MODEL (from Phase 1 + 2)
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("PART 3: RDT Model")
print("=" * 55)

class EmbeddingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, N_EMBED)
        self.pos_emb   = nn.Embedding(BLOCK_SIZE, N_EMBED)
    def forward(self, x):
        B, T = x.shape
        return (self.token_emb(x) +
                self.pos_emb(torch.arange(T, device=x.device)))

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

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([SingleHead() for _ in range(N_HEADS)])
        self.proj  = nn.Linear(N_EMBED, N_EMBED)
        self.drop  = nn.Dropout(DROPOUT)
    def forward(self, x):
        return self.drop(self.proj(
            torch.cat([h(x) for h in self.heads], dim=-1)))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBED, FF_MULT*N_EMBED), nn.GELU(),
            nn.Linear(FF_MULT*N_EMBED, N_EMBED), nn.Dropout(DROPOUT))
    def forward(self, x): return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1=nn.LayerNorm(N_EMBED); self.attn=MultiHeadAttention()
        self.ln2=nn.LayerNorm(N_EMBED); self.ffn=FeedForward()
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class RDT(nn.Module):
    """
    Full RDT model — same architecture as Phase 2.
    Fine-tuned here for tool calling instead of Shakespeare.
    """
    def __init__(self):
        super().__init__()
        self.embedding = EmbeddingLayer()
        self.prelude   = nn.Sequential(
            *[TransformerBlock() for _ in range(N_PRELUDE)])
        # Recurrent block components
        self.ln1  = nn.LayerNorm(N_EMBED)
        self.attn = MultiHeadAttention()
        self.ln2  = nn.LayerNorm(N_EMBED)
        self.ffn  = FeedForward()
        self.ln_h = nn.LayerNorm(N_EMBED)
        self.A = nn.Parameter(
            torch.eye(N_EMBED) + 0.01*torch.randn(N_EMBED, N_EMBED))
        self.B = nn.Parameter(
            torch.eye(N_EMBED) + 0.01*torch.randn(N_EMBED, N_EMBED))
        self.coda     = nn.Sequential(
            *[TransformerBlock() for _ in range(N_CODA)])
        self.ln_final = nn.LayerNorm(N_EMBED)
        self.lm_head  = nn.Linear(N_EMBED, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.embedding.token_emb.weight
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

    def forward(self, x, targets=None, n_loops=None):
        loops = n_loops or N_LOOP_STEPS
        e = self.prelude(self.embedding(x))
        h = self._recurrent_forward(e, loops)
        h = self.ln_final(self.coda(h))
        logits = self.lm_head(h)
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B*T, V),
                targets.reshape(B*T),
                ignore_index=0)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100,
                 temperature=0.7, stop_token=None):
        """Generate tokens until stop_token or max_new_tokens."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs  = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
            # Stop if we generated the stop token
            if stop_token and next_token.item() == stop_token:
                break
        return idx

# Instantiate
model = RDT().to(device)
params = sum(p.numel() for p in model.parameters())
print(f"\nRDT model: {params:,} parameters")
print(f"Architecture: Embed → Prelude → Loop×{N_LOOP_STEPS} → Coda → LM Head")

# Test forward pass
x, y = get_finetune_batch()
logits, loss = model(x, y)
print(f"Forward pass: input {x.shape} → logits {logits.shape}")
print(f"Initial loss: {loss.item():.4f}")


# ════════════════════════════════════════════════════════════
# PART 4 — FINE-TUNING
# ════════════════════════════════════════════════════════════
# CONCEPT:
# Fine-tuning = continue training a model on new data
# with a lower learning rate.
#
# We're starting from a randomly initialized RDT here.
# In a full pipeline you'd load the Phase 1 Shakespeare
# weights first, then fine-tune on tool calling.
# Starting from scratch still works for demonstrating
# the tool-calling behavior — it just needs more steps.
#
# Lower LR than pretraining because:
#   - We don't want to destroy general knowledge
#   - Tool calling requires precise output format
#   - Gentle updates → more stable JSON generation

print("\n" + "=" * 55)
print("PART 4: Fine-Tuning for Tool Calling")
print("=" * 55)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=FINETUNE_LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=FINETUNE_ITERS, eta_min=1e-5)

best_loss = float('inf')
start = time.time()

print(f"\nFine-tuning config:")
print(f"  Steps:         {FINETUNE_ITERS:,}")
print(f"  Learning rate: {FINETUNE_LR} → 1e-5 (cosine)")
print(f"  Batch size:    {BATCH_SIZE}")
print(f"\nTraining...\n")

for step in range(FINETUNE_ITERS):
    x, y = get_finetune_batch("train")
    logits, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    model.enforce_stability()
    scheduler.step()

    if step % EVAL_INTERVAL == 0 or step == FINETUNE_ITERS - 1:
        # Val loss
        model.eval()
        with torch.no_grad():
            xv, yv = get_finetune_batch("test")
            _, val_loss = model(xv, yv)
        model.train()

        elapsed = time.time() - start
        saved = ""
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            torch.save(model.state_dict(), RESULTS_DIR / "best_rdt_agent.pt")
            saved = " ← saved"

        print(f"  step {step:4d} | "
              f"train: {loss.item():.4f} | "
              f"val: {val_loss.item():.4f} | "
              f"t: {elapsed:.0f}s{saved}")

print(f"\n  Fine-tuning complete")
print(f"   Best val loss: {best_loss:.4f}")


# ════════════════════════════════════════════════════════════
# PART 5 — TEST TOOL CALL GENERATION
# ════════════════════════════════════════════════════════════
# Now let's see if the model generates valid tool calls.
# We give it a task prefix and let it complete the rest.

print("\n" + "=" * 55)
print("PART 5: Testing Tool Call Generation")
print("=" * 55)

model.load_state_dict(
    torch.load(RESULTS_DIR / "best_rdt_agent.pt", map_location=device))
model.eval()

def generate_tool_call(task: str, max_tokens: int = 80) -> str:
    """
    Given a task, generate the model's tool call.
    Returns the raw generated text.
    """
    prompt = f"<task>{task}</task><tool>"
    tokens = enc.encode(prompt)
    idx    = torch.tensor([tokens], dtype=torch.long, device=device)

    # Generate until we see </tool> closing tag
    output_ids = model.generate(idx, max_new_tokens=max_tokens,
                                 temperature=0.3)
    generated  = enc.decode(output_ids[0].tolist())

    # Extract just the tool call part
    if "<tool>" in generated and "</tool>" in generated:
        start = generated.find("<tool>") + len("<tool>")
        end   = generated.find("</tool>")
        return generated[start:end].strip()
    elif "<tool>" in generated:
        start = generated.find("<tool>") + len("<tool>")
        return generated[start:start+100].strip()
    return generated

def parse_tool_call(raw: str) -> dict:
    """Parse raw model output into tool call dict."""
    try:
        # Find JSON in the output
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
    except:
        pass
    return {}

# Test on sample tasks
test_tasks = [
    "What is 42 multiplied by 7?",
    "What is the capital of France?",
    "Convert 'machine learning' to uppercase.",
    "Calculate 100 divided by 4.",
    "What is the tallest mountain?",
    "Reverse the string 'hello'.",
    "How many words are in 'the quick brown fox'?",
]

print(f"\nTesting {len(test_tasks)} tasks:\n")
correct = 0
for task in test_tasks:
    raw       = generate_tool_call(task)
    tool_call = parse_tool_call(raw)

    print(f"  Task:      {task}")
    print(f"  Generated: {raw[:80]}")

    if tool_call and "tool" in tool_call:
        result = execute_tool(tool_call)
        print(f"  Parsed:    {tool_call}")
        print(f"  Executed:  {result}")
        correct += 1
        status = " "
    else:
        print(f"  Parsed:      invalid JSON")
        status = " "

    print(f"  Status:    {status}\n")

print(f"Valid tool calls: {correct}/{len(test_tasks)}")
print(f"JSON parse rate:  {correct/len(test_tasks)*100:.0f}%")


