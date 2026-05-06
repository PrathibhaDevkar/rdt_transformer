

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import random
import time
import tiktoken
from pathlib import Path

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"✅ Device: {device}\n")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ── HYPERPARAMETERS ─────────────────────────────────────────
VOCAB_SIZE     = 50_257
BLOCK_SIZE     = 192     # larger — CoT adds tokens
BATCH_SIZE     = 8
N_EMBED        = 384
N_HEADS        = 6
HEAD_SIZE      = N_EMBED // N_HEADS
N_LOOP_STEPS   = 6
N_PRELUDE      = 1
N_CODA         = 1
DROPOUT        = 0.1
FF_MULT        = 4
SPECTRAL_CAP   = 0.9

FINETUNE_LR    = 1e-4
FINETUNE_ITERS = 6000
EVAL_INTERVAL  = 500
TRAIN_SAMPLES  = 10000
TEST_SAMPLES   = 500

enc = tiktoken.get_encoding("gpt2")


# ════════════════════════════════════════════════════════════
# PART 1 — TOOLS
# ════════════════════════════════════════════════════════════

print("=" * 55)
print("PART 1: Tools")
print("=" * 55)

KNOWLEDGE_BASE = {
    "capital of france":        "Paris",
    "capital of japan":         "Tokyo",
    "capital of germany":       "Berlin",
    "capital of australia":     "Canberra",
    "capital of brazil":        "Brasilia",
    "capital of india":         "New Delhi",
    "capital of canada":        "Ottawa",
    "capital of argentina":     "Buenos Aires",
    "largest planet":           "Jupiter",
    "closest planet to sun":    "Mercury",
    "boiling point of water":   "100",
    "freezing point of water":  "0",
    "number of continents":     "7",
    "number of oceans":         "5",
    "inventor of telephone":    "Alexander Graham Bell",
    "author of hamlet":         "William Shakespeare",
    "year of moon landing":     "1969",
    "tallest mountain":         "Mount Everest",
    "longest river":            "Nile River",
    "speed of light":           "299792458",
}

def tool_calculator(expression: str) -> str:
    try:
        allowed = set('0123456789+-*/()., ')
        if not all(c in allowed for c in expression):
            return "Error: invalid expression"
        result = eval(expression, {"__builtins__": {}}, {})
        return str(round(float(result), 4))
    except Exception as e:
        return f"Error: {e}"

def tool_search(query: str) -> str:
    key = query.lower().strip()
    if key in KNOWLEDGE_BASE:
        return KNOWLEDGE_BASE[key]
    for kb_key, value in KNOWLEDGE_BASE.items():
        if key in kb_key or kb_key in key:
            return value
    return f"Not found: '{query}'"

def tool_transform(op: str, text: str) -> str:
    ops = {
        "upper":       text.upper(),
        "lower":       text.lower(),
        "reverse":     text[::-1],
        "count_words": str(len(text.split())),
        "count_chars": str(len(text)),
        "title":       text.title(),
    }
    return ops.get(op.lower().strip(), f"Unknown op: {op}")

def execute_tool(tool_call: dict) -> str:
    name = tool_call.get("tool", "")
    if name == "calculator":
        return tool_calculator(tool_call.get("input", ""))
    elif name == "search":
        return tool_search(tool_call.get("input", ""))
    elif name == "transform":
        return tool_transform(
            tool_call.get("op", ""),
            tool_call.get("input", ""))
    return f"Unknown tool: {name}"

print("Tools ready.\n")


# ════════════════════════════════════════════════════════════
# PART 2 — CHAIN-OF-THOUGHT DATASET
# ════════════════════════════════════════════════════════════
# THE KEY CHANGE:
# Every example now has a <think> step that explicitly:
#   1. Identifies the values to extract from the task
#   2. Names the tool and operation
#   3. States the exact argument to use
#
# The model trains to produce this thinking step FIRST
# before generating the tool call. This forces it to
# read and copy values from the task rather than
# hallucinating plausible-looking arguments.
#
# Format:
#   <task>{task}</task>
#   <think>{reasoning}</think>
#   <tool>{json}</tool>
#   <r>{result}</r>
#   <answer>{answer}</answer>

print("=" * 55)
print("PART 2: Chain-of-Thought Dataset")
print("=" * 55)

def make_calculator_cot():
    """Calculator example with explicit CoT reasoning."""
    a = random.randint(2, 999)
    b = random.randint(2, 99)

    ops = [
        # (task, think, expression, answer)
        (
            f"What is {a} plus {b}?",
            f"The task asks to add {a} and {b}. "
            f"I need the calculator tool with input \"{a} + {b}\".",
            f"{a} + {b}",
            f"{a} plus {b} equals {tool_calculator(f'{a} + {b}')}."
        ),
        (
            f"Calculate {a} times {b}.",
            f"The task asks to multiply {a} by {b}. "
            f"I need the calculator tool with input \"{a} * {b}\".",
            f"{a} * {b}",
            f"{a} times {b} is {tool_calculator(f'{a} * {b}')}."
        ),
        (
            f"What is {a} minus {b}?",
            f"The task asks to subtract {b} from {a}. "
            f"I need the calculator tool with input \"{a} - {b}\".",
            f"{a} - {b}",
            f"{a} minus {b} equals {tool_calculator(f'{a} - {b}')}."
        ),
        (
            f"Divide {a} by {b}.",
            f"The task asks to divide {a} by {b}. "
            f"I need the calculator tool with input \"{a} / {b}\".",
            f"{a} / {b}",
            f"{a} divided by {b} is {tool_calculator(f'{a} / {b}')}."
        ),
        (
            f"Find the product of {a} and {b}.",
            f"The task asks for the product of {a} and {b}. "
            f"I need the calculator tool with input \"{a} * {b}\".",
            f"{a} * {b}",
            f"The product of {a} and {b} is {tool_calculator(f'{a} * {b}')}."
        ),
        (
            f"Compute {a} divided by {b}.",
            f"The task asks to divide {a} by {b}. "
            f"I need the calculator tool with input \"{a} / {b}\".",
            f"{a} / {b}",
            f"{a} divided by {b} equals {tool_calculator(f'{a} / {b}')}."
        ),
        (
            f"What is the sum of {a} and {b}?",
            f"The task asks for the sum of {a} and {b}. "
            f"I need the calculator tool with input \"{a} + {b}\".",
            f"{a} + {b}",
            f"The sum of {a} and {b} is {tool_calculator(f'{a} + {b}')}."
        ),
        (
            f"Subtract {b} from {a}.",
            f"The task asks to subtract {b} from {a}. "
            f"I need the calculator tool with input \"{a} - {b}\".",
            f"{a} - {b}",
            f"{a} minus {b} is {tool_calculator(f'{a} - {b}')}."
        ),
    ]

    task, think, expr, answer = random.choice(ops)
    result    = tool_calculator(expr)
    tool_json = json.dumps({"tool": "calculator", "input": expr})
    return task, think, tool_json, result, answer

def make_search_cot():
    """Search example with explicit CoT reasoning."""
    key   = random.choice(list(KNOWLEDGE_BASE.keys()))
    value = KNOWLEDGE_BASE[key]

    phrasings = [
        (f"What is the {key}?",
         f"The task asks about the {key}. "
         f"I need the search tool with input \"{key}\"."),
        (f"Tell me the {key}.",
         f"The task wants the {key}. "
         f"I need the search tool with input \"{key}\"."),
        (f"Find the {key}.",
         f"The task asks me to find the {key}. "
         f"I need the search tool with input \"{key}\"."),
        (f"Look up the {key}.",
         f"The task asks to look up the {key}. "
         f"I need the search tool with input \"{key}\"."),
        (f"Search for the {key}.",
         f"The task asks to search for the {key}. "
         f"I need the search tool with input \"{key}\"."),
        (f"Can you find the {key}?",
         f"The task wants me to find the {key}. "
         f"I need the search tool with input \"{key}\"."),
        (f"I need to know the {key}.",
         f"The task needs the {key}. "
         f"I need the search tool with input \"{key}\"."),
        (f"Please look up the {key} for me.",
         f"The task asks for the {key}. "
         f"I need the search tool with input \"{key}\"."),
    ]

    task, think  = random.choice(phrasings)
    tool_json    = json.dumps({"tool": "search", "input": key})
    result       = value
    answer       = f"The {key} is {value}."
    return task, think, tool_json, result, answer

def make_transform_cot():
    """Transform example with explicit CoT reasoning."""
    texts = [
        "hello world", "python programming",
        "machine learning", "deep neural network",
        "artificial intelligence", "gradient descent",
        "recurrent transformer", "attention mechanism",
        "language model", "neural network",
        "open source code", "deep learning",
    ]
    text = random.choice(texts)

    ops = {
        "upper": (
            f"Convert '{text}' to uppercase.",
            f"The task asks to uppercase the text '{text}'. "
            f"I need the transform tool with op \"upper\" "
            f"and input \"{text}\".",
            text,
            f"'{text}' in uppercase is '{tool_transform('upper', text)}'."
        ),
        "lower": (
            f"Convert '{text.upper()}' to lowercase.",
            f"The task asks to lowercase the text '{text.upper()}'. "
            f"I need the transform tool with op \"lower\" "
            f"and input \"{text.upper()}\".",
            text.upper(),
            f"'{text.upper()}' in lowercase is "
            f"'{tool_transform('lower', text.upper())}'."
        ),
        "reverse": (
            f"Reverse the string '{text}'.",
            f"The task asks to reverse the string '{text}'. "
            f"I need the transform tool with op \"reverse\" "
            f"and input \"{text}\".",
            text,
            f"'{text}' reversed is '{tool_transform('reverse', text)}'."
        ),
        "count_words": (
            f"How many words are in '{text}'?",
            f"The task asks to count words in '{text}'. "
            f"I need the transform tool with op \"count_words\" "
            f"and input \"{text}\".",
            text,
            f"'{text}' has {tool_transform('count_words', text)} word(s)."
        ),
        "count_chars": (
            f"How many characters are in '{text}'?",
            f"The task asks to count characters in '{text}'. "
            f"I need the transform tool with op \"count_chars\" "
            f"and input \"{text}\".",
            text,
            f"'{text}' has {tool_transform('count_chars', text)} characters."
        ),
        "title": (
            f"Convert '{text}' to title case.",
            f"The task asks to title-case the text '{text}'. "
            f"I need the transform tool with op \"title\" "
            f"and input \"{text}\".",
            text,
            f"'{text}' in title case is "
            f"'{tool_transform('title', text)}'."
        ),
    }

    op = random.choice(list(ops.keys()))
    task, think, input_text, answer = ops[op]
    tool_json = json.dumps({
        "tool": "transform", "op": op, "input": input_text})
    result = tool_transform(op, input_text)
    return task, think, tool_json, result, answer

def make_example():
    """Generate one CoT training example."""
    generators = [
        make_calculator_cot,
        make_search_cot,
        make_transform_cot,
    ]
    return random.choice(generators)()

def format_cot_example(task, think, tool_json, result, answer):
    """
    Format with chain-of-thought reasoning step.
    The <think> tag teaches the model to reason before acting.
    """
    return (f"<task>{task}</task>"
            f"<think>{think}</think>"
            f"<tool>{tool_json}</tool>"
            f"<r>{result}</r>"
            f"<answer>{answer}</answer>")

def build_dataset(n_samples):
    sequences = []
    skipped   = 0
    for _ in range(n_samples):
        task, think, tool_json, result, answer = make_example()
        text   = format_cot_example(task, think, tool_json, result, answer)
        tokens = enc.encode(text)
        if len(tokens) < BLOCK_SIZE:
            padded = tokens + [0] * (BLOCK_SIZE - len(tokens))
            sequences.append(padded[:BLOCK_SIZE])
        else:
            skipped += 1
    if skipped > 0:
        print(f"  (skipped {skipped} examples > {BLOCK_SIZE} tokens)")
    return torch.tensor(sequences, dtype=torch.long)

print("\nBuilding chain-of-thought dataset...")
train_data = build_dataset(TRAIN_SAMPLES)
test_data  = build_dataset(TEST_SAMPLES)
print(f"  Train: {len(train_data):,} examples")
print(f"  Test:  {len(test_data):,}  examples")

# Show CoT examples so you can see the difference
print("\nSample CoT examples — notice the <think> step:")
for i in range(3):
    task, think, tool_json, result, answer = make_example()
    formatted = format_cot_example(task, think, tool_json, result, answer)
    print(f"\n  Example {i+1}:")
    print(f"    Task:   {task}")
    print(f"    Think:  {think}")
    print(f"    Tool:   {tool_json}")
    print(f"    Result: {result}")
    print(f"    Answer: {answer}")
    print(f"    Tokens: {len(enc.encode(formatted))}")

def get_batch(split="train"):
    source = train_data if split == "train" else test_data
    idx    = torch.randint(len(source), (BATCH_SIZE,))
    seqs   = source[idx].to(device)
    return seqs[:, :-1], seqs[:, 1:]


# ════════════════════════════════════════════════════════════
# PART 3 — RDT MODEL
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
    def __init__(self):
        super().__init__()
        self.embedding = EmbeddingLayer()
        self.prelude   = nn.Sequential(
            *[TransformerBlock() for _ in range(N_PRELUDE)])
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
                targets.reshape(B*T), ignore_index=0)
        return logits, loss

    @torch.no_grad()
    def generate_tokens(self, idx, max_new_tokens=150,
                        temperature=0.3, stop_strings=None,
                        n_loops=None):
        prompt_len = idx.shape[1]
        loops = n_loops or N_LOOP_STEPS
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond, n_loops=loops)
            logits = logits[:, -1, :] / temperature
            probs  = F.softmax(logits, dim=-1)
            next_t = torch.multinomial(probs, num_samples=1)
            idx    = torch.cat([idx, next_t], dim=1)
            if stop_strings:
                generated = enc.decode(
                    idx[0, prompt_len:].tolist())
                if any(s in generated for s in stop_strings):
                    break
        return enc.decode(idx[0, prompt_len:].tolist())

model = RDT().to(device)
params = sum(p.numel() for p in model.parameters())
print(f"\nRDT: {params:,} parameters")

x, y = get_batch()
_, loss = model(x, y)
print(f"Initial loss: {loss.item():.4f} (expected ~10.8)")


# ════════════════════════════════════════════════════════════
# PART 4 — FINE-TUNING
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("PART 4: Fine-Tuning with CoT")
print("=" * 55)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=FINETUNE_LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=FINETUNE_ITERS, eta_min=1e-5)

best_loss = float('inf')
start     = time.time()

print(f"\nConfig: {FINETUNE_ITERS} steps, "
      f"{TRAIN_SAMPLES} samples, "
      f"BLOCK_SIZE={BLOCK_SIZE}\n")

for step in range(FINETUNE_ITERS):
    x, y = get_batch("train")
    logits, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    model.enforce_stability()
    scheduler.step()

    if step % EVAL_INTERVAL == 0 or step == FINETUNE_ITERS - 1:
        model.eval()
        with torch.no_grad():
            xv, yv = get_batch("test")
            _, val_loss = model(xv, yv)
        model.train()

        saved = ""
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            torch.save(model.state_dict(), RESULTS_DIR / "best_rdt_agent.pt")
            saved = " ← saved"

        print(f"  step {step:4d} | "
              f"train: {loss.item():.4f} | "
              f"val: {val_loss.item():.4f} | "
              f"t: {time.time()-start:.0f}s{saved}")

print(f"\n✅ Fine-tuning complete. Best val loss: {best_loss:.4f}")


# ════════════════════════════════════════════════════════════
# PART 5 — EVALUATION
# ════════════════════════════════════════════════════════════
# IMPORTANT: generation prompt now includes <think>
# The model will output its reasoning THEN the tool call.
# We parse the tool call from after </think>.

print("\n" + "=" * 55)
print("PART 5: Evaluation")
print("=" * 55)

model.load_state_dict(
    torch.load(RESULTS_DIR / "best_rdt_agent.pt", map_location=device))
model.eval()

def generate_cot_tool_call(task: str) -> tuple:
    """
    Generate with CoT prompt.
    Returns (think_text, raw_tool_output, parsed_tool_call).
    The model outputs <think>...</think> then <tool>...</tool>
    """
    # Prompt includes <think> to trigger reasoning step
    prompt  = f"<task>{task}</task><think>"
    tokens  = enc.encode(prompt)
    idx     = torch.tensor(
        [tokens[-BLOCK_SIZE:]], dtype=torch.long, device=device)

    # Generate thinking step
    think_output = model.generate_tokens(
        idx, max_new_tokens=80, temperature=0.3,
        stop_strings=["</think>"])

    # Now generate tool call after thinking
    full_prompt = f"<task>{task}</task><think>{think_output}</think><tool>"
    tokens2     = enc.encode(full_prompt)
    idx2        = torch.tensor(
        [tokens2[-BLOCK_SIZE:]], dtype=torch.long, device=device)

    tool_output = model.generate_tokens(
        idx2, max_new_tokens=80, temperature=0.3,
        stop_strings=["</tool>"])

    # Parse JSON from tool output
    try:
        start = tool_output.find("{")
        end   = tool_output.rfind("}") + 1
        if start >= 0 and end > start:
            tc = json.loads(tool_output[start:end])
            return think_output, tool_output, tc
    except:
        pass
    return think_output, tool_output, {}

# 10 test tasks with correctness checks
test_tasks = [
    ("What is 42 multiplied by 7?",
     "calculator",
     lambda tc: "42" in tc.get("input","") and
                "7" in tc.get("input","")),

    ("What is the capital of France?",
     "search",
     lambda tc: "france" in tc.get("input","").lower()),

    ("Convert 'machine learning' to uppercase.",
     "transform",
     lambda tc: tc.get("op") == "upper" and
                "machine learning" in tc.get("input","").lower()),

    ("Calculate 100 divided by 4.",
     "calculator",
     lambda tc: "100" in tc.get("input","") and
                "4" in tc.get("input","")),

    ("What is the tallest mountain?",
     "search",
     lambda tc: "tallest mountain" in tc.get("input","").lower()),

    ("Reverse the string 'hello'.",
     "transform",
     lambda tc: tc.get("op") == "reverse" and
                "hello" in tc.get("input","").lower()),

    ("How many words are in 'the quick brown fox'?",
     "transform",
     lambda tc: tc.get("op") == "count_words" and
                "quick brown fox" in tc.get("input","").lower()),

    ("What is 237 plus 84?",
     "calculator",
     lambda tc: "237" in tc.get("input","") and
                "84" in tc.get("input","")),

    ("Find the inventor of the telephone.",
     "search",
     lambda tc: "telephone" in tc.get("input","").lower()),

    ("Convert 'recurrent transformer' to title case.",
     "transform",
     lambda tc: tc.get("op") == "title" and
                "recurrent transformer" in tc.get("input","").lower()),
]

print(f"\nEvaluating {len(test_tasks)} tasks with CoT generation:\n")
json_valid   = 0
tool_correct = 0
arg_correct  = 0

for task, expected_tool, check_fn in test_tasks:
    think, raw, tc = generate_cot_tool_call(task)

    is_json = bool(tc)
    is_tool = is_json and tc.get("tool") == expected_tool
    is_arg  = is_tool and check_fn(tc)

    if is_json: json_valid   += 1
    if is_tool: tool_correct += 1
    if is_arg:  arg_correct  += 1

    status = "✅" if is_arg else ("⚠️" if is_tool else "❌")
    result = execute_tool(tc) if is_json else "N/A"

    print(f"  {status} {task}")
    print(f"     Think: {think[:80].strip()}")
    print(f"     Tool:  {tc}")
    print(f"     Result:{result}\n")

n = len(test_tasks)
print(f"Progress across all lessons:")
print(f"  Lesson 1   — arg accuracy: 29%  (2/7)")
print(f"  Lesson 1b  — arg accuracy: 60%  (6/10)")
print(f"  Lesson 1c  — arg accuracy: {arg_correct/n*100:.0f}%  ({arg_correct}/{n})")
print(f"\n  JSON valid:   {json_valid}/{n}")
print(f"  Tool correct: {tool_correct}/{n}")
print(f"  Arg correct:  {arg_correct}/{n}")


# ════════════════════════════════════════════════════════════
# PART 6 — MULTI-STEP CoT EXAMPLES
# ════════════════════════════════════════════════════════════
# Show the model thinking step by step on a few tasks.
# This is what makes the demo compelling —
# you can see the model reasoning before acting.

print("\n" + "=" * 55)
print("PART 6: Multi-Step CoT Demo")
print("=" * 55)

demo_tasks = [
    "What is the capital of India?",
    "What is 15 times 8?",
    "Reverse the string 'transformer'.",
    "What is the year of the moon landing?",
    "Convert 'deep learning' to uppercase.",
]

print("\nShowing full CoT reasoning for each task:\n")
for task in demo_tasks:
    think, raw, tc = generate_cot_tool_call(task)
    result = execute_tool(tc) if tc else "N/A"

    print(f"  Task:   {task}")
    print(f"  Think:  {think[:100].strip()}")
    print(f"  Tool:   {tc}")
    print(f"  Result: {result}")
