import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import re
import time
import tiktoken
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else \
         "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}\n")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

enc = tiktoken.get_encoding("gpt2")

# ── HYPERPARAMETERS (must match Lesson 1) ───────────────────
VOCAB_SIZE   = 50_257
BLOCK_SIZE   = 192
N_EMBED      = 384
N_HEADS      = 6
HEAD_SIZE    = N_EMBED // N_HEADS
N_LOOP_STEPS = 6
N_PRELUDE    = 1
N_CODA       = 1
DROPOUT      = 0.1
FF_MULT      = 4
SPECTRAL_CAP = 0.9


# ════════════════════════════════════════════════════════════
# PART 1 — TOOLS
# ════════════════════════════════════════════════════════════

KNOWLEDGE_BASE = {
    "capital of france":      "Paris",
    "capital of japan":       "Tokyo",
    "capital of germany":     "Berlin",
    "capital of australia":   "Canberra",
    "capital of brazil":      "Brasília",
    "capital of india":       "New Delhi",
    "capital of canada":      "Ottawa",
    "capital of argentina":   "Buenos Aires",
    "largest planet":         "Jupiter",
    "closest planet to sun":  "Mercury",
    "boiling point of water": "100",
    "freezing point of water":"0",
    "number of continents":   "7",
    "number of oceans":       "5",
    "inventor of telephone":  "Alexander Graham Bell",
    "author of hamlet":       "William Shakespeare",
    "year of moon landing":   "1969",
    "tallest mountain":       "Mount Everest",
    "longest river":          "Nile River",
}

def tool_calculator(expression):
    try:
        allowed = set('0123456789+-*/()., ')
        if not all(c in allowed for c in expression):
            return "Error: invalid expression"
        result = eval(expression, {"__builtins__": {}}, {})
        return str(round(float(result), 4))
    except Exception as e:
        return f"Error: {e}"

def tool_search(query):
    key = query.lower().strip()
    if key in KNOWLEDGE_BASE:
        return KNOWLEDGE_BASE[key]
    for kb_key, value in KNOWLEDGE_BASE.items():
        if key in kb_key or kb_key in key:
            return value
    return f"Not found: '{query}'"

def tool_transform(op, text):
    ops = {
        "upper":       text.upper(),
        "lower":       text.lower(),
        "reverse":     text[::-1],
        "count_words": str(len(text.split())),
        "count_chars": str(len(text)),
        "title":       text.title(),
    }
    return ops.get(op.lower().strip(), f"Unknown op: {op}")

def execute_tool(tool_call):
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

print("   Tools loaded\n")


# ════════════════════════════════════════════════════════════
# PART 2 — RDT MODEL
# ════════════════════════════════════════════════════════════

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
                generated = enc.decode(idx[0, prompt_len:].tolist())
                if any(s in generated for s in stop_strings):
                    break
        return enc.decode(idx[0, prompt_len:].tolist())

# Load model
print("=" * 55)
print("PART 2: Loading Fine-Tuned RDT")
print("=" * 55)

model = RDT().to(device)
model.load_state_dict(
    torch.load(RESULTS_DIR / "best_rdt_agent.pt", map_location=device))
model.eval()
params = sum(p.numel() for p in model.parameters())
print(f"\n✅ Loaded: {params:,} parameters")


# ════════════════════════════════════════════════════════════
# PART 3 — AGENT LOOP
# ════════════════════════════════════════════════════════════

def parse_tool_call(text):
    try:
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except:
        pass
    return {}

def parse_answer(text):
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def run_agent(task, max_steps=4, n_loops=None, verbose=True):
    """
    Full agent loop.
    Runs until model produces <answer> or max_steps reached.
    """
    if verbose:
        print(f"\n{'='*52}")
        print(f"TASK: {task}")
        print(f"{'='*52}")

    context    = f"<task>{task}</task>"
    steps      = []
    tool_calls = []
    start_time = time.time()

    for step in range(max_steps):

        # Generate tool call attempt
        tool_prompt = context + "<tool>"
        tool_tokens = enc.encode(tool_prompt)
        tool_idx    = torch.tensor(
            [tool_tokens[-BLOCK_SIZE:]], dtype=torch.long, device=device)
        tool_output = model.generate_tokens(
            tool_idx, max_new_tokens=60, temperature=0.3,
            stop_strings=["</tool>", "<answer>"],
            n_loops=n_loops)

        # Generate answer attempt
        ans_prompt = context + "<answer>"
        ans_tokens = enc.encode(ans_prompt)
        ans_idx    = torch.tensor(
            [ans_tokens[-BLOCK_SIZE:]], dtype=torch.long, device=device)
        ans_output = model.generate_tokens(
            ans_idx, max_new_tokens=60, temperature=0.3,
            stop_strings=["</answer>"],
            n_loops=n_loops)

        # Parse tool call
        tool_call = parse_tool_call(tool_output)
        use_tool  = (bool(tool_call) and
                     "tool" in tool_call and
                     step < max_steps - 1)

        if use_tool:
            result = execute_tool(tool_call)
            context += (f"<tool>{json.dumps(tool_call)}</tool>"
                       f"<r>{result}</r>")
            tool_calls.append(tool_call)
            steps.append({"step": step+1, "type": "tool",
                          "tool": tool_call, "result": result})
            if verbose:
                print(f"\n  Step {step+1} [TOOL]")
                print(f"    Call:   {tool_call}")
                print(f"    Result: {result}")
        else:
            answer  = parse_answer(ans_output)
            elapsed = time.time() - start_time
            steps.append({"step": step+1, "type": "answer",
                          "answer": answer})
            if verbose:
                print(f"\n  Step {step+1} [ANSWER]")
                print(f"    {answer}")
                print(f"\n  Done in {elapsed:.1f}s "
                      f"({len(tool_calls)} tool call(s))")
            return {"answer": answer, "steps": steps,
                    "tool_calls": tool_calls,
                    "n_steps": len(tool_calls),
                    "success": True, "time": elapsed}

    return {"answer": "Max steps reached.", "steps": steps,
            "tool_calls": tool_calls,
            "n_steps": len(tool_calls),
            "success": False,
            "time": time.time() - start_time}

print("\nAgent loop ready.\n")


# ════════════════════════════════════════════════════════════
# PART 4 — SINGLE STEP TASKS
# ════════════════════════════════════════════════════════════

print("=" * 55)
print("PART 4: Single-Step Tasks")
print("=" * 55)

single_tasks = [
    "What is 15 multiplied by 8?",
    "What is the capital of Germany?",
    "Convert 'deep learning' to uppercase.",
    "What is the largest planet?",
]

single_results = []
for task in single_tasks:
    r = run_agent(task, max_steps=3)
    single_results.append(r)

s_success = sum(1 for r in single_results if r['success'])
print(f"\nSingle-step: {s_success}/{len(single_tasks)} completed")


# ════════════════════════════════════════════════════════════
# PART 5 — MULTI-STEP TASKS
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("PART 5: Multi-Step Tasks (chained tool calls)")
print("=" * 55)

multi_tasks = [
    "Find the capital of Japan and convert it to uppercase.",
    "Find the boiling point of water and add 273 to it.",
    "What is the tallest mountain? Reverse its name.",
    "Calculate 7 times 6, then reverse the word 'result'.",
]

multi_results = []
for task in multi_tasks:
    r = run_agent(task, max_steps=5)
    multi_results.append(r)

m_success = sum(1 for r in multi_results if r['success'])
avg_steps = sum(r['n_steps'] for r in multi_results)/len(multi_results)
print(f"\nMulti-step: {m_success}/{len(multi_tasks)} completed")
print(f"Avg tool calls: {avg_steps:.1f}")


# ════════════════════════════════════════════════════════════
# PART 6 — LOOP DEPTH vs COMPLEXITY
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("PART 6: Loop Depth vs Task Complexity")
print("=" * 55)
print("Does more loops help more on complex tasks?")

def check_tool_valid(task, expected_tool, n_loops):
    prompt  = f"<task>{task}</task><tool>"
    tokens  = enc.encode(prompt)
    idx     = torch.tensor(
        [tokens[-BLOCK_SIZE:]], dtype=torch.long, device=device)
    output  = model.generate_tokens(
        idx, max_new_tokens=60, temperature=0.3,
        stop_strings=["</tool>"], n_loops=n_loops)
    tc = parse_tool_call(output)
    return bool(tc) and tc.get("tool") == expected_tool

simple_tasks = [
    ("What is 5 plus 3?",            "calculator"),
    ("What is the capital of India?", "search"),
    ("Uppercase 'hello'.",            "transform"),
]
complex_tasks = [
    ("Find boiling point of water and add 273.",    "calculator"),
    ("Capital of Canada reversed then uppercased.", "transform"),
    ("Year of moon landing multiplied by 2.",       "calculator"),
]

loop_counts = [2, 4, 6, 8, 10]
print(f"\n{'Loops':>6} | {'Simple (%)':>11} | {'Complex (%)':>12}")
print("-" * 35)

simple_scores  = {}
complex_scores = {}

for n_loops in loop_counts:
    s = sum(1 for t, e in simple_tasks
            if check_tool_valid(t, e, n_loops))
    c = sum(1 for t, e in complex_tasks
            if check_tool_valid(t, e, n_loops))
    simple_scores[n_loops]  = s
    complex_scores[n_loops] = c
    s_pct = s/len(simple_tasks)*100
    c_pct = c/len(complex_tasks)*100
    print(f"{n_loops:>6} | {s_pct:>10.0f}% | {c_pct:>11.0f}%")

s_trend = simple_scores[10]  - simple_scores[2]
c_trend = complex_scores[10] - complex_scores[2]
print(f"\nSimple  2→10 loops: {s_trend:+d} tasks")
print(f"Complex 2→10 loops: {c_trend:+d} tasks")
if c_trend > s_trend:
    print("✅ Complex tasks benefit MORE from more loops")
elif c_trend == s_trend:
    print("→ Both respond equally to more loops")
else:
    print("→ Marginal — more fine-tuning would sharpen this")


# ════════════════════════════════════════════════════════════
# PART 7 — CUSTOM DEMO TASKS
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("PART 7: Custom Demo Tasks")
print("=" * 55)

demo_tasks = [
    "What is 2025 divided by 5?",
    "Find the inventor of the telephone and reverse their name.",
    "What is the year of the moon landing? "
    "Multiply it by 2 and subtract 3000.",
]

for task in demo_tasks:
    run_agent(task, max_steps=4, verbose=True)

# ── Final summary ────────────────────────────────────────────
total   = len(single_tasks) + len(multi_tasks)
success = s_success + m_success
