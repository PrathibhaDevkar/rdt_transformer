# RDT Transformer

A from-scratch implementation of the **Recurrent Depth Transformer (RDT)** architecture, built as a progressive learning series. Each phase and lesson builds on the previous, going from basic tokenization all the way to a reasoning agent.

## What is RDT?

An RDT is a transformer that processes input through *recurrent loop steps* rather than simply stacking more layers. At each step the model refines a hidden state using the update rule:

```
h(t+1) = A·h(t) + B·e + Transformer(h(t), e)
```

This lets the model "think longer" about hard problems without increasing parameter count proportionally.

## Project Structure

```
rdt_transformer/
├── data/               # TinyShakespeare dataset (included; re-downloaded if missing)
├── results/            # Benchmark plots and JSON results
├── phase1/             # GPT baseline — build a standard transformer
│   ├── lesson1_tokenizer.py
│   ├── lesson2_embeddings.py
│   ├── lesson3_attention.py
│   ├── lesson4_feedforward.py
│   └── lesson5_training.py
├── phase2/             # RDT core — recurrent depth architecture
│   ├── lesson1_rdt.py
│   ├── lesson2_benchmark.py
│   ├── lesson2b_benchmark.py
│   └── lesson2c_curriculum.py
└── phase3/             # Fine-tuning, chain-of-thought, and agents
    ├── lesson1_finetune.py
    ├── lesson1b_finetune.py
    ├── lesson1c_cot.py
    └── lesson2_agent.py
```

## Phases

### Phase 1 - GPT Baseline
Builds a standard GPT-style transformer from scratch on TinyShakespeare. Covers tokenization (BPE via tiktoken), positional embeddings, multi-head self-attention, feed-forward layers, and full training.

### Phase 2 - RDT Architecture
Introduces the recurrent block with learned `A` and `B` matrices and a fixed-iteration loop. Benchmarks RDT vs GPT on multi-hop reasoning tasks across hop depths 1–10. Includes curriculum learning experiments.

### Phase 3 - Fine-tuning and Agents
Fine-tunes the trained RDT on reasoning datasets, adds chain-of-thought (CoT) prompting, and builds a tool-calling agent loop backed by the RDT model.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ and PyTorch 2.0+. MPS (Apple Silicon) and CUDA are both supported; falls back to CPU automatically.

## Running

Run lessons in order within each phase:

```bash
python phase1/lesson1_tokenizer.py
python phase1/lesson2_embeddings.py
# ... and so on
```

The dataset (`data/shakespeare.txt`) is included in the repo, so no download is needed. If it's ever missing, `phase1/lesson1_tokenizer.py` will fetch it automatically on first run.

## Results

Benchmark plots and raw JSON results are in `results/`. Across the three benchmark variants, the RDT matches the 6-layer GPT baseline with ~47% fewer parameters (0.64M vs 1.20M, 0.65M vs 1.21M, and 2.54M vs 4.78M). On the first (a fixed-rule chain task) both models score 100% at 1–5 hops and fall to chance at the unseen depths 7 and 10. On the two harder in-context lookup tasks both models stay only modestly above the ~17% chance level, with the RDT within about 4 points of GPT either way (500 test examples per hop depth, one run each). Running extra loops at inference did not change accuracy in any of them, so the benchmarks show parameter efficiency at comparable accuracy, not a reasoning advantage.
