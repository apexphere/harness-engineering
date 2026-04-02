# Harness Engineering

A pilot project for building a production **harness** around `claude -p` (Claude Code
headless mode) — the scaffolding that makes AI systems reliable and safe without
a human in the loop.

## What is Harness Engineering?

An AI model alone is just a function: text in, text out. The **harness** is everything
you build around it to turn that function into a product:

```
┌─────────────────────────────────────────────────┐
│                   HARNESS                       │
│                                                 │
│  ┌───────────┐  ┌───────┐  ┌───────────────┐   │
│  │  Prompts   │  │ Tools │  │  Guardrails   │   │
│  └─────┬─────┘  └───┬───┘  └──────┬────────┘   │
│        │            │              │             │
│        ▼            ▼              ▼             │
│  ┌─────────────────────────────────────────┐    │
│  │             AI MODEL (LLM)              │    │
│  └─────────────────────────────────────────┘    │
│        │            │              │             │
│        ▼            ▼              ▼             │
│  ┌───────────┐  ┌───────┐  ┌───────────────┐   │
│  │  Parsing   │  │ Evals │  │ Orchestration │   │
│  └───────────┘  └───────┘  └───────────────┘   │
│                                                 │
└─────────────────────────────────────────────────┘
```

In production, there's no human to catch hallucinations, re-ask when output is
bad, or judge quality. The harness does that. It's a **deterministic containment
structure** around a non-deterministic core — your code bounds the LLM's behavior
so the system is reliable without a human in the loop.

## Architecture

The pipeline runs on `claude -p` (Claude Code headless mode) — non-interactive,
no human required, uses your subscription. This is the production runtime.

```
  Trigger (API, cron, webhook, CLI)
         │
         ▼
┌─────── PIPELINE (your code) ────────────────────────────────┐
│                                                              │
│  Layer 1: Prompts ──► Layer 2: Retrieval ──► [claude -p] ──►│
│  (deterministic)      (deterministic)        (non-determ)    │
│                                                     │        │
│  ◄── Layer 3: Parsing ◄── Layer 4: Guardrails ◄────┘        │
│      (deterministic)       (deterministic)                   │
│             │                                                │
│             ▼                                                │
│      Layer 5: Evals ──► Response with metadata               │
│      (deterministic)    (observable intermediate state)       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
         │
         ▼
       Output
```

Your code controls what goes in (prompt, schema, context) and validates what
comes out (parsing, guardrails, evals). `claude -p` is the non-deterministic
component — bounded, not controlled.

## The Six Layers

| # | Layer | What it does | File |
|---|-------|-------------|------|
| 1 | **Prompts** | System prompts, templates, output format instructions | `src/prompts.py` |
| 2 | **Tools** | Retrieval — search your notes via vector embeddings | `src/tools.py` |
| 3 | **Parsing** | Extract structured JSON from model output | `src/parsing.py` |
| 4 | **Guardrails** | Input validation, hallucination filter | `src/guardrails.py` |
| 5 | **Evals** | Golden set scoring, live property checks | `src/evals.py` |
| 6 | **Orchestration** | Backend dispatch, retries, control flow | `src/orchestration.py` |

The LLM call is abstracted in `src/backend.py`. Default backend is `claude -p`
(no API key needed — uses your subscription).

## Quick Start

```bash
git clone <this-repo>
cd harness-engineering
pip install -e ".[dev]"

python -m src.main --init-sample                             # Generate sample notes
python -m src.main --ingest sample_notes/                    # Load into vector DB
python -m src.main --stage 6 --verbose "What is attention?"  # Run the full pipeline
```

The `--verbose` flag shows every layer running — the containment structure
doing its job around the LLM call:

```
  [1/6] prompts...
         → 0 chars, confidence=0.00, sources=0
  [2/6] tools...
         → 0 chars, confidence=0.00, sources=3
  [3/6] parsing...
         → 142 chars, confidence=0.85, sources=2
  [4/6] guardrails...
         → 142 chars, confidence=0.85, sources=2
  [5/6] evals...
         → 142 chars, confidence=0.85, sources=2
  [6/6] orchestration...
         → 203 chars, confidence=0.85, sources=2
```

## Learning Path

Work through each layer in order. Each file has a `# LEARN:` comment at
the top explaining the concept, followed by working code.

1. **`src/prompts.py`** — how system prompts shape model behavior
2. **`src/tools.py`** — giving the model access to your data via retrieval
3. **`src/parsing.py`** — extracting structured data from free-form text
4. **`src/guardrails.py`** — catching hallucinations programmatically
5. **`src/evals.py`** — testing probabilistic systems with property checks
6. **`src/orchestration.py`** — the control flow, retries, and backend dispatch
7. **`src/backend.py`** — the LLM call abstraction (how `claude -p` works)

Use `--stage N` to activate layers progressively and `--compare` to see the
difference each layer makes on the same question:

```bash
python -m src.main --stage 1 "What is attention?"   # Prompt only
python -m src.main --stage 3 "What is attention?"   # + retrieval + parsing
python -m src.main --stage 6 "What is attention?"   # Full pipeline
python -m src.main --compare "What is attention?"   # All stages side by side
```

## CLI Reference

```bash
python -m src.main --init-sample                    # Generate sample notes
python -m src.main --ingest ~/notes/                # Ingest your real notes
python -m src.main --stage N "question"             # Query at stage 1-6
python -m src.main --compare "question"             # Compare all stages
python -m src.main --eval --stage N                 # Run golden set evals
python -m src.main --journal                        # View learning journal
python -m src.main --verbose "question"             # Layer-by-layer trace
python -m src.main --cost "question"                # Show token costs
python -m src.main --session ID "follow-up"         # Resume a conversation
```

## Running Tests

```bash
pytest tests/ -v          # All 115 tests
pytest tests/ -k backend  # Backend abstraction tests
pytest tests/ -k layers   # Just the layer tests
pytest tests/ -k stages   # Just the stage controller tests
```
