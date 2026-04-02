# TODOs

## P1: Productionize the Harness

The scaffold is working (6 layers, 115 tests, `claude -p` backend). These are
the gaps between "working pilot" and "production-grade harness."

### Structured Logging

**What:** Replace `--verbose` stdout printing with structured JSON logging that
can feed into monitoring systems (CloudWatch, Datadog, etc.).

**Why:** In production there's no terminal. You need logs that are queryable,
alertable, and dashboardable. Each layer transition should emit a structured
log entry with layer name, duration, input/output sizes, confidence delta.

**Effort:** S

### Response Caching

**What:** Cache LLM responses in `.cache/`, keyed by `(query_hash, stage, system_prompt_hash)`.
Auto-invalidate when layer source files or the system prompt changes. `--no-cache` flag
to force fresh calls. `--eval` always bypasses cache.

**Why:** `claude -p` subprocess calls are slow (~5-10s each). During development and
eval runs, you ask the same questions repeatedly. Caching avoids burning subscription
quota and speeds up iteration.

**Effort:** M

### Stronger Hallucination Detection

**What:** Replace the noun-phrase substring matcher with something that actually works.
Options: embedding similarity between claims and source chunks, or a second `claude -p`
call that checks "is this claim supported by the context?"

**Why:** The current filter (`extract_noun_phrases` + `check_against_chunks`) catches
obvious fabrications but misses subtle hallucinations — paraphrased claims, correct
facts not in the source, plausible-sounding nonsense.

**Effort:** M

### Prompt Versioning

**What:** Track system prompt changes and their impact on eval scores. Store prompt
versions in `journal/prompt_history.jsonl` with timestamp, prompt hash, and eval
score. `--journal` shows score trends over time.

**Why:** The system prompt is the most-changed part of any harness. Without version
tracking, you can't tell which prompt change improved or degraded quality.

**Effort:** S

### Concurrency / Batch Processing

**What:** Run multiple queries through the pipeline concurrently. `--eval` should
run golden set questions in parallel (asyncio + subprocess). `--compare` should
run all 6 stages in parallel.

**Why:** Sequential `claude -p` calls make eval runs take minutes. Parallel execution
makes development iteration fast enough to actually use.

**Effort:** M

### Data Source Abstraction

**What:** Abstract the retrieval layer so it works with sources beyond Chroma +
markdown files. Interface: `search(query, n_results) -> list[Chunk]`. Implementations:
Chroma (current), filesystem, SQLite, external API.

**Why:** A production harness needs to work with whatever data the product requires,
not just local markdown files.

**Effort:** M

---

## P2: --playground Interactive REPL

**What:** Interactive REPL mode where you modify the system prompt on the fly and see
how it changes the model's behavior. Type a question, see the answer, tweak the prompt,
ask again. No need to edit files and re-run.

**Why:** Prompt iteration is where harness engineers spend 80% of their time in practice.
Making this loop tight teaches the most important skill.

**Context:** Deferred from V1 during CEO review because it's a different interaction model
(REPL vs CLI) and medium effort. The current CLI works well for structured experiments
(--stage, --compare, --eval) but the prompt iteration loop requires rapid feedback.

**Effort:** M

**Depends on:** The `--session` flag and `ClaudeCodeBackend`'s `--resume` support
provide the foundation for multi-turn conversation continuity.
