# TODOs

## P2: --playground Interactive REPL

**What:** Interactive REPL mode where you modify the system prompt on the fly and see
how it changes the model's behavior. Type a question, see the answer, tweak the prompt,
ask again. No need to edit files and re-run.

**Why:** Prompt iteration is where harness engineers spend 80% of their time in practice.
Making this loop tight teaches the most important skill.

**Context:** Deferred from V1 during CEO review because it's a different interaction model
(REPL vs CLI) and medium effort. The current CLI works well for structured experiments
(--stage, --compare, --eval) but the prompt iteration loop requires rapid feedback.

**Effort:** M (human: ~4 hours / CC: ~20 min)

**Depends on:** V1 implementation complete. The REPL would use the same pipeline
interface and stage controller, just with an interactive prompt editor. The `--session`
flag and `ClaudeCodeBackend`'s `--resume` support provide the foundation for
multi-turn conversation continuity.
