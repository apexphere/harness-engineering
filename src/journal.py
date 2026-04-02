# LEARN: Learning Journal — Your Course Writes Itself
#
# Every time you run --eval, the scores are logged. The journal reads these
# logs and generates markdown entries showing what changed at each stage.
# The auto-generated sections (scores, stats) are filled in. The Observations
# section is left blank for you to fill in with your own insights.

import json
import os
from datetime import datetime


JOURNAL_DIR = "journal"
EVAL_HISTORY_FILE = os.path.join(JOURNAL_DIR, "eval_history.jsonl")


def log_eval_run(stage: int, scores: list[dict], avg_score: float) -> None:
    """Append an eval run to the history file."""
    os.makedirs(JOURNAL_DIR, exist_ok=True)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "stage": stage,
        "avg_score": round(avg_score, 2),
        "total_questions": len(scores),
        "passed": sum(1 for s in scores if s.get("passed", False)),
        "scores": scores,
    }

    with open(EVAL_HISTORY_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def read_eval_history() -> list[dict]:
    """Read all eval history entries."""
    if not os.path.exists(EVAL_HISTORY_FILE):
        return []

    entries = []
    with open(EVAL_HISTORY_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def generate_journal_entry(stage: int, current_scores: list[dict],
                           previous_entry: dict | None = None) -> str:
    """Generate a markdown journal entry for a stage.

    Auto-fills: Score Impact, What Changed, Queries That Changed Most, Stats.
    Leaves Observations blank for the learner.
    """
    now = datetime.now().strftime("%Y-%m-%d")
    stage_names = {
        1: "Prompts", 2: "Tools", 3: "Parsing",
        4: "Guardrails", 5: "Evals", 6: "Orchestration",
    }
    stage_name = stage_names.get(stage, f"Stage {stage}")

    avg_score = sum(s.get("score", 0) for s in current_scores) / max(len(current_scores), 1)
    passed = sum(1 for s in current_scores if s.get("passed", False))
    total = len(current_scores)

    lines = [f"# Stage {stage}: {stage_name} — {now}", ""]

    # Score Impact
    lines.append("## Score Impact")
    if previous_entry:
        prev_avg = previous_entry.get("avg_score", 0)
        delta = avg_score - prev_avg
        pct = (delta / prev_avg * 100) if prev_avg > 0 else 0
        lines.append(f"- Before (stage {previous_entry.get('stage', '?')}): {prev_avg:.1f}/5")
        lines.append(f"- After (stage {stage}): {avg_score:.1f}/5")
        lines.append(f"- Delta: {delta:+.1f} ({pct:+.0f}%)")
    else:
        lines.append(f"- Score: {avg_score:.1f}/5 (first measurement)")
    lines.append("")

    # What Changed
    lines.append("## What Changed")
    lines.append(f"- Added {stage_name.lower()} layer to the pipeline")
    lines.append("")

    # Observations (blank for learner)
    lines.append("## Observations")
    lines.append("<!-- Fill this in yourself — what surprised you? -->")
    lines.append("")
    lines.append("")

    # Queries That Changed Most
    lines.append("## Queries That Changed Most")
    for s in current_scores:
        if s.get("score", 0) < 3:
            checks_str = ", ".join(
                f"{k}={'✓' if v else '✗'}"
                for k, v in s.get("checks", {}).items()
            )
            lines.append(f"- {s['id']}: {s['score']}/5 — {checks_str}")
    if all(s.get("score", 0) >= 3 for s in current_scores):
        lines.append("- All questions scored 3/5 or higher!")
    lines.append("")

    # Stats
    lines.append("## Stats")
    lines.append(f"- Queries run: {total} | Pass rate: {passed}/{total} ({passed/max(total,1)*100:.0f}%)")
    lines.append("")

    return "\n".join(lines)


def write_journal_entry(stage: int, content: str) -> str:
    """Write a journal entry to disk. Returns the file path."""
    os.makedirs(JOURNAL_DIR, exist_ok=True)
    filename = f"stage_{stage}.md"
    path = os.path.join(JOURNAL_DIR, filename)
    with open(path, "w") as f:
        f.write(content)
    return path


def show_journal() -> str:
    """Read and format all journal entries for display."""
    if not os.path.isdir(JOURNAL_DIR):
        return "No journal entries yet. Run --eval to generate them."

    entries = []
    for filename in sorted(os.listdir(JOURNAL_DIR)):
        if filename.startswith("stage_") and filename.endswith(".md"):
            path = os.path.join(JOURNAL_DIR, filename)
            with open(path) as f:
                entries.append(f.read())

    if not entries:
        return "No journal entries yet. Run --eval to generate them."

    return "\n\n---\n\n".join(entries)
