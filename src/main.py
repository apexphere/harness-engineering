# LEARN: The Harness Lab CLI
#
# This wires all layers together into a CLI with progressive unlocking.
# Run with: python -m src.main --help
#
# Key commands:
#   --stage N "question"     Run at a specific stage (1-6)
#   --compare "question"     Compare all stages side by side
#   --eval --stage N         Run the golden set at a stage
#   --ingest ~/notes/        Ingest your notes into Chroma
#   --init-sample            Generate sample notes + golden set
#   --journal                View the learning journal
#   --verbose                Show layer-by-layer trace
#   --cost                   Show token/dollar costs

import argparse
import json
import os
import sys
import time

from src.compare import run_compare, format_comparison_table
from src.evals import load_golden_set, score_response
from src.ingestion import ingest_directory
from src.journal import (
    log_eval_run,
    read_eval_history,
    generate_journal_entry,
    write_journal_entry,
    show_journal,
)
from src.sample_notes import generate_samples
from src.stages import register_real_layers, run_pipeline


# Haiku pricing (per 1M tokens)
HAIKU_INPUT_COST = 0.80   # $/1M input tokens
HAIKU_OUTPUT_COST = 4.00  # $/1M output tokens


def format_cost(response) -> str:
    """Format token usage and estimated cost."""
    tokens_in = int(response.get_metadata("tokens_in", "0"))
    tokens_out = int(response.get_metadata("tokens_out", "0"))
    cost = (tokens_in * HAIKU_INPUT_COST + tokens_out * HAIKU_OUTPUT_COST) / 1_000_000
    return f"${cost:.4f} ({tokens_in} in + {tokens_out} out)"


def cmd_query(args):
    """Run a query at a specific stage."""
    register_real_layers()
    config = _build_config(args)

    start = time.time()
    response = run_pipeline(args.query, stage=args.stage, config=config)
    elapsed = time.time() - start

    print(response.text)

    if response.sources:
        print(f"\nSources: {', '.join(s.file for s in response.sources)}")

    if response.gaps:
        print(f"Gaps: {', '.join(response.gaps)}")

    if args.cost:
        print(f"\nCost: {format_cost(response)} | {elapsed:.1f}s")


def cmd_compare(args):
    """Run compare mode across all stages."""
    register_real_layers()
    config = _build_config(args)

    print(f"Comparing: {args.query}\n")
    results = run_compare(args.query, config)
    print(format_comparison_table(results))

    if args.cost:
        total_tokens = sum(
            int(r["response"].get_metadata("tokens_out", "0"))
            for r in results
        )
        total_time = sum(r["latency_ms"] for r in results)
        print(f"\nTotal: {total_tokens} tokens, {total_time:.0f}ms")


def cmd_eval(args):
    """Run the golden set evaluation."""
    register_real_layers()
    config = _build_config(args)

    try:
        questions = load_golden_set(args.golden_set)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Running golden set ({len(questions)} questions) at stage {args.stage}...\n")

    all_scores = []
    for q in questions:
        response = run_pipeline(q["question"], stage=args.stage, config=dict(config))
        cited = [s.file for s in response.sources]
        result = score_response(q, response.text, cited)
        all_scores.append(result)

        status = "PASS" if result["passed"] else "FAIL"
        checks_detail = ""
        if not result["passed"]:
            failed = [k for k, v in result["checks"].items() if not v]
            checks_detail = f" — missing: {failed}"
        print(f"  [{status}] {result['id']}  {result['score']}/5  \"{q['question'][:50]}\"{checks_detail}")

    avg = sum(s["score"] for s in all_scores) / len(all_scores)
    passed = sum(1 for s in all_scores if s["passed"])
    print(f"\n{'─' * 40}")
    print(f"Score: {avg:.1f}/5 | Pass: {passed}/{len(all_scores)} ({passed/len(all_scores)*100:.0f}%)")

    # Log to eval history
    log_eval_run(args.stage, all_scores, avg)

    # Generate journal entry
    history = read_eval_history()
    previous = None
    for entry in reversed(history[:-1]):  # skip current
        if entry["stage"] < args.stage:
            previous = entry
            break

    content = generate_journal_entry(args.stage, all_scores, previous)
    path = write_journal_entry(args.stage, content)
    print(f"\nJournal entry written: {path}")


def cmd_ingest(args):
    """Ingest notes from a directory."""
    print(f"Ingesting notes from: {args.directory}")
    try:
        result = ingest_directory(args.directory, db_path=args.db_path)
    except (FileNotFoundError, ImportError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"  Files processed: {result['files_processed']}")
    print(f"  Chunks stored: {result['chunks_stored']}")
    if result.get("files_skipped"):
        print(f"  Skipped: {', '.join(result['files_skipped'])}")
    if result.get("message"):
        print(f"  {result['message']}")


def cmd_init_sample(args):
    """Generate sample notes and golden set."""
    out_dir = generate_samples()
    print(f"Sample notes generated in: {out_dir}/")
    print(f"Golden set generated: golden_set.json")
    print(f"\nNext steps:")
    print(f"  python -m src.main --ingest {out_dir}/")
    print(f"  python -m src.main --stage 1 \"What is self-attention?\"")


def cmd_journal(args):
    """Show the learning journal."""
    print(show_journal())


def _build_config(args) -> dict:
    """Build the config dict from CLI args."""
    config = {
        "db_path": getattr(args, "db_path", ".chroma"),
        "verbose": getattr(args, "verbose", False),
    }

    # Session continuity via --resume
    session_id = getattr(args, "session", None)
    if session_id is not None:
        config["session_id"] = session_id

    return config


def main():
    parser = argparse.ArgumentParser(
        description="The Harness Lab — Learn harness engineering by building a personal knowledge assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python -m src.main --init-sample                          Generate sample notes
  python -m src.main --ingest sample_notes/                 Ingest into Chroma
  python -m src.main --stage 1 "What is attention?"         Query at stage 1
  python -m src.main --stage 6 "What is attention?"         Query at stage 6
  python -m src.main --compare "What is attention?"         Compare all stages
  python -m src.main --eval --stage 4                       Run golden set evals
  python -m src.main --journal                              View learning journal
""",
    )

    # Global flags
    parser.add_argument("--verbose", "-v", action="store_true", help="Show layer-by-layer trace")
    parser.add_argument("--cost", action="store_true", help="Show token/dollar costs")
    parser.add_argument("--db-path", default=".chroma", help="Chroma database path")
    parser.add_argument("--session", metavar="ID", help="Resume a conversation by session ID")

    # Query mode (default)
    parser.add_argument("--stage", type=int, default=6, help="Stage to run (1-6)")
    parser.add_argument("query", nargs="?", help="Question to ask")

    # Compare mode
    parser.add_argument("--compare", metavar="QUERY", help="Compare all stages for a query")

    # Eval mode
    parser.add_argument("--eval", action="store_true", help="Run golden set evaluation")
    parser.add_argument("--golden-set", default="golden_set.json", help="Path to golden set JSON")

    # Ingest mode
    parser.add_argument("--ingest", metavar="DIR", help="Ingest notes from directory")

    # Init sample mode
    parser.add_argument("--init-sample", action="store_true", help="Generate sample notes + golden set")

    # Journal mode
    parser.add_argument("--journal", action="store_true", help="Show learning journal")

    args = parser.parse_args()

    # Route to the right command
    if args.init_sample:
        cmd_init_sample(args)
    elif args.ingest:
        args.directory = args.ingest
        cmd_ingest(args)
    elif args.journal:
        cmd_journal(args)
    elif args.eval:
        cmd_eval(args)
    elif args.compare:
        args.query = args.compare
        cmd_compare(args)
    elif args.query:
        cmd_query(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
