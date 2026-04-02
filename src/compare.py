# LEARN: Compare Mode — See What Each Layer Buys You
#
# This is the flagship learning feature. Run the same query through all 6
# stages and see the results side by side. You'll see how each layer
# changes the answer, adds sources, filters hallucinations, and improves
# confidence.

import asyncio
import time

from src.response import Response
from src.stages import run_pipeline, get_layer_names


def run_compare(query: str, config: dict) -> list[dict]:
    """Run the query through all stages and collect results.

    Returns list of {"stage": int, "name": str, "response": Response,
    "latency_ms": float}
    """
    results = []
    num_stages = len(get_layer_names())

    for stage in range(1, num_stages + 1):
        start = time.time()
        try:
            response = run_pipeline(query, stage=stage, config=dict(config))
        except Exception as e:
            response = Response(text=f"[ERROR] {e}")
        elapsed_ms = (time.time() - start) * 1000

        results.append({
            "stage": stage,
            "name": get_layer_names()[stage - 1] if stage <= num_stages else "?",
            "response": response,
            "latency_ms": elapsed_ms,
        })

    return results


def format_comparison_table(results: list[dict]) -> str:
    """Format comparison results as a terminal table."""
    lines = []
    lines.append("── LAYER COMPARISON ──────────────────────────────────")
    lines.append(f"{'STAGE':<12} {'ANSWER (truncated)':<45} {'TOKENS':<8} {'LATENCY':<10} {'SCORE'}")
    lines.append("─" * 90)

    for r in results:
        resp = r["response"]
        # Truncate answer
        answer = resp.text.replace("\n", " ")[:42]
        if len(resp.text) > 42:
            answer += "..."

        tokens = resp.get_metadata("tokens_out", "—")
        latency = f"{r['latency_ms']:.0f}ms"

        conf = resp.confidence
        if conf >= 0.8:
            score = f"{conf:.1f} ★"
        elif conf >= 0.5:
            score = f"{conf:.1f}"
        else:
            score = f"{conf:.1f} ⚠"

        stage_label = f"{r['stage']} {r['name']}"
        lines.append(f"{stage_label:<12} {answer:<45} {tokens:<8} {latency:<10} {score}")

    lines.append("─" * 90)

    # Insight section
    if len(results) >= 2:
        first = results[0]["response"]
        last = results[-1]["response"]
        lines.append("")
        lines.append("── INSIGHT ──")
        if first.confidence != last.confidence:
            lines.append(f"  Confidence: {first.confidence:.1f} → {last.confidence:.1f}")
        lines.append(f"  Sources: {len(first.sources)} → {len(last.sources)}")

        total_tokens = sum(
            int(r["response"].get_metadata("tokens_out", "0"))
            for r in results
        )
        if total_tokens > 0:
            lines.append(f"  Total tokens: {total_tokens}")

    return "\n".join(lines)
