# LEARN: Eval Memory — Learning from Past Failures
#
# Agents without memory repeat the same mistakes. This module stores eval
# results (query, score, failures) and retrieves similar past results using
# BM25 text matching. No API calls, works offline.
#
# How it fits in the pipeline:
#   - Layer 1 (prompts) READS memory: retrieves past results for similar
#     queries and appends them to the system prompt. The LLM sees what
#     went wrong before and can avoid the same failures.
#   - Layer 5 (evals) WRITES memory: after scoring, stores the query
#     and its results so future runs can learn from them.
#
# Memory is deterministic — BM25 is a scored keyword match, not a model call.
# It's part of the containment structure, not the non-deterministic core.

import json
import os
import re
from datetime import datetime, timezone


MEMORY_FILE = "journal/eval_memory.jsonl"


def _tokenize(text: str) -> list[str]:
    """Simple tokenization: lowercase, split on non-alphanumeric."""
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


class EvalMemory:
    """BM25-based memory over past eval results.

    Stores query/score/failure tuples in a JSONL file. Retrieves similar
    past queries using BM25 text matching. No API calls needed.
    """

    def __init__(self, path: str = MEMORY_FILE) -> None:
        self._path = path
        self._entries: list[dict] = []
        self._corpus: list[list[str]] = []
        self._bm25 = None
        self._load()

    def _load(self) -> None:
        """Load existing entries from disk."""
        if not os.path.exists(self._path):
            return
        with open(self._path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    self._entries.append(entry)
                    self._corpus.append(_tokenize(entry["query"]))
                except (json.JSONDecodeError, KeyError):
                    continue
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild BM25 index from corpus."""
        if not self._corpus:
            self._bm25 = None
            return
        from rank_bm25 import BM25Okapi
        self._bm25 = BM25Okapi(self._corpus)

    def store(
        self,
        query: str,
        stage: int,
        score: int,
        max_score: int,
        failures: list[str],
    ) -> None:
        """Store an eval result in memory."""
        entry = {
            "query": query,
            "stage": stage,
            "score": score,
            "max_score": max_score,
            "failures": failures,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        with open(self._path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        self._entries.append(entry)
        self._corpus.append(_tokenize(query))
        self._rebuild_index()

    def search(self, query: str, n_results: int = 3) -> list[dict]:
        """Find similar past queries using BM25.

        Falls back to keyword overlap when BM25 scores are all zero
        (happens with single-document corpus or very short queries).
        """
        if not self._entries:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        # Try BM25 first
        if self._bm25 is not None:
            scores = self._bm25.get_scores(tokens)
            max_score = max(scores) if len(scores) > 0 and max(scores) > 0 else 0.0
        else:
            scores = [0.0] * len(self._entries)
            max_score = 0.0

        # Fallback: simple token overlap when BM25 gives all zeros
        if max_score == 0.0:
            query_set = set(tokens)
            scores = [
                len(query_set & set(doc)) / max(len(query_set), 1)
                for doc in self._corpus
            ]
            max_score = max(scores) if scores and max(scores) > 0 else 1.0

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]

        results = []
        for i in top_indices:
            if scores[i] <= 0:
                continue
            entry = self._entries[i]
            results.append({
                **entry,
                "similarity": round(scores[i] / max_score, 2),
            })

        return results

    def format_for_prompt(self, matches: list[dict]) -> str:
        """Render memory matches as text for system prompt injection."""
        if not matches:
            return ""

        lines = ["Past attempts on similar queries:"]
        for m in matches:
            score_str = f"{m['score']}/{m['max_score']}"
            if m["failures"]:
                fail_str = f", failed: {', '.join(m['failures'])}"
            else:
                fail_str = ", passed all checks"
            lines.append(f"- '{m['query']}' scored {score_str}{fail_str}")

        return "\n".join(lines)
