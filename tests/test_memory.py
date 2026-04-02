import json
import os
import tempfile

import pytest

from src.memory import EvalMemory, _tokenize


def test_tokenize():
    tokens = _tokenize("What is self-attention?")
    assert tokens == ["what", "is", "self", "attention"]


def test_store_and_search(tmp_path):
    path = str(tmp_path / "memory.jsonl")
    mem = EvalMemory(path=path)

    mem.store("What is attention?", stage=6, score=2, max_score=5, failures=["confidence", "sources"])
    mem.store("How does backpropagation work?", stage=6, score=4, max_score=5, failures=["word_count"])
    mem.store("Explain self-attention mechanisms", stage=6, score=3, max_score=5, failures=["sources"])

    results = mem.search("What is self-attention?", n_results=2)
    assert len(results) > 0
    # The attention-related queries should rank higher than backpropagation
    queries = [r["query"] for r in results]
    assert any("attention" in q.lower() for q in queries)


def test_search_empty(tmp_path):
    path = str(tmp_path / "memory.jsonl")
    mem = EvalMemory(path=path)
    results = mem.search("anything")
    assert results == []


def test_format_for_prompt(tmp_path):
    path = str(tmp_path / "memory.jsonl")
    mem = EvalMemory(path=path)

    mem.store("What is attention?", stage=6, score=2, max_score=5, failures=["confidence", "sources"])

    results = mem.search("attention", n_results=1)
    formatted = mem.format_for_prompt(results)

    assert "Past attempts on similar queries:" in formatted
    assert "2/5" in formatted
    assert "confidence" in formatted
    assert "sources" in formatted


def test_format_empty():
    mem = EvalMemory.__new__(EvalMemory)
    assert mem.format_for_prompt([]) == ""


def test_bm25_relevance(tmp_path):
    path = str(tmp_path / "memory.jsonl")
    mem = EvalMemory(path=path)

    mem.store("What is attention in transformers?", stage=6, score=3, max_score=5, failures=["sources"])
    mem.store("How to cook pasta carbonara?", stage=6, score=5, max_score=5, failures=[])
    mem.store("Explain multi-head attention", stage=6, score=2, max_score=5, failures=["confidence"])

    results = mem.search("self-attention mechanism", n_results=3)
    # Attention queries should rank above cooking
    if len(results) >= 2:
        attention_results = [r for r in results if "attention" in r["query"].lower()]
        cooking_results = [r for r in results if "cook" in r["query"].lower()]
        if attention_results and cooking_results:
            assert attention_results[0]["similarity"] > cooking_results[0]["similarity"]


def test_store_appends_jsonl(tmp_path):
    path = str(tmp_path / "memory.jsonl")
    mem = EvalMemory(path=path)

    mem.store("query 1", stage=6, score=3, max_score=5, failures=["a"])
    mem.store("query 2", stage=6, score=4, max_score=5, failures=[])

    with open(path) as f:
        lines = [l for l in f.readlines() if l.strip()]
    assert len(lines) == 2

    entry1 = json.loads(lines[0])
    assert entry1["query"] == "query 1"
    assert entry1["score"] == 3
    assert entry1["failures"] == ["a"]
    assert "timestamp" in entry1

    entry2 = json.loads(lines[1])
    assert entry2["query"] == "query 2"
    assert entry2["score"] == 4


def test_persistence(tmp_path):
    """Memory persists across instances."""
    path = str(tmp_path / "memory.jsonl")

    mem1 = EvalMemory(path=path)
    mem1.store("What is attention?", stage=6, score=3, max_score=5, failures=["sources"])

    # New instance loads from disk
    mem2 = EvalMemory(path=path)
    results = mem2.search("attention", n_results=1)
    assert len(results) == 1
    assert results[0]["query"] == "What is attention?"


def test_format_passed_all_checks(tmp_path):
    path = str(tmp_path / "memory.jsonl")
    mem = EvalMemory(path=path)
    mem.store("Good query", stage=6, score=5, max_score=5, failures=[])

    results = mem.search("Good query", n_results=1)
    formatted = mem.format_for_prompt(results)
    assert "passed all checks" in formatted
