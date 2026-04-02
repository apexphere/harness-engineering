"""Tests for the pipeline layer functions added to each module."""
import os
import tempfile

from src.response import Response, Source
from src.prompts import layer_prompts
from src.tools import layer_tools
from src.parsing import layer_parsing, parse_json_response
from src.guardrails import (
    layer_guardrails,
    extract_noun_phrases,
    check_against_chunks,
)
from src.evals import layer_evals, score_response, load_golden_set
from src.ingestion import ingest_directory
from src.stages import register_real_layers, get_layer_names


# --- Layer 1: Prompts ---

def test_layer_prompts_sets_system_prompt():
    config = {}
    result = layer_prompts("test query", Response(), config)
    assert "system_prompt" in config
    assert "personal knowledge assistant" in config["system_prompt"].lower()
    assert result.get_metadata("system_prompt_tokens") != ""


# --- Layer 2: Tools ---

def test_layer_tools_with_chroma():
    """Integration test: ingest sample notes, then search."""
    with tempfile.TemporaryDirectory() as tmpdir:
        notes_dir = os.path.join(tmpdir, "notes")
        os.makedirs(notes_dir)
        with open(os.path.join(notes_dir, "test.md"), "w") as f:
            f.write("## Attention\nSelf-attention computes relevance scores between tokens.\n")

        db_path = os.path.join(tmpdir, ".chroma")
        ingest_directory(notes_dir, db_path=db_path)

        config = {"db_path": db_path}
        result = layer_tools("attention mechanisms", Response(), config)

        assert len(result.sources) > 0
        assert "retrieved_context" in config
        assert result.get_metadata("chunks_retrieved") != ""


def test_layer_tools_empty_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"db_path": os.path.join(tmpdir, ".chroma")}
        result = layer_tools("anything", Response(), config)
        assert "no relevant notes" in result.text.lower()


# --- Layer 3: Parsing ---

def test_parse_json_response_valid():
    text = '{"answer": "hello", "sources": [], "confidence": 0.9, "gaps": []}'
    parsed = parse_json_response(text)
    assert parsed is not None
    assert parsed["answer"] == "hello"


def test_parse_json_response_code_fence():
    text = '```json\n{"answer": "hello"}\n```'
    parsed = parse_json_response(text)
    assert parsed is not None


def test_parse_json_response_invalid():
    parsed = parse_json_response("just plain text, no JSON here")
    assert parsed is None


def test_layer_parsing_valid_json():
    text = '{"answer": "Attention uses dot products", "sources": [{"file": "ml.md", "excerpt": "dot product"}], "confidence": 0.85, "gaps": ["cross-attention"]}'
    response = Response(text=text)
    result = layer_parsing("test", response, {})
    assert result.text == "Attention uses dot products"
    assert result.confidence == 0.85
    assert "cross-attention" in result.gaps
    assert result.get_metadata("parse_status") == "success"


def test_layer_parsing_fallback():
    response = Response(text="plain text, not JSON")
    result = layer_parsing("test", response, {})
    assert result.get_metadata("parse_status") == "fallback"
    assert result.confidence == 0.0


# --- Layer 4: Guardrails ---

def test_extract_noun_phrases():
    text = 'Self Attention is used in "transformer models" for processing.'
    phrases = extract_noun_phrases(text)
    assert "Self Attention" in phrases
    assert "transformer models" in phrases


def test_check_against_chunks_verified():
    phrases = ["Self Attention", "transformer"]
    chunks = ["Self Attention is a mechanism in transformer architectures"]
    verified, unverified = check_against_chunks(phrases, chunks)
    assert "Self Attention" in verified or "self attention" in [v.lower() for v in verified]


def test_check_against_chunks_unverified():
    phrases = ["Quantum Computing"]
    chunks = ["Self attention is a mechanism"]
    verified, unverified = check_against_chunks(phrases, chunks)
    assert "Quantum Computing" in unverified


def test_layer_guardrails_passes_valid():
    response = Response(text="A valid answer about attention mechanisms.")
    result = layer_guardrails("What is attention?", response, {})
    assert "blocked" not in result.text.lower()


def test_layer_guardrails_blocks_empty_query():
    response = Response(text="some answer")
    result = layer_guardrails("", response, {})
    assert "blocked" in result.text.lower()


def test_layer_guardrails_hallucination_filter():
    response = Response(text="Quantum Computing is used for attention mechanisms.")
    config = {"retrieved_chunks": ["attention mechanisms use self-attention"]}
    result = layer_guardrails("test", response, config)
    assert "[UNVERIFIED]" in result.text


# --- Layer 5: Evals ---

def test_layer_evals_with_good_response():
    response = Response(
        text="Short answer.",
        sources=(Source(file="test.md", excerpt="x"),),
        confidence=0.8,
    )
    result = layer_evals("test", response, {})
    assert result.get_metadata("eval_passed") == "True"


def test_layer_evals_no_sources():
    response = Response(text="Answer without any sources.")
    result = layer_evals("test", response, {})
    assert "has_sources=✗" in result.get_metadata("eval_checks")


# --- Golden Set Scoring ---

def test_score_response_perfect():
    question = {
        "id": "q01",
        "required_concepts": ["attention", "tokens"],
        "required_sources": ["ml.md"],
        "forbidden_content": [],
        "max_words": 200,
    }
    result = score_response(
        question,
        "Attention computes scores between tokens using dot products.",
        ["ml.md"],
    )
    assert result["score"] == 5
    assert result["passed"]


def test_score_response_missing_concepts():
    question = {
        "id": "q02",
        "required_concepts": ["quantum", "entanglement"],
        "required_sources": [],
        "forbidden_content": [],
        "max_words": 200,
    }
    result = score_response(question, "Hello world.", [])
    assert result["score"] < 5


def test_score_response_forbidden_content():
    question = {
        "id": "q03",
        "required_concepts": [],
        "required_sources": [],
        "forbidden_content": ["specific number"],
        "max_words": 200,
    }
    result = score_response(
        question,
        "The answer is a specific number: 42.",
        [],
    )
    assert not result["checks"]["no_forbidden"]


# --- Real Layers Registration ---

def test_register_real_layers():
    register_real_layers()
    names = get_layer_names()
    assert names == ["prompts", "tools", "parsing", "guardrails", "evals", "orchestration"]
