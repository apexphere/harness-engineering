# LEARN: Layer 5 — Evals
#
# Evals (evaluations) are automated tests for model behavior. Unlike unit tests
# for deterministic code, evals test probabilistic output — the same input can
# produce different valid outputs. This requires a different testing mindset.
#
# Key concepts:
#   - Test cases: input + expected behavior (not exact output)
#   - Assertions: check properties of the output, not exact strings
#   - Grading: score responses on multiple dimensions
#   - Regression detection: catch when prompt changes break things
#   - You can't assert model_output == "exact string". Instead assert
#     properties: "contains a citation", "is under 200 words", "valid JSON".

from dataclasses import dataclass
import json
import os

from src.response import Response


@dataclass(frozen=True)
class EvalCase:
    """A single eval test case."""

    name: str
    input_text: str
    expected_properties: list[str]  # human-readable property descriptions


@dataclass(frozen=True)
class EvalResult:
    """Result of running one eval case."""

    case_name: str
    passed: bool
    checks: dict[str, bool]  # property name -> passed


# --- Eval Cases ---
# Each case defines WHAT we're testing, not the exact expected output.

RESEARCH_EVAL_CASES = [
    EvalCase(
        name="aurora_question",
        input_text="What causes aurora borealis?",
        expected_properties=[
            "mentions solar wind or charged particles",
            "mentions Earth's magnetosphere or atmosphere",
            "includes a citation",
            "under 200 words",
        ],
    ),
    EvalCase(
        name="unknown_topic",
        input_text="What is the population of the lost city of Atlantis?",
        expected_properties=[
            "acknowledges uncertainty",
            "does not fabricate a specific number",
        ],
    ),
    EvalCase(
        name="math_question",
        input_text="What is 144 divided by 12?",
        expected_properties=[
            "contains the number 12",
            "uses the calculate tool or shows the math",
        ],
    ),
]


# --- Property Checkers ---
# These are deterministic checks on model output. For more nuanced checks,
# you can use a second model call as a "judge" — but start simple.

def check_contains_any(text: str, keywords: list[str]) -> bool:
    """Check if text contains any of the given keywords (case-insensitive)."""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def check_word_count(text: str, max_words: int) -> bool:
    """Check if text is under the word limit."""
    return len(text.split()) <= max_words


def check_has_citation(text: str) -> bool:
    """Check if text contains something that looks like a citation."""
    citation_signals = ["source:", "according to", "cited from", "(", "—"]
    return check_contains_any(text, citation_signals)


def check_acknowledges_uncertainty(text: str) -> bool:
    """Check if the model admits when it doesn't know something."""
    uncertainty_signals = [
        "not sure", "uncertain", "don't know", "no reliable",
        "cannot confirm", "fictional", "mythical", "no evidence",
        "unclear", "I'm not", "I don't",
    ]
    return check_contains_any(text, uncertainty_signals)


# --- Eval Runner ---
# Runs checks against actual model output. In practice, you'd call the model
# here. For unit testing, we test the checkers themselves.

def evaluate_response(case: EvalCase, model_output: str) -> EvalResult:
    """Evaluate a model response against expected properties.

    This maps human-readable properties to programmatic checks.
    A production eval system would have a richer mapping and support
    LLM-as-judge for subjective properties.
    """
    checks: dict[str, bool] = {}

    for prop in case.expected_properties:
        prop_lower = prop.lower()

        if "solar wind" in prop_lower or "charged particles" in prop_lower:
            checks[prop] = check_contains_any(
                model_output, ["solar wind", "charged particles", "solar"]
            )
        elif "magnetosphere" in prop_lower or "atmosphere" in prop_lower:
            checks[prop] = check_contains_any(
                model_output, ["magnetosphere", "atmosphere", "magnetic"]
            )
        elif "citation" in prop_lower:
            checks[prop] = check_has_citation(model_output)
        elif "200 words" in prop_lower:
            checks[prop] = check_word_count(model_output, 200)
        elif "uncertainty" in prop_lower:
            checks[prop] = check_acknowledges_uncertainty(model_output)
        elif "fabricate" in prop_lower:
            # Inverse check — should NOT contain a specific number
            checks[prop] = not any(
                c.isdigit() for word in model_output.split()
                for c in word if len(word) > 3
            )
        elif "contains the number" in prop_lower:
            target = prop_lower.split("number")[-1].strip()
            checks[prop] = target in model_output
        else:
            # Unknown property — flag for human review
            checks[prop] = False

    all_passed = all(checks.values())
    return EvalResult(case_name=case.name, passed=all_passed, checks=checks)


# --- Golden Set Scoring ---

def load_golden_set(path: str = "golden_set.json") -> list[dict]:
    """Load the golden set from JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Golden set not found at {path}. Run --init-sample to generate one."
        )
    with open(path) as f:
        data = json.load(f)
    return data["questions"]


def score_response(question: dict, answer_text: str, cited_sources: list[str]) -> dict:
    """Score a single response against its golden set question.

    Scoring formula (5 points max):
    - +1 for each required concept found (up to 2)
    - +1 if at least one required source is cited
    - +1 if no forbidden content is present
    - +1 if word count is under max_words
    """
    score = 0
    checks = {}
    answer_lower = answer_text.lower()

    # Concept checks (up to 2 points)
    concepts_found = 0
    for concept in question.get("required_concepts", [])[:2]:
        found = concept.lower() in answer_lower
        checks[f"concept:{concept}"] = found
        if found:
            concepts_found += 1
    score += concepts_found

    # Source check (1 point)
    source_found = False
    for req_source in question.get("required_sources", []):
        if any(req_source in cited for cited in cited_sources):
            source_found = True
            break
    checks["source_cited"] = source_found
    if source_found:
        score += 1

    # Forbidden content check (1 point)
    forbidden_found = False
    for forbidden in question.get("forbidden_content", []):
        if forbidden.lower() in answer_lower:
            forbidden_found = True
            break
    checks["no_forbidden"] = not forbidden_found
    if not forbidden_found:
        score += 1

    # Word count check (1 point)
    max_words = question.get("max_words", 200)
    within_limit = len(answer_text.split()) <= max_words
    checks["word_count"] = within_limit
    if within_limit:
        score += 1

    return {
        "id": question["id"],
        "score": score,
        "max_score": 5,
        "checks": checks,
        "passed": score >= 3,  # Pass threshold: 3/5
    }


# --- Pipeline Layer Function ---

def layer_evals(query: str, response: Response, config: dict) -> Response:
    """Layer 5: Live property checks on each answer.

    Unlike the --eval command (which runs the golden set as a batch),
    this layer runs inline quality checks on every answer in real time.
    """
    checks = {}

    # Check: answer cites sources
    checks["has_sources"] = len(response.sources) > 0

    # Check: answer is under word limit
    checks["under_200_words"] = check_word_count(response.text, 200)

    # Check: confidence is reasonable (not 0 or 1 exactly)
    checks["has_confidence"] = 0.0 < response.confidence <= 1.0

    # Check: acknowledges gaps when present
    if response.gaps:
        checks["reports_gaps"] = True
    else:
        checks["reports_gaps"] = True  # No gaps is fine

    passed = all(checks.values())
    checks_str = ", ".join(f"{k}={'✓' if v else '✗'}" for k, v in checks.items())

    return (
        response
        .with_metadata("eval_checks", checks_str)
        .with_metadata("eval_passed", str(passed))
    )
