# LEARN: Layer 4 — Guardrails
#
# Guardrails are checks that run BEFORE input reaches the model and AFTER the
# model produces output. They enforce safety, quality, and business rules
# without relying on the model to police itself.
#
# Key concepts:
#   - Input validation: reject or sanitize bad input before it reaches the model
#   - Output filtering: catch and fix problematic model output before the user sees it
#   - Defense in depth: don't rely on just the prompt — add programmatic checks
#   - The model is not a security boundary. Guardrails are YOUR code, deterministic
#     and auditable, running outside the model.

from dataclasses import dataclass
import re

from src.response import Response


@dataclass(frozen=True)
class GuardrailResult:
    """Result of a guardrail check. Passed or blocked with reason."""

    passed: bool
    reason: str = ""
    sanitized_input: str = ""


# --- Input Guardrails ---
# These run BEFORE the user's input reaches the model.

# Topics the research assistant should not engage with.
BLOCKED_TOPICS = [
    "how to hack",
    "make a weapon",
    "illegal",
]

MAX_INPUT_LENGTH = 2000


def check_input(user_input: str) -> GuardrailResult:
    """Validate user input before sending to the model.

    Checks:
    1. Input is not empty
    2. Input is not too long (prevents abuse / cost explosion)
    3. Input doesn't contain blocked topics

    In production, you'd also check for prompt injection attempts,
    PII that shouldn't be sent to the API, etc.
    """
    stripped = user_input.strip()

    if not stripped:
        return GuardrailResult(passed=False, reason="Input is empty")

    if len(stripped) > MAX_INPUT_LENGTH:
        return GuardrailResult(
            passed=False,
            reason=f"Input too long ({len(stripped)} chars, max {MAX_INPUT_LENGTH})",
        )

    input_lower = stripped.lower()
    for topic in BLOCKED_TOPICS:
        if topic in input_lower:
            return GuardrailResult(
                passed=False,
                reason=f"Blocked topic detected: '{topic}'",
            )

    return GuardrailResult(passed=True, sanitized_input=stripped)


# --- Output Guardrails ---
# These run AFTER the model responds, before the user sees it.

REDACT_PATTERNS = [
    # Fake patterns for demonstration — in production, use regex for
    # SSNs, credit cards, API keys, etc.
    ("555-", "[REDACTED-PHONE]"),
    ("sk-", "[REDACTED-KEY]"),
]


def check_output(model_output: str) -> GuardrailResult:
    """Validate and sanitize model output before showing to the user.

    Checks:
    1. Output is not empty (model returned something)
    2. Redact any sensitive-looking patterns
    3. Output isn't suspiciously long (possible runaway generation)
    """
    if not model_output.strip():
        return GuardrailResult(passed=False, reason="Model returned empty output")

    sanitized = model_output
    for pattern, replacement in REDACT_PATTERNS:
        sanitized = sanitized.replace(pattern, replacement)

    if len(sanitized) > 5000:
        return GuardrailResult(
            passed=False,
            reason="Output exceeds maximum length — possible runaway generation",
        )

    was_modified = sanitized != model_output
    return GuardrailResult(
        passed=True,
        sanitized_input=sanitized,
        reason="Output was sanitized" if was_modified else "",
    )


# --- Hallucination Filter ---

def extract_noun_phrases(text: str) -> list[str]:
    """Extract key noun phrases using simple tokenization.

    No NLP library needed. Keeps capitalized multi-word sequences and
    quoted strings. This is a rough heuristic, not production NER.
    """
    phrases = []

    # Quoted strings
    for match in re.findall(r'"([^"]+)"', text):
        phrases.append(match)

    # Capitalized multi-word sequences (2+ words starting with uppercase)
    for match in re.findall(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", text):
        phrases.append(match)

    # Single capitalized words that aren't sentence starters
    sentences = text.split(". ")
    for sentence in sentences:
        words = sentence.split()
        for word in words[1:]:  # skip first word (sentence starter)
            clean = word.strip(".,;:!?()")
            if clean and clean[0].isupper() and len(clean) > 2:
                phrases.append(clean)

    return list(set(phrases))


def check_against_chunks(phrases: list[str], chunks: list[str]) -> tuple[list[str], list[str]]:
    """Check which phrases appear in the retrieved chunks.

    Returns (verified, unverified) phrase lists.
    """
    chunks_lower = " ".join(chunks).lower()
    verified = []
    unverified = []

    for phrase in phrases:
        if phrase.lower() in chunks_lower:
            verified.append(phrase)
        else:
            unverified.append(phrase)

    return verified, unverified


def layer_guardrails(query: str, response: Response, config: dict) -> Response:
    """Layer 4: Input validation + hallucination filter.

    Input validation runs the existing check_input function.
    The hallucination filter extracts noun phrases from the model's answer
    and checks them against the retrieved chunks. Unverified claims are
    flagged with [UNVERIFIED].
    """
    # Input validation
    input_check = check_input(query)
    if not input_check.passed:
        return response.with_text(f"Input blocked: {input_check.reason}").with_confidence(0.0)

    # Output validation
    output_check = check_output(response.text)
    if not output_check.passed:
        return response.with_text(f"Output blocked: {output_check.reason}").with_confidence(0.0)

    text = output_check.sanitized_input or response.text

    # Hallucination filter
    chunks = config.get("retrieved_chunks", [])
    if chunks and text:
        phrases = extract_noun_phrases(text)
        verified, unverified = check_against_chunks(phrases, chunks)

        unverified_count = len(unverified)
        total_phrases = len(phrases)

        if total_phrases > 0 and unverified_count / total_phrases > 0.5:
            text += "\n\n⚠ Low source coverage — the model may be guessing."

        for phrase in unverified:
            text = text.replace(phrase, f"{phrase} [UNVERIFIED]", 1)

        return (
            response
            .with_text(text)
            .with_metadata("verified_claims", str(len(verified)))
            .with_metadata("unverified_claims", str(unverified_count))
        )

    return response.with_text(text)
