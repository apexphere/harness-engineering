# LEARN: Layer 3 — Parsing
#
# Models return text. Your application needs structured data. Parsing bridges
# the gap — extracting fields, validating format, and converting free-form
# model output into something your code can reliably use.
#
# Key concepts:
#   - Structured extraction: pull specific fields from model output
#   - Format validation: check that output matches expected structure
#   - Graceful degradation: handle malformed output without crashing
#   - The model will sometimes ignore your format instructions. Your parser
#     must handle this. Never trust model output blindly.

from dataclasses import dataclass
import json
import re

from src.response import Response, Source


@dataclass(frozen=True)
class Summary:
    """Parsed summary with structured fields."""

    topic: str
    key_points: list[str]
    confidence: str  # "high" | "medium" | "low"


@dataclass(frozen=True)
class ParseError:
    """Returned when parsing fails, with the raw text preserved."""

    raw_text: str
    error: str


def parse_summary(text: str) -> Summary | ParseError:
    """Parse the model's summary output into structured data.

    This handles the format defined in SUMMARIZER_SYSTEM prompt:
        TOPIC: ...
        KEY_POINTS:
        - point 1
        - point 2
        CONFIDENCE: high | medium | low

    Try sending malformed text to see how it degrades gracefully.
    """
    # Extract TOPIC
    topic_match = re.search(r"TOPIC:\s*(.+)", text)
    if topic_match is None:
        return ParseError(raw_text=text, error="Missing TOPIC field")
    topic = topic_match.group(1).strip()

    # Extract KEY_POINTS
    points_section = re.search(
        r"KEY_POINTS:\s*\n((?:\s*-\s*.+\n?)+)", text
    )
    if points_section is None:
        return ParseError(raw_text=text, error="Missing KEY_POINTS field")
    key_points = [
        line.strip().lstrip("- ").strip()
        for line in points_section.group(1).strip().split("\n")
        if line.strip().startswith("-")
    ]

    # Extract CONFIDENCE
    conf_match = re.search(r"CONFIDENCE:\s*(high|medium|low)", text, re.IGNORECASE)
    if conf_match is None:
        return ParseError(raw_text=text, error="Missing or invalid CONFIDENCE field")
    confidence = conf_match.group(1).lower()

    return Summary(topic=topic, key_points=key_points, confidence=confidence)


def extract_citations(text: str) -> list[str]:
    """Extract source citations from model output.

    Looks for patterns like "Source: X" or "(X)" at end of sentences.
    This is a simple heuristic — production systems often use the model
    to self-annotate citations via structured output.
    """
    source_pattern = re.findall(r"Source:\s*([^.\n]+)", text)
    return [s.strip() for s in source_pattern]


# --- Pipeline Layer Function ---

def parse_json_response(text: str) -> dict | None:
    """Try to extract JSON from model output.

    The model may wrap JSON in markdown code fences or include preamble text.
    This handles common variations.
    """
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from code fences
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding JSON object in text
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass

    return None


def layer_parsing(query: str, response: Response, config: dict) -> Response:
    """Layer 3: Extract structured data from model output.

    Two parse modes depending on backend:
      - "schema-enforced" (claude-code): --json-schema guarantees structure.
        Parsing is validation — confirm the data is correct and map it to Response.
      - "best-effort" (api): model may or may not follow format instructions.
        Parsing is extraction — hunt for JSON in free-form text.

    Both modes fall back to raw text if parsing fails.
    """
    # If the backend already provided structured output (via --json-schema),
    # use it directly instead of re-parsing the text.
    structured = config.get("structured_output")
    if structured is not None:
        parsed = structured
        parse_mode = "schema-enforced"
    else:
        parsed = parse_json_response(response.text)
        parse_mode = "best-effort"

    if parsed is None:
        # Fallback: treat raw text as the answer, no structured data
        return (
            response
            .with_confidence(0.0)
            .with_metadata("parse_status", "fallback")
            .with_metadata("parse_mode", parse_mode)
        )

    answer = parsed.get("answer", response.text)
    confidence = float(parsed.get("confidence", 0.0))
    gaps = tuple(parsed.get("gaps", []))

    # Extract sources from parsed JSON
    parsed_sources = []
    for s in parsed.get("sources", []):
        if isinstance(s, dict) and "file" in s:
            parsed_sources.append(Source(
                file=s["file"],
                excerpt=s.get("excerpt", ""),
            ))

    # Merge parsed sources with retrieved sources (layer 2)
    existing_sources = response.sources
    if parsed_sources:
        existing_sources = tuple(parsed_sources)

    return (
        response
        .with_text(answer)
        .with_sources(existing_sources)
        .with_confidence(confidence)
        .with_gaps(gaps)
        .with_metadata("parse_status", "success")
        .with_metadata("parse_mode", parse_mode)
    )
