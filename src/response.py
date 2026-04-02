# LEARN: The Pipeline Interface
#
# This is the connector shape that every layer must conform to. Think of it
# like a USB-C port: every layer plugs in the same way, regardless of what
# it does internally.
#
# The Response dataclass accumulates information as it flows through layers:
#   Stage 1 (prompt): fills text
#   Stage 2 (tools): fills sources
#   Stage 3 (parsing): fills confidence, gaps
#   Stage 4 (guardrails): may modify text (redact unverified claims)
#   Stage 5 (evals): fills metadata with check results
#   Stage 6 (orchestration): may retry and produce a better Response
#
# Every layer function has the same signature:
#   (query: str, response: Response, config: dict) -> Response
#
# This consistency is what makes the stage controller possible.

from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class Source:
    """A cited source from the user's notes."""

    file: str
    excerpt: str


@dataclass(frozen=True)
class Response:
    """Immutable pipeline response that accumulates data across layers.

    metadata uses tuple-of-pairs instead of dict because frozen dataclasses
    require all fields to be hashable/immutable. Use dict(response.metadata)
    to convert when reading.
    """

    text: str = ""
    sources: tuple[Source, ...] = ()
    confidence: float = 0.0
    gaps: tuple[str, ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()

    def with_text(self, text: str) -> "Response":
        """Return a new Response with updated text."""
        return replace(self, text=text)

    def with_sources(self, sources: tuple[Source, ...]) -> "Response":
        """Return a new Response with updated sources."""
        return replace(self, sources=sources)

    def with_confidence(self, confidence: float) -> "Response":
        """Return a new Response with updated confidence."""
        return replace(self, confidence=confidence)

    def with_gaps(self, gaps: tuple[str, ...]) -> "Response":
        """Return a new Response with updated gaps."""
        return replace(self, gaps=gaps)

    def with_metadata(self, key: str, value: str) -> "Response":
        """Return a new Response with an additional metadata entry."""
        return replace(self, metadata=(*self.metadata, (key, value)))

    def get_metadata(self, key: str, default: str = "") -> str:
        """Get a metadata value by key."""
        for k, v in self.metadata:
            if k == key:
                return v
        return default
