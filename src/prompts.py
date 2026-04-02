# LEARN: Layer 1 — Prompts
#
# The prompt is the most important part of any harness. It defines the model's
# persona, capabilities, constraints, and output format. A good prompt is the
# difference between a toy demo and a reliable product.
#
# Key concepts:
#   - System prompt: persistent instructions that shape every response
#   - Few-shot examples: show the model what good output looks like
#   - Template variables: make prompts reusable across different inputs
#   - Prompt composition: build complex prompts from simple, testable parts

from dataclasses import dataclass

from src.response import Response


@dataclass(frozen=True)
class Prompt:
    """An immutable prompt with a system message and user template."""

    system: str
    user_template: str

    def render_user(self, **kwargs: str) -> str:
        """Fill in template variables. Raises KeyError if a variable is missing."""
        return self.user_template.format(**kwargs)


# --- System Prompts ---
# These define the model's persona and constraints.
# Notice how each is focused and specific — vague prompts get vague results.

RESEARCH_SYSTEM = """You are a research assistant. Your job is to answer questions
accurately and concisely using the tools available to you.

Rules:
- Always cite your sources by name
- If you're unsure, say so — never fabricate information
- Keep answers under 200 words unless asked for detail
- Use the search tool before answering factual questions"""

SUMMARIZER_SYSTEM = """You are a summarizer. Given a block of text, produce a
structured summary.

Output format (strict):
- TOPIC: one-line topic
- KEY_POINTS: 2-4 bullet points
- CONFIDENCE: high | medium | low"""


# --- Few-Shot Examples ---
# These teach the model by example. They're especially useful for enforcing
# output format without complex parsing logic.

SUMMARY_EXAMPLES = [
    {
        "role": "user",
        "content": "Summarize: The James Webb Space Telescope launched in December 2021. "
        "It orbits the Sun at the L2 Lagrange point, about 1.5 million km from Earth. "
        "Its primary mirror is 6.5 meters wide, made of gold-plated beryllium segments.",
    },
    {
        "role": "assistant",
        "content": (
            "TOPIC: James Webb Space Telescope overview\n"
            "KEY_POINTS:\n"
            "- Launched December 2021, orbits at Sun-Earth L2 point\n"
            "- 6.5m primary mirror made of gold-plated beryllium\n"
            "- Located ~1.5 million km from Earth\n"
            "CONFIDENCE: high"
        ),
    },
]


# --- Prompt Composition ---
# Build complex prompts by combining parts. This keeps each piece testable.

def build_research_prompt(question: str, context: str = "") -> Prompt:
    """Compose a research prompt with optional prior context.

    Try modifying the system prompt and see how the model's behavior changes.
    That's the core loop of prompt engineering.
    """
    system = RESEARCH_SYSTEM
    if context:
        system += f"\n\nPrior context the user has shared:\n{context}"

    return Prompt(
        system=system,
        user_template="{question}",
    )


def build_summary_prompt() -> Prompt:
    """Compose a summarization prompt."""
    return Prompt(
        system=SUMMARIZER_SYSTEM,
        user_template="Summarize: {text}",
    )


# --- Pipeline Layer Function ---
# This is the function registered with the stage controller.

KNOWLEDGE_ASSISTANT_SYSTEM = """You are a personal knowledge assistant. You answer questions
using ONLY the context provided from the user's notes. If the context doesn't contain
enough information to answer, say so clearly.

Rules:
- ONLY use information from the provided context
- Cite source files by name (e.g., "According to ml/attention.md...")
- If the context doesn't cover the topic, say "Your notes don't cover this topic"
- Keep answers under 200 words unless the user asks for detail
- Be specific — quote the user's own words when possible

Output your response as JSON:
{
  "answer": "your answer text here",
  "sources": [{"file": "path/to/file.md", "excerpt": "relevant quote"}],
  "confidence": 0.0 to 1.0,
  "gaps": ["topics not covered in the user's notes"]
}"""

# JSON Schema matching the output format requested in KNOWLEDGE_ASSISTANT_SYSTEM.
# When using the claude-code backend with --json-schema, this enforces structure
# at the LLM call level. When using the api backend, layer 3 (parsing) extracts
# the same structure from free-form text.
RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "excerpt": {"type": "string"},
                },
                "required": ["file", "excerpt"],
            },
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "gaps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "sources", "confidence", "gaps"],
}


def layer_prompts(query: str, response: Response, config: dict) -> Response:
    """Layer 1: Build the system prompt and store it in config for later layers.

    This layer doesn't call the model — it prepares the prompt that Layer 6
    (orchestration) will use when making the claude -p call.

    If eval memory is available, retrieves past results for similar queries
    and appends them to the system prompt. The LLM sees what went wrong
    before and can avoid the same failures.
    """
    system_prompt = KNOWLEDGE_ASSISTANT_SYSTEM

    # Inject memory context if available
    memory = config.get("memory")
    memory_matches = 0
    if memory is not None:
        matches = memory.search(query, n_results=3)
        memory_matches = len(matches)
        memory_context = memory.format_for_prompt(matches)
        if memory_context:
            system_prompt = f"{system_prompt}\n\n{memory_context}"

    config["system_prompt"] = system_prompt
    config["json_schema"] = RESPONSE_JSON_SCHEMA
    config["user_query"] = query

    result = response.with_metadata("system_prompt_tokens", str(len(system_prompt.split())))
    result = result.with_metadata("memory_matches", str(memory_matches))
    return result
