# LEARN: Layer 2 — Tools
#
# Tools give the model capabilities beyond text generation. Instead of hoping
# the model "knows" something, you let it call functions to look things up,
# calculate, or interact with external systems.
#
# Key concepts:
#   - Tool schema: a JSON description of what the tool does and its parameters
#   - Tool execution: your code runs when the model "calls" a tool
#   - Tool results: fed back to the model so it can incorporate the answer
#   - The model doesn't run code — it outputs a structured request, and YOUR
#     harness executes it. You control what actually happens.

from dataclasses import dataclass
from typing import Any

from src.response import Response, Source
from src.ingestion import search_notes


@dataclass(frozen=True)
class ToolResult:
    """Immutable result from a tool execution."""

    tool_name: str
    output: str
    is_error: bool = False


# --- Tool Definitions ---
# These are the JSON schemas sent to the API. The model reads these to decide
# when and how to use each tool.

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "search",
        "description": (
            "Search a knowledge base for information. Use this for any factual "
            "question. Returns relevant passages with source names."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression. Use for any arithmetic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A mathematical expression, e.g. '2 + 2' or 'sqrt(144)'",
                },
            },
            "required": ["expression"],
        },
    },
]


# --- Tool Implementations ---
# These are the actual functions that run when the model requests a tool.
# The model never sees this code — it only sees the schema above and the
# result you send back.

# A tiny in-memory knowledge base for demonstration.
_KNOWLEDGE_BASE: dict[str, str] = {
    "aurora": (
        "Aurora borealis is caused by charged particles from the Sun (solar wind) "
        "interacting with Earth's magnetosphere. These particles excite atmospheric "
        "gases, producing light. Source: NOAA Space Weather"
    ),
    "photosynthesis": (
        "Photosynthesis converts CO2 and water into glucose and oxygen using sunlight. "
        "The light reactions occur in thylakoids; the Calvin cycle in the stroma. "
        "Source: Campbell Biology"
    ),
    "gravity": (
        "Gravity is the curvature of spacetime caused by mass and energy, as described "
        "by Einstein's general relativity. Newtonian gravity approximates it as a force "
        "proportional to mass and inversely proportional to distance squared. "
        "Source: Misner, Thorne & Wheeler"
    ),
}


def execute_search(query: str) -> ToolResult:
    """Search the knowledge base. Returns the best matching entry."""
    query_lower = query.lower()
    for keyword, passage in _KNOWLEDGE_BASE.items():
        if keyword in query_lower:
            return ToolResult(tool_name="search", output=passage)
    return ToolResult(
        tool_name="search",
        output=f"No results found for: {query}",
    )


def execute_calculate(expression: str) -> ToolResult:
    """Safely evaluate a math expression. Only allows basic arithmetic."""
    allowed_chars = set("0123456789+-*/.() ")
    if not all(c in allowed_chars for c in expression):
        return ToolResult(
            tool_name="calculate",
            output=f"Invalid expression: only basic arithmetic allowed",
            is_error=True,
        )
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return ToolResult(tool_name="calculate", output=str(result))
    except Exception as e:
        return ToolResult(
            tool_name="calculate",
            output=f"Calculation error: {e}",
            is_error=True,
        )


# --- Tool Router ---
# Maps tool names to their implementations. The orchestration layer uses this.

TOOL_ROUTER: dict[str, Any] = {
    "search": lambda inputs: execute_search(inputs["query"]),
    "calculate": lambda inputs: execute_calculate(inputs["expression"]),
}


def execute_tool(name: str, inputs: dict[str, Any]) -> ToolResult:
    """Route a tool call to its implementation."""
    handler = TOOL_ROUTER.get(name)
    if handler is None:
        return ToolResult(
            tool_name=name,
            output=f"Unknown tool: {name}",
            is_error=True,
        )
    return handler(inputs)


# --- Pipeline Layer Function ---

def layer_tools(query: str, response: Response, config: dict) -> Response:
    """Layer 2: Search the user's notes via Chroma and inject context.

    This layer retrieves relevant chunks from the vector database and
    builds a context string that the model will use to answer the question.
    The actual API call happens in layer 6 (orchestration) or is simulated
    for stages 2-5 by directly calling the model here.
    """
    db_path = config.get("db_path", ".chroma")
    n_results = config.get("n_results", 5)

    results = search_notes(query, db_path=db_path, n_results=n_results)

    if not results:
        return response.with_text(
            "No relevant notes found. Run --ingest to add your notes."
        ).with_confidence(0.0)

    # Build context from search results
    context_parts = []
    sources = []
    for r in results:
        source_label = r["source_file"]
        if r.get("heading"):
            source_label += f" > {r['heading']}"
        context_parts.append(f"[{source_label}]\n{r['text']}")
        sources.append(Source(file=r["source_file"], excerpt=r["text"][:200]))

    context = "\n\n---\n\n".join(context_parts)
    config["retrieved_context"] = context
    config["retrieved_chunks"] = [r["text"] for r in results]

    return (
        response
        .with_sources(tuple(sources))
        .with_metadata("chunks_retrieved", str(len(results)))
    )
