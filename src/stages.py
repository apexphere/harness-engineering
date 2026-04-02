# LEARN: The Stage Controller — The Heart of the Harness Lab
#
# This is where harness engineering comes together. The stage controller
# composes layers into a pipeline based on the --stage N flag.
#
# The key insight: each layer is a function with the SAME interface:
#   (query: str, response: Response, config: dict) -> Response
#
# The controller chains them: stage 1 runs only the prompt layer.
# Stage 2 runs prompt + tools. Stage 6 runs everything. Each stage
# builds on the previous one, and you can measure the difference.
#
# NO frameworks (LangChain, LlamaIndex) are used here. The whole point
# of this project is to understand what those frameworks do by building
# the pipeline by hand. Once you understand this, you'll know exactly
# what a framework is abstracting away.

from typing import Callable

from src.response import Response

# Type alias for layer functions
LayerFn = Callable[[str, Response, dict], Response]

# Layer registry: maps stage number to which layer functions are active.
# Each layer is registered by name so --verbose can print which layer is running.
_LAYER_REGISTRY: list[tuple[str, LayerFn]] = []


def register_layer(name: str, fn: LayerFn) -> None:
    """Register a layer function. Layers run in registration order."""
    _LAYER_REGISTRY.append((name, fn))


def clear_layers() -> None:
    """Clear all registered layers. Used in tests."""
    _LAYER_REGISTRY.clear()


def get_layer_names() -> list[str]:
    """Return names of all registered layers."""
    return [name for name, _ in _LAYER_REGISTRY]


def run_pipeline(
    query: str,
    stage: int,
    config: dict | None = None,
) -> Response:
    """Run the pipeline up to the given stage.

    Stage N activates layers 1 through N. Each layer receives the Response
    from the previous layer and returns a new Response.

    Args:
        query: The user's question
        stage: Which stage to run (1-6)
        config: Dependencies and settings (chroma_collection, session_id, etc.)

    Returns:
        The final Response after all active layers have processed it
    """
    if stage < 1 or stage > len(_LAYER_REGISTRY):
        raise ValueError(
            f"Stage must be between 1 and {len(_LAYER_REGISTRY)}, got {stage}"
        )

    config = config or {}
    response = Response()
    verbose = config.get("verbose", False)

    for i, (name, layer_fn) in enumerate(_LAYER_REGISTRY[:stage]):
        if verbose:
            print(f"  [{i + 1}/{stage}] {name}...", flush=True)

        response = layer_fn(query, response, config)

        if verbose:
            token_count = response.get_metadata("tokens_out", "?")
            print(f"         → {len(response.text)} chars, confidence={response.confidence:.2f}, "
                  f"sources={len(response.sources)}, tokens_out={token_count}")

    return response


# --- Default Layer Implementations (stubs) ---
# These are pass-through stubs that get replaced as each layer is implemented.
# Having stubs means the stage controller works from day one.


def _stub_prompts(query: str, response: Response, config: dict) -> Response:
    """Stub: just passes query through as response text."""
    return response.with_text(f"[Stage 1 stub] Query: {query}")


def _stub_tools(query: str, response: Response, config: dict) -> Response:
    """Stub: no-op, passes response through."""
    return response


def _stub_parsing(query: str, response: Response, config: dict) -> Response:
    """Stub: no-op, passes response through."""
    return response


def _stub_guardrails(query: str, response: Response, config: dict) -> Response:
    """Stub: no-op, passes response through."""
    return response


def _stub_evals(query: str, response: Response, config: dict) -> Response:
    """Stub: no-op, passes response through."""
    return response


def _stub_orchestration(query: str, response: Response, config: dict) -> Response:
    """Stub: no-op, passes response through."""
    return response


def register_default_layers() -> None:
    """Register the default 6 layers with stub implementations.

    Call this at startup. Individual layers replace their stubs as they're
    implemented. This is the progressive unlocking in action.
    """
    clear_layers()
    register_layer("prompts", _stub_prompts)
    register_layer("tools", _stub_tools)
    register_layer("parsing", _stub_parsing)
    register_layer("guardrails", _stub_guardrails)
    register_layer("evals", _stub_evals)
    register_layer("orchestration", _stub_orchestration)


def register_real_layers() -> None:
    """Register the real layer implementations.

    This replaces stubs with the actual layer functions from each module.
    The import is done here (not at module level) to avoid circular imports
    and to make the dependency explicit.
    """
    from src.prompts import layer_prompts
    from src.tools import layer_tools
    from src.parsing import layer_parsing
    from src.guardrails import layer_guardrails
    from src.evals import layer_evals
    from src.orchestration import layer_orchestration

    clear_layers()
    register_layer("prompts", layer_prompts)
    register_layer("tools", layer_tools)
    register_layer("parsing", layer_parsing)
    register_layer("guardrails", layer_guardrails)
    register_layer("evals", layer_evals)
    register_layer("orchestration", layer_orchestration)
