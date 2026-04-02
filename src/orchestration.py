# LEARN: Layer 6 — Orchestration
#
# Orchestration is the control flow that ties everything together.
# In this harness, it's the layer that makes the LLM call via `claude -p`
# and handles retries.
#
# Key concepts:
#   - Backend dispatch: calling claude -p as a subprocess
#   - Retry logic: what to do when confidence is low
#   - Context injection: combining retrieved notes with the query
#   - State tracking: conversation history via --session/--resume
#
# Earlier layers prepared the inputs (system prompt, retrieved context).
# Later layers validate the output (parsing, guardrails, evals).
# This layer is the non-deterministic core that the rest contains.

from dataclasses import dataclass

from src.response import Response


# --- Conversation State (for multi-turn tracking) ---

@dataclass(frozen=True)
class Turn:
    """One turn in the conversation — immutable record."""

    role: str  # "user", "assistant"
    content: str


@dataclass(frozen=True)
class ConversationState:
    """Immutable conversation state. New turns produce new states."""

    turns: tuple[Turn, ...]
    is_complete: bool = False
    error: str = ""

    def add_turn(self, turn: Turn) -> "ConversationState":
        """Return a new state with the turn appended."""
        return ConversationState(turns=(*self.turns, turn))

    def with_error(self, error: str) -> "ConversationState":
        """Return a new state marked as errored."""
        return ConversationState(
            turns=self.turns, is_complete=True, error=error
        )

    def completed(self) -> "ConversationState":
        """Return a new state marked as complete."""
        return ConversationState(turns=self.turns, is_complete=True)


# --- Pipeline Layer Function ---

def layer_orchestration(query: str, response: Response, config: dict) -> Response:
    """Layer 6: Call claude -p and handle retries.

    This is where the LLM call actually happens. Earlier layers prepared
    the system prompt (layer 1) and retrieved context (layer 2). The
    response will be parsed (layer 3) and checked (layers 4-5) after this.

    This layer is the non-deterministic core. The rest of the pipeline
    is the deterministic containment structure around it.
    """
    from src.backend import get_backend, BackendError

    system_prompt = config.get("system_prompt", "You are a helpful assistant.")
    context = config.get("retrieved_context", "")
    max_retries = config.get("max_retries", 1)

    # Build the user message with context
    if context:
        user_message = f"Context from my notes:\n\n{context}\n\n---\n\nQuestion: {query}"
    else:
        user_message = query

    # Call claude -p via the backend
    try:
        backend = get_backend(config)
        llm_result = backend.call(system_prompt, user_message, config)
    except BackendError as e:
        return response.with_text(f"Backend error: {e}").with_confidence(0.0)

    # Map LLMResult metadata onto the Response
    result = response.with_text(llm_result.text)
    for key, value in llm_result.metadata:
        result = result.with_metadata(key, value)

    # If structured output is available (from --json-schema), store it
    # so layer 3 (parsing) can use it directly instead of extracting from text.
    if llm_result.structured is not None:
        config["structured_output"] = llm_result.structured

    # Retry logic: if confidence is low after parsing, retry with a refined query.
    # We check confidence > 0 to avoid retrying unparsed responses.
    if max_retries > 0 and result.confidence > 0 and result.confidence < 0.5:
        refined_message = (
            f"The previous answer had low confidence. Please try again with more "
            f"specific information from the context.\n\n{user_message}"
        )
        try:
            retry_result = backend.call(system_prompt, refined_message, config)
            if retry_result.text:
                result = result.with_text(retry_result.text).with_metadata("retried", "true")
                for key, value in retry_result.metadata:
                    result = result.with_metadata(key, value)
        except BackendError:
            pass  # Retry failure is not fatal

    return result
