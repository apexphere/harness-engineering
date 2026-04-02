# LEARN: The Backend — Calling claude -p
#
# This module handles the LLM call. The rest of the pipeline (prompts,
# retrieval, parsing, guardrails, evals) doesn't care how the call is
# made. It only sees the LLMResult.
#
# The backend calls `claude -p` (Claude Code headless mode) as a subprocess.
# This uses your subscription — no API key needed. Claude Code handles its
# own retries and context management internally. Your harness controls what
# goes in (prompt, schema, context) and validates what comes out.
#
# This is the non-deterministic component inside your deterministic
# containment structure. Everything else in the pipeline bounds it.

import json
import shutil
import subprocess
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMResult:
    """The output of an LLM call.

    text: the raw text response
    structured: parsed JSON if --json-schema was used (None otherwise)
    metadata: diagnostics (session_id, duration, etc.)
    """

    text: str
    structured: dict | None = None
    metadata: tuple[tuple[str, str], ...] = ()


class BackendError(Exception):
    """Raised when the LLM call fails."""

    def __init__(self, message: str, exit_code: int = -1):
        super().__init__(message)
        self.exit_code = exit_code


class ClaudeCodeBackend:
    """Calls Claude via `claude -p` subprocess.

    Uses your Claude subscription — no API key needed. Claude Code handles
    its own retries, tool execution, and context management internally.
    Your harness controls what goes in (prompt, schema) and validates what
    comes out (parsing, guardrails, evals).
    """

    def __init__(self) -> None:
        claude_path = shutil.which("claude")
        if claude_path is None:
            raise BackendError(
                "Claude Code CLI not found. Install it: https://docs.anthropic.com/en/docs/claude-code"
            )
        self._claude_path = claude_path

    def call(self, system: str, user_message: str, config: dict) -> LLMResult:
        """Run a single claude -p call and return the result."""
        cmd = self._build_command(system, user_message, config)
        timeout = config.get("timeout", 120)

        start = time.monotonic()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise BackendError(f"Claude Code timed out after {timeout}s", exit_code=-1)

        duration_ms = int((time.monotonic() - start) * 1000)

        if result.returncode != 0:
            stderr_preview = result.stderr[:300] if result.stderr else "no stderr"
            raise BackendError(
                f"Claude Code exited with code {result.returncode}: {stderr_preview}",
                exit_code=result.returncode,
            )

        return self._parse_output(result.stdout, duration_ms)

    def _build_command(
        self, system: str, user_message: str, config: dict
    ) -> list[str]:
        """Build the claude CLI command."""
        cmd = [
            self._claude_path,
            "--bare",
            "-p",
            user_message,
            "--output-format",
            "json",
            "--append-system-prompt",
            system,
        ]

        json_schema = config.get("json_schema")
        if json_schema is not None:
            cmd.extend(["--json-schema", json.dumps(json_schema)])

        session_id = config.get("session_id")
        if session_id is not None:
            cmd.extend(["--resume", session_id])

        allowed_tools = config.get("allowed_tools")
        if allowed_tools is not None:
            cmd.extend(["--allowedTools", allowed_tools])

        return cmd

    def _parse_output(self, stdout: str, duration_ms: int) -> LLMResult:
        """Parse the JSON output from claude -p."""
        try:
            output = json.loads(stdout)
        except json.JSONDecodeError as e:
            raise BackendError(f"Failed to parse Claude Code output as JSON: {e}")

        text = output.get("result", "")
        session_id = output.get("session_id", "")
        structured = output.get("structured_output")

        metadata = (
            ("backend", "claude-code"),
            ("session_id", session_id),
            ("duration_ms", str(duration_ms)),
        )

        return LLMResult(text=text, structured=structured, metadata=metadata)


def get_backend(config: dict) -> ClaudeCodeBackend:
    """Create the backend."""
    return ClaudeCodeBackend()
