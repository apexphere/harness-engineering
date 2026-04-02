import json
import subprocess
from unittest.mock import patch

import pytest

from src.backend import (
    LLMResult,
    BackendError,
    ClaudeCodeBackend,
    get_backend,
)


# --- LLMResult ---


def test_llm_result_is_frozen():
    result = LLMResult(text="hello")
    with pytest.raises(AttributeError):
        result.text = "changed"


def test_llm_result_defaults():
    result = LLMResult(text="hello")
    assert result.structured is None
    assert result.metadata == ()


# --- ClaudeCodeBackend ---


@patch("shutil.which", return_value="/usr/local/bin/claude")
def test_claude_code_backend_init(mock_which):
    backend = ClaudeCodeBackend()
    assert backend._claude_path == "/usr/local/bin/claude"


@patch("shutil.which", return_value=None)
def test_claude_code_backend_init_missing_cli(mock_which):
    with pytest.raises(BackendError, match="Claude Code CLI not found"):
        ClaudeCodeBackend()


@patch("shutil.which", return_value="/usr/local/bin/claude")
@patch("subprocess.run")
def test_claude_code_backend_builds_correct_command(mock_run, mock_which):
    mock_run.return_value = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=json.dumps({"result": "test answer", "session_id": "sess-123"}),
        stderr="",
    )

    backend = ClaudeCodeBackend()
    config = {
        "json_schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
    }
    backend.call("You are helpful.", "What is attention?", config)

    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "/usr/local/bin/claude"
    assert "--bare" in cmd
    assert "-p" in cmd
    assert "What is attention?" in cmd
    assert "--output-format" in cmd
    assert "json" in cmd
    assert "--append-system-prompt" in cmd
    assert "You are helpful." in cmd
    assert "--json-schema" in cmd


@patch("shutil.which", return_value="/usr/local/bin/claude")
@patch("subprocess.run")
def test_claude_code_backend_parses_output(mock_run, mock_which):
    mock_run.return_value = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=json.dumps({
            "result": "Attention is a mechanism...",
            "session_id": "sess-456",
            "structured_output": {
                "answer": "Attention is a mechanism...",
                "sources": [{"file": "ml/attention.md", "excerpt": "..."}],
                "confidence": 0.9,
                "gaps": [],
            },
        }),
        stderr="",
    )

    backend = ClaudeCodeBackend()
    result = backend.call("system", "query", {})

    assert result.text == "Attention is a mechanism..."
    assert result.structured is not None
    assert result.structured["confidence"] == 0.9
    assert dict(result.metadata)["backend"] == "claude-code"
    assert dict(result.metadata)["session_id"] == "sess-456"


@patch("shutil.which", return_value="/usr/local/bin/claude")
@patch("subprocess.run")
def test_claude_code_backend_nonzero_exit(mock_run, mock_which):
    mock_run.return_value = subprocess.CompletedProcess(
        args=[], returncode=1, stdout="", stderr="Error: authentication failed",
    )

    backend = ClaudeCodeBackend()
    with pytest.raises(BackendError, match="exited with code 1"):
        backend.call("system", "query", {})


@patch("shutil.which", return_value="/usr/local/bin/claude")
@patch("subprocess.run")
def test_claude_code_backend_timeout(mock_run, mock_which):
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=120)

    backend = ClaudeCodeBackend()
    with pytest.raises(BackendError, match="timed out"):
        backend.call("system", "query", {})


@patch("shutil.which", return_value="/usr/local/bin/claude")
@patch("subprocess.run")
def test_claude_code_backend_invalid_json(mock_run, mock_which):
    mock_run.return_value = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="not json at all", stderr="",
    )

    backend = ClaudeCodeBackend()
    with pytest.raises(BackendError, match="Failed to parse"):
        backend.call("system", "query", {})


@patch("shutil.which", return_value="/usr/local/bin/claude")
@patch("subprocess.run")
def test_claude_code_backend_resume_session(mock_run, mock_which):
    mock_run.return_value = subprocess.CompletedProcess(
        args=[], returncode=0,
        stdout=json.dumps({"result": "continued", "session_id": "sess-789"}),
        stderr="",
    )

    backend = ClaudeCodeBackend()
    backend.call("system", "continue this", {"session_id": "sess-789"})

    cmd = mock_run.call_args[0][0]
    assert "--resume" in cmd
    assert "sess-789" in cmd


@patch("shutil.which", return_value="/usr/local/bin/claude")
@patch("subprocess.run")
def test_claude_code_backend_no_schema(mock_run, mock_which):
    mock_run.return_value = subprocess.CompletedProcess(
        args=[], returncode=0,
        stdout=json.dumps({"result": "plain answer", "session_id": "s1"}),
        stderr="",
    )

    backend = ClaudeCodeBackend()
    backend.call("system", "query", {})

    cmd = mock_run.call_args[0][0]
    assert "--json-schema" not in cmd


# --- get_backend ---


@patch("shutil.which", return_value="/usr/local/bin/claude")
def test_get_backend_returns_claude_code(mock_which):
    backend = get_backend({})
    assert isinstance(backend, ClaudeCodeBackend)
