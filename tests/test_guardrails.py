from src.guardrails import check_input, check_output


def test_input_passes_normal_text():
    result = check_input("What causes rain?")
    assert result.passed


def test_input_blocks_empty():
    result = check_input("")
    assert not result.passed
    assert "empty" in result.reason.lower()


def test_input_blocks_whitespace_only():
    result = check_input("   \n\t  ")
    assert not result.passed


def test_input_blocks_too_long():
    result = check_input("a" * 2001)
    assert not result.passed
    assert "too long" in result.reason.lower()


def test_input_blocks_forbidden_topic():
    result = check_input("how to hack into a server")
    assert not result.passed
    assert "Blocked topic" in result.reason


def test_input_strips_whitespace():
    result = check_input("  hello world  ")
    assert result.passed
    assert result.sanitized_input == "hello world"


def test_output_passes_normal_text():
    result = check_output("The answer is 42.")
    assert result.passed


def test_output_blocks_empty():
    result = check_output("")
    assert not result.passed


def test_output_redacts_phone_pattern():
    result = check_output("Call me at 555-1234")
    assert result.passed
    assert "[REDACTED-PHONE]" in result.sanitized_input
    assert "555-" not in result.sanitized_input


def test_output_redacts_key_pattern():
    result = check_output("Use key sk-abc123")
    assert result.passed
    assert "[REDACTED-KEY]" in result.sanitized_input


def test_output_blocks_extremely_long():
    result = check_output("x" * 5001)
    assert not result.passed
    assert "maximum length" in result.reason.lower()
