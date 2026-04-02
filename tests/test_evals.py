from src.evals import (
    evaluate_response,
    check_contains_any,
    check_word_count,
    check_has_citation,
    check_acknowledges_uncertainty,
    RESEARCH_EVAL_CASES,
)


def test_check_contains_any_found():
    assert check_contains_any("Solar wind causes auroras", ["solar wind"])


def test_check_contains_any_not_found():
    assert not check_contains_any("Hello world", ["solar wind"])


def test_check_contains_any_case_insensitive():
    assert check_contains_any("SOLAR WIND", ["solar wind"])


def test_word_count_under():
    assert check_word_count("one two three", 10)


def test_word_count_over():
    assert not check_word_count("word " * 201, 200)


def test_has_citation():
    assert check_has_citation("According to NASA, the sky is blue")


def test_no_citation():
    assert not check_has_citation("The sky is blue")


def test_acknowledges_uncertainty():
    assert check_acknowledges_uncertainty("I'm not sure about that")


def test_does_not_acknowledge_uncertainty():
    assert not check_acknowledges_uncertainty("The answer is definitely 42")


def test_evaluate_aurora_response():
    """Test a good response to the aurora question."""
    case = RESEARCH_EVAL_CASES[0]  # aurora_question
    good_response = (
        "Aurora borealis is caused by charged particles from the solar wind "
        "interacting with Earth's magnetosphere and atmosphere. These particles "
        "excite gas molecules, producing colorful light. Source: NOAA"
    )
    result = evaluate_response(case, good_response)
    assert result.passed
    assert all(result.checks.values())


def test_evaluate_bad_aurora_response():
    """A vague response should fail some checks."""
    case = RESEARCH_EVAL_CASES[0]
    bad_response = "Auroras are pretty lights in the sky."
    result = evaluate_response(case, bad_response)
    assert not result.passed
