from src.parsing import parse_summary, extract_citations, Summary, ParseError


VALID_SUMMARY = """TOPIC: James Webb Space Telescope
KEY_POINTS:
- Launched in 2021
- Orbits at L2 point
- 6.5m mirror
CONFIDENCE: high"""


def test_parse_valid_summary():
    result = parse_summary(VALID_SUMMARY)
    assert isinstance(result, Summary)
    assert result.topic == "James Webb Space Telescope"
    assert len(result.key_points) == 3
    assert result.confidence == "high"


def test_parse_missing_topic():
    result = parse_summary("KEY_POINTS:\n- a\nCONFIDENCE: low")
    assert isinstance(result, ParseError)
    assert "TOPIC" in result.error


def test_parse_missing_confidence():
    result = parse_summary("TOPIC: test\nKEY_POINTS:\n- a\n")
    assert isinstance(result, ParseError)
    assert "CONFIDENCE" in result.error


def test_parse_invalid_confidence_value():
    text = "TOPIC: test\nKEY_POINTS:\n- a\nCONFIDENCE: very_high"
    result = parse_summary(text)
    assert isinstance(result, ParseError)


def test_extract_citations():
    text = "The sky is blue. Source: NASA. Also, Source: NOAA report"
    citations = extract_citations(text)
    assert len(citations) == 2
    assert "NASA" in citations[0]


def test_extract_no_citations():
    citations = extract_citations("Just a plain sentence.")
    assert citations == []


def test_summary_is_immutable():
    s = Summary(topic="t", key_points=["a"], confidence="high")
    try:
        s.topic = "new"  # type: ignore
        assert False, "Should have raised"
    except AttributeError:
        pass
