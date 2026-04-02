from src.response import Response, Source


def test_response_default_values():
    r = Response()
    assert r.text == ""
    assert r.sources == ()
    assert r.confidence == 0.0
    assert r.gaps == ()
    assert r.metadata == ()


def test_response_with_text():
    r = Response().with_text("hello")
    assert r.text == "hello"


def test_response_with_sources():
    s = Source(file="notes/ml.md", excerpt="attention is all you need")
    r = Response().with_sources((s,))
    assert len(r.sources) == 1
    assert r.sources[0].file == "notes/ml.md"


def test_response_with_confidence():
    r = Response().with_confidence(0.85)
    assert r.confidence == 0.85


def test_response_with_gaps():
    r = Response().with_gaps(("cross-attention", "multi-head"))
    assert len(r.gaps) == 2


def test_response_with_metadata():
    r = Response().with_metadata("tokens", "342").with_metadata("cost", "0.001")
    assert r.get_metadata("tokens") == "342"
    assert r.get_metadata("cost") == "0.001"
    assert r.get_metadata("missing", "default") == "default"


def test_response_is_immutable():
    r = Response(text="original")
    r2 = r.with_text("modified")
    assert r.text == "original"
    assert r2.text == "modified"


def test_response_frozen():
    r = Response(text="hello")
    try:
        r.text = "nope"  # type: ignore
        assert False, "Should have raised"
    except AttributeError:
        pass


def test_source_is_frozen():
    s = Source(file="a.md", excerpt="text")
    try:
        s.file = "b.md"  # type: ignore
        assert False, "Should have raised"
    except AttributeError:
        pass


def test_response_chaining():
    """Layers chain with_* calls to build up a Response."""
    r = (
        Response()
        .with_text("Attention mechanisms use self-attention")
        .with_sources((Source(file="ml.md", excerpt="self-attention"),))
        .with_confidence(0.9)
        .with_gaps(("cross-attention",))
        .with_metadata("tokens_in", "100")
        .with_metadata("tokens_out", "50")
    )
    assert r.text.startswith("Attention")
    assert len(r.sources) == 1
    assert r.confidence == 0.9
    assert r.gaps == ("cross-attention",)
    assert len(r.metadata) == 2
