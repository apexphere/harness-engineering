import os
import tempfile

from src.ingestion import (
    chunk_markdown,
    chunk_text,
    ingest_directory,
    search_notes,
    _estimate_tokens,
)


def test_estimate_tokens():
    assert _estimate_tokens("hello world") > 0
    # Roughly 1.3 tokens per word
    assert _estimate_tokens("one two three four five") == int(5 * 1.3)


def test_chunk_text_short():
    """Short text fits in one chunk."""
    chunks = chunk_text("hello world", max_tokens=100)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "hello world"


def test_chunk_text_long():
    """Long text splits into multiple chunks."""
    text = " ".join(["word"] * 1000)
    chunks = chunk_text(text, max_tokens=100, overlap_tokens=20)
    assert len(chunks) > 1
    # Each chunk should be within bounds
    for c in chunks:
        assert len(c["text"].split()) <= int(100 / 1.3) + 1


def test_chunk_text_overlap():
    """Chunks should overlap."""
    text = " ".join([f"word{i}" for i in range(200)])
    chunks = chunk_text(text, max_tokens=100, overlap_tokens=20)
    if len(chunks) >= 2:
        # Last words of chunk 0 should appear in chunk 1
        words_0 = set(chunks[0]["text"].split()[-10:])
        words_1 = set(chunks[1]["text"].split()[:10])
        assert len(words_0 & words_1) > 0


def test_chunk_markdown_by_headings():
    """Markdown splits on ## headings."""
    md = """## Section One
Content for section one.

## Section Two
Content for section two.

## Section Three
Content for section three.
"""
    chunks = chunk_markdown(md)
    assert len(chunks) == 3
    assert chunks[0]["heading"] == "Section One"
    assert chunks[1]["heading"] == "Section Two"
    assert chunks[2]["heading"] == "Section Three"


def test_chunk_markdown_no_headings():
    """Markdown without headings treated as one chunk."""
    md = "Just plain text without any headings."
    chunks = chunk_markdown(md)
    assert len(chunks) == 1
    assert chunks[0]["heading"] is None


def test_chunk_markdown_preserves_char_positions():
    md = """## First
Short content.

## Second
More content here.
"""
    chunks = chunk_markdown(md)
    for c in chunks:
        assert c["char_start"] >= 0
        assert c["char_end"] > c["char_start"]


def test_ingest_and_search():
    """Full integration: ingest files, then search them."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test notes
        notes_dir = os.path.join(tmpdir, "notes")
        os.makedirs(os.path.join(notes_dir, "ml"))
        with open(os.path.join(notes_dir, "ml", "test.md"), "w") as f:
            f.write("## Neural Networks\nNeural networks learn by adjusting weights.\n")

        db_path = os.path.join(tmpdir, ".chroma")

        result = ingest_directory(notes_dir, db_path=db_path)
        assert result["files_processed"] == 1
        assert result["chunks_stored"] >= 1

        # Search
        results = search_notes("neural networks", db_path=db_path)
        assert len(results) >= 1
        assert "neural" in results[0]["text"].lower()


def test_ingest_skips_binary():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write a binary file with .txt extension
        with open(os.path.join(tmpdir, "binary.txt"), "wb") as f:
            f.write(b"\x00\x01\x02\xff\xfe")

        # Write a valid text file
        with open(os.path.join(tmpdir, "valid.txt"), "w") as f:
            f.write("This is valid text content.")

        db_path = os.path.join(tmpdir, ".chroma")
        result = ingest_directory(tmpdir, db_path=db_path)
        assert result["files_processed"] == 1
        assert len(result["files_skipped"]) >= 1


def test_ingest_empty_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, ".chroma")
        result = ingest_directory(tmpdir, db_path=db_path)
        assert result["files_processed"] == 0
        assert "message" in result


def test_ingest_missing_directory():
    try:
        ingest_directory("/nonexistent/path/xyz")
        assert False, "Should have raised"
    except FileNotFoundError:
        pass


def test_search_empty_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, ".chroma")
        results = search_notes("anything", db_path=db_path)
        assert results == []
