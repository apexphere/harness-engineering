# LEARN: Layer 2 Foundation — Getting Real Data into Vector Storage
#
# This module handles the bridge between your filesystem and the vector
# database. It reads files, splits them into chunks, and stores them as
# embeddings that the tools layer can search.
#
# Key concepts:
#   - Chunking: splitting documents into pieces small enough for the model's
#     context window, but large enough to be meaningful
#   - Heading-aware splitting: using document structure (## headings) to
#     create semantically coherent chunks
#   - Overlap: including some text from the previous chunk to preserve
#     context across chunk boundaries
#   - Metadata: storing source file and position with each chunk so
#     answers can cite their sources

import os
import re

try:
    import chromadb
except ImportError:
    chromadb = None  # type: ignore

# Approximate tokens as words * 1.3 (rough heuristic, avoids tiktoken dependency)
TOKENS_PER_WORD = 1.3
DEFAULT_MAX_TOKENS = 512
DEFAULT_OVERLAP_TOKENS = 64


def _estimate_tokens(text: str) -> int:
    """Rough token estimate. Good enough for chunking decisions."""
    return int(len(text.split()) * TOKENS_PER_WORD)


def chunk_markdown(text: str, max_tokens: int = DEFAULT_MAX_TOKENS,
                   overlap_tokens: int = DEFAULT_OVERLAP_TOKENS) -> list[dict]:
    """Split markdown by ## headings, then by token windows if needed.

    Returns list of {"text": str, "heading": str | None, "char_start": int, "char_end": int}
    """
    # Split on ## headings
    sections = re.split(r"(?=^## )", text, flags=re.MULTILINE)
    chunks = []
    char_offset = 0

    for section in sections:
        if not section.strip():
            char_offset += len(section)
            continue

        # Extract heading if present
        heading = None
        heading_match = re.match(r"^## (.+)", section)
        if heading_match:
            heading = heading_match.group(1).strip()

        if _estimate_tokens(section) <= max_tokens:
            chunks.append({
                "text": section.strip(),
                "heading": heading,
                "char_start": char_offset,
                "char_end": char_offset + len(section),
            })
        else:
            # Window the section
            sub_chunks = chunk_text(section, max_tokens, overlap_tokens)
            for sc in sub_chunks:
                chunks.append({
                    "text": sc["text"],
                    "heading": heading,
                    "char_start": char_offset + sc["char_start"],
                    "char_end": char_offset + sc["char_end"],
                })

        char_offset += len(section)

    return chunks


def chunk_text(text: str, max_tokens: int = DEFAULT_MAX_TOKENS,
               overlap_tokens: int = DEFAULT_OVERLAP_TOKENS) -> list[dict]:
    """Split plain text into fixed-size windows with overlap.

    Returns list of {"text": str, "char_start": int, "char_end": int}
    """
    words = text.split()
    max_words = int(max_tokens / TOKENS_PER_WORD)
    overlap_words = int(overlap_tokens / TOKENS_PER_WORD)
    chunks = []
    start_word = 0

    while start_word < len(words):
        end_word = min(start_word + max_words, len(words))
        chunk_words = words[start_word:end_word]
        chunk_text_str = " ".join(chunk_words)

        # Calculate char positions
        char_start = len(" ".join(words[:start_word])) + (1 if start_word > 0 else 0)
        char_end = char_start + len(chunk_text_str)

        chunks.append({
            "text": chunk_text_str,
            "char_start": char_start,
            "char_end": char_end,
        })

        if end_word >= len(words):
            break
        start_word = end_word - overlap_words

    return chunks


def ingest_directory(
    directory: str,
    db_path: str = ".chroma",
    collection_name: str = "notes",
) -> dict:
    """Read all .md and .txt files, chunk them, and store in Chroma.

    Returns {"files_processed": int, "chunks_stored": int, "files_skipped": list[str]}
    """
    if chromadb is None:
        raise ImportError(
            "chromadb is not installed. Run: pip install chromadb"
        )

    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    client = chromadb.PersistentClient(path=db_path)

    # Delete existing collection to re-ingest cleanly
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.get_or_create_collection(name=collection_name)

    files_processed = 0
    chunks_stored = 0
    files_skipped: list[str] = []

    for root, _dirs, files in os.walk(directory):
        for filename in sorted(files):
            if not filename.endswith((".md", ".txt")):
                continue

            filepath = os.path.join(root, filename)
            relative_path = os.path.relpath(filepath, directory)

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                files_skipped.append(f"binary: {relative_path}")
                continue
            except FileNotFoundError:
                files_skipped.append(f"not found: {relative_path}")
                continue

            if not content.strip():
                files_skipped.append(f"empty: {relative_path}")
                continue

            # Choose chunking strategy based on file type
            if filename.endswith(".md"):
                chunks = chunk_markdown(content)
            else:
                chunks = chunk_text(content)

            for i, chunk in enumerate(chunks):
                chunk_id = f"{relative_path}::chunk_{i}"
                collection.add(
                    ids=[chunk_id],
                    documents=[chunk["text"]],
                    metadatas=[{
                        "source_file": relative_path,
                        "chunk_index": i,
                        "char_start": chunk["char_start"],
                        "char_end": chunk["char_end"],
                        "heading": chunk.get("heading") or "",
                    }],
                )
                chunks_stored += 1

            files_processed += 1

    if files_processed == 0:
        return {
            "files_processed": 0,
            "chunks_stored": 0,
            "files_skipped": files_skipped,
            "message": f"No .md or .txt files found in {directory}",
        }

    return {
        "files_processed": files_processed,
        "chunks_stored": chunks_stored,
        "files_skipped": files_skipped,
    }


def search_notes(
    query: str,
    db_path: str = ".chroma",
    collection_name: str = "notes",
    n_results: int = 5,
) -> list[dict]:
    """Search the Chroma collection for relevant chunks.

    Returns list of {"text": str, "source_file": str, "heading": str, "distance": float}
    """
    if chromadb is None:
        raise ImportError("chromadb is not installed. Run: pip install chromadb")

    client = chromadb.PersistentClient(path=db_path)

    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        return []

    if collection.count() == 0:
        return []

    results = collection.query(query_texts=[query], n_results=min(n_results, collection.count()))

    search_results = []
    for i in range(len(results["ids"][0])):
        search_results.append({
            "text": results["documents"][0][i],
            "source_file": results["metadatas"][0][i]["source_file"],
            "heading": results["metadatas"][0][i].get("heading", ""),
            "distance": results["distances"][0][i] if results["distances"] else 0.0,
        })

    return search_results
