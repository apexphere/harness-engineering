import json
import os
import tempfile

from src.sample_notes import generate_samples, SAMPLE_NOTES, SAMPLE_GOLDEN_SET


def test_generate_creates_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = os.path.join(tmpdir, "notes")
        golden_path = os.path.join(tmpdir, "golden_set.json")
        # Temporarily change to tmpdir so golden_set.json lands there
        original_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            generate_samples(out_dir)
            for relative_path in SAMPLE_NOTES:
                full = os.path.join(out_dir, relative_path)
                assert os.path.exists(full), f"Missing: {relative_path}"
            assert os.path.exists(golden_path)
        finally:
            os.chdir(original_cwd)


def test_golden_set_has_20_questions():
    assert len(SAMPLE_GOLDEN_SET["questions"]) == 20


def test_golden_set_questions_have_required_fields():
    for q in SAMPLE_GOLDEN_SET["questions"]:
        assert "id" in q
        assert "question" in q
        assert "required_concepts" in q
        assert "required_sources" in q
        assert "forbidden_content" in q
        assert "max_words" in q


def test_golden_set_ids_are_unique():
    ids = [q["id"] for q in SAMPLE_GOLDEN_SET["questions"]]
    assert len(ids) == len(set(ids))


def test_sample_notes_cover_golden_set_sources():
    """Every source in the golden set must exist in sample notes."""
    all_sources = set()
    for q in SAMPLE_GOLDEN_SET["questions"]:
        for s in q["required_sources"]:
            all_sources.add(s)
    for source in all_sources:
        assert source in SAMPLE_NOTES, f"Golden set references {source} but no sample note exists"


def test_generate_is_idempotent():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = os.path.join(tmpdir, "notes")
        original_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            generate_samples(out_dir)
            generate_samples(out_dir)  # second call should not raise
            assert os.path.exists(os.path.join(out_dir, "ml/attention.md"))
        finally:
            os.chdir(original_cwd)
