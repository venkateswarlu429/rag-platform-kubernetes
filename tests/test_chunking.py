"""Chunking: sizes, exact overlap, and edge cases."""

import pytest

from app.chunking import chunk_text


def _words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_short_text_is_a_single_chunk():
    assert chunk_text("hello world") == ["hello world"]


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\t  ") == []


def test_chunk_sizes_and_exact_overlap():
    text = _words(250)
    chunks = chunk_text(text, chunk_words=100, overlap_words=20)
    sizes = [len(c.split()) for c in chunks]
    assert sizes == [100, 100, 90]
    # Overlap is exact: the last 20 words of chunk[i] are the first 20
    # words of chunk[i+1].
    for i in range(len(chunks) - 1):
        tail = chunks[i].split()[-20:]
        head = chunks[i + 1].split()[:20]
        assert tail == head


def test_no_word_is_split():
    text = _words(30, prefix="longword")
    for chunk in chunk_text(text, chunk_words=10, overlap_words=2):
        for word in chunk.split():
            assert word.startswith("longword")


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_text(_words(50), chunk_words=10, overlap_words=10)
    with pytest.raises(ValueError):
        chunk_text(_words(50), chunk_words=10, overlap_words=11)
