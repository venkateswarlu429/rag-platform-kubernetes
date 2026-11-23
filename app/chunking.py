"""Word-based text chunking with exact overlap.

Why word-based instead of character-based: the overlap between
consecutive chunks is exact — chunk[i+1] starts with the same words
chunk[i] ends with — which is easy to reason about, easy to test, and
avoids splitting words in half. Chunk boundaries therefore never cut
a word, and retrieval always sees whole words.
"""

from __future__ import annotations


def chunk_text(
    text: str, chunk_words: int = 100, overlap_words: int = 20
) -> list[str]:
    """Split ``text`` into word chunks with overlapping windows.

    Args:
        text: Raw document text (plain text or markdown).
        chunk_words: Maximum words per chunk.
        overlap_words: Words shared between consecutive chunks
            (must be smaller than ``chunk_words``).

    Returns:
        A list of chunk strings. A document shorter than one chunk
        returns a single chunk. Empty/whitespace-only input returns [].
    """
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if not 0 <= overlap_words < chunk_words:
        raise ValueError("overlap_words must satisfy 0 <= overlap < chunk_words")

    words = text.split()
    if not words:
        return []

    step = chunk_words - overlap_words
    chunks: list[str] = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start : start + chunk_words]))
        if start + chunk_words >= len(words):
            break
        start += step
    return chunks
