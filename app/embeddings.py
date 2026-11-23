"""Embedding clients.

Two implementations behind one small interface:

- ``OpenAICompatibleEmbeddingClient`` — talks to any OpenAI-compatible
  ``/embeddings`` endpoint (OpenAI, Azure OpenAI, Ollama, vLLM, ...).
- ``FakeEmbeddingClient`` — deterministic, keyless, offline. Words are
  hashed into fixed buckets, so texts that share vocabulary end up
  close together in vector space. Used for local dev and tests; it is
  NOT a real embedding model and says so in its docstring.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import httpx

_WORD_RE = re.compile(r"[a-z0-9]+")


class EmbeddingClient(Protocol):
    """Minimal interface the rest of the app depends on."""

    dimensions: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one (L2-normalised) vector per input text."""
        ...


class OpenAICompatibleEmbeddingClient:
    """Embeddings via any OpenAI-compatible ``POST {base_url}/embeddings``."""

    def __init__(
        self, base_url: str, api_key: str, model: str, dimensions: int
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = httpx.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": texts},
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        # The API returns embeddings in request order; sort defensively.
        return [item["embedding"] for item in sorted(data, key=lambda d: d["index"])]


class FakeEmbeddingClient:
    """Deterministic bag-of-words vectors — for tests and offline dev only.

    Each word is hashed into one of ``dimensions`` buckets; the vector is
    the (L2-normalised) bucket count. Because similarity is driven by
    shared vocabulary, a query about "kubernetes" ranks a kubernetes doc
    above an unrelated doc. This is a stand-in, not a semantic model.
    """

    def __init__(self, dimensions: int = 32) -> None:
        self.dimensions = dimensions

    def _vector(self, text: str) -> list[float]:
        counts = [0.0] * self.dimensions
        for word in _WORD_RE.findall(text.lower()):
            bucket = int(hashlib.sha256(word.encode()).hexdigest(), 16)
            counts[bucket % self.dimensions] += 1.0
        norm = math.sqrt(sum(c * c for c in counts))
        if norm == 0:
            return counts  # empty text -> zero vector
        return [c / norm for c in counts]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]
