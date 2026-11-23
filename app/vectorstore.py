"""Vector stores.

One interface, two implementations:

- ``InMemoryVectorStore`` — exact cosine similarity over an in-process
  list. Used for local dev and tests; data is lost on restart.
- ``QdrantVectorStore`` — the same interface over Qdrant's HTTP API
  (the Qdrant StatefulSet in k8s/). No qdrant-client dependency: the
  REST API is small and httpx is already a dependency.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Protocol

import httpx


@dataclass
class Chunk:
    """One chunk of an ingested document, ready to embed and store."""

    doc_id: str
    chunk_index: int
    text: str
    source: str | None = None
    title: str | None = None
    # Stable id derived from doc_id + chunk_index (set in __post_init__).
    id: str = field(init=False)

    def __post_init__(self) -> None:
        self.id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.doc_id}:{self.chunk_index}"))


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float  # cosine similarity, -1..1 (higher = more similar)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity; returns 0.0 if either vector is all zeros."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorStore(Protocol):
    """Minimal interface the API depends on."""

    def ensure_collection(self) -> None:
        """Create the backing collection/index if it does not exist."""
        ...

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> list[str]:
        """Store chunks with their vectors; return the chunk ids."""
        ...

    def search(self, query_vector: list[float], top_k: int = 4) -> list[ScoredChunk]:
        """Return the top-k most similar chunks, best first."""
        ...

    def count(self) -> int:
        """Number of chunks currently stored."""
        ...


class InMemoryVectorStore:
    """Exact in-memory cosine search. Ephemeral — for dev and tests."""

    def __init__(self) -> None:
        self._items: list[tuple[Chunk, list[float]]] = []

    def ensure_collection(self) -> None:
        return None

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> list[str]:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        # Replace any existing chunks with the same ids (re-ingest = update).
        new_ids = {x.id for x in chunks}
        self._items = [(c, v) for c, v in self._items if c.id not in new_ids]
        self._items.extend(zip(chunks, vectors))
        return [c.id for c in chunks]

    def search(self, query_vector: list[float], top_k: int = 4) -> list[ScoredChunk]:
        scored = [
            ScoredChunk(chunk=c, score=cosine_similarity(query_vector, v))
            for c, v in self._items
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        return len(self._items)


class QdrantVectorStore:
    """Qdrant over its REST API (no extra client dependency).

    Collection is created on startup (``ensure_collection``) with cosine
    distance. Payloads carry doc_id / chunk_index / text / source / title
    so retrieved points can be cited without a second lookup.
    """

    def __init__(self, url: str, collection: str, dimensions: int) -> None:
        self.url = url.rstrip("/")
        self.collection = collection
        self.dimensions = dimensions

    def ensure_collection(self) -> None:
        resp = httpx.get(f"{self.url}/collections/{self.collection}", timeout=10.0)
        if resp.status_code == 200:
            return
        create = httpx.put(
            f"{self.url}/collections/{self.collection}",
            json={"vectors": {"size": self.dimensions, "distance": "Cosine"}},
            timeout=10.0,
        )
        create.raise_for_status()

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> list[str]:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        points = [
            {
                "id": chunk.id,
                "vector": vector,
                "payload": {
                    "doc_id": chunk.doc_id,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "source": chunk.source,
                    "title": chunk.title,
                },
            }
            for chunk, vector in zip(chunks, vectors)
        ]
        resp = httpx.put(
            f"{self.url}/collections/{self.collection}/points",
            json={"points": points},
            timeout=30.0,
        )
        resp.raise_for_status()
        return [c.id for c in chunks]

    def search(self, query_vector: list[float], top_k: int = 4) -> list[ScoredChunk]:
        resp = httpx.post(
            f"{self.url}/collections/{self.collection}/points/search",
            json={
                "vector": query_vector,
                "limit": top_k,
                "with_payload": True,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        results = []
        for hit in resp.json().get("result", []):
            payload = hit["payload"]
            results.append(
                ScoredChunk(
                    chunk=Chunk(
                        doc_id=payload["doc_id"],
                        chunk_index=payload["chunk_index"],
                        text=payload["text"],
                        source=payload.get("source"),
                        title=payload.get("title"),
                    ),
                    score=float(hit["score"]),
                )
            )
        return results

    def count(self) -> int:
        resp = httpx.get(f"{self.url}/collections/{self.collection}", timeout=10.0)
        resp.raise_for_status()
        return int(resp.json()["result"]["points_count"])
