"""Retrieval ranking with the fake embedding client.

The fake client is vocabulary-driven, so a document that shares words
with the query must rank above one that does not. This proves the
store's cosine ranking works end-to-end without any real model.
"""

from app.chunking import chunk_text
from app.embeddings import FakeEmbeddingClient
from app.vectorstore import Chunk, InMemoryVectorStore


def _ingest(store, embeddings, doc_id, text):
    texts = chunk_text(text, chunk_words=100, overlap_words=20)
    vectors = embeddings.embed(texts)
    chunks = [
        Chunk(doc_id=doc_id, chunk_index=i, text=t)
        for i, t in enumerate(texts)
    ]
    return store.upsert(chunks, vectors)


def test_relevant_doc_ranks_above_irrelevant_doc():
    store = InMemoryVectorStore()
    embeddings = FakeEmbeddingClient(dimensions=128)
    _ingest(
        store,
        embeddings,
        "k8s-doc",
        "Kubernetes deployment rolling update strategy with replicas and "
        "readiness probes for zero downtime releases",
    )
    _ingest(
        store,
        embeddings,
        "cooking-doc",
        "Pasta recipes with tomato sauce garlic basil and parmesan cheese "
        "for a quick Italian dinner",
    )

    query_vector = embeddings.embed(["kubernetes deployment"])[0]
    results = store.search(query_vector, top_k=2)

    assert len(results) == 2
    assert results[0].chunk.doc_id == "k8s-doc"
    assert results[1].chunk.doc_id == "cooking-doc"
    assert results[0].score > results[1].score


def test_search_respects_top_k_and_order():
    store = InMemoryVectorStore()
    embeddings = FakeEmbeddingClient(dimensions=128)
    for i in range(5):
        _ingest(store, embeddings, f"doc-{i}", "kubernetes cluster node pod")
    results = store.search(embeddings.embed(["kubernetes"])[0], top_k=3)
    assert len(results) == 3
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_reingest_replaces_chunks():
    store = InMemoryVectorStore()
    embeddings = FakeEmbeddingClient(dimensions=128)
    _ingest(store, embeddings, "doc-1", "kubernetes deployment")
    assert store.count() == 1
    _ingest(store, embeddings, "doc-1", "kubernetes deployment updated text")
    assert store.count() == 1  # same doc re-ingested: no duplicates
    results = store.search(embeddings.embed(["kubernetes"])[0], top_k=1)
    assert "updated" in results[0].chunk.text
