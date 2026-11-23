"""API tests: /health, /ingest, /ask (with citations), /metrics."""

K8S_DOC = (
    "Kubernetes deployments use a rolling update strategy. "
    "The deployment controller replaces pods gradually, checking "
    "readiness probes before moving on, so releases happen with "
    "zero downtime for users."
)


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["vector_store"] == "memory"
    assert body["chunks_indexed"] == 0


def test_ingest_stores_chunks(client):
    resp = client.post(
        "/ingest",
        json={"doc_id": "k8s-guide", "text": K8S_DOC, "source": "runbook.md"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["doc_id"] == "k8s-guide"
    assert body["chunks_stored"] == 1
    assert len(body["chunk_ids"]) == 1

    health = client.get("/health").json()
    assert health["chunks_indexed"] == 1


def test_ingest_rejects_empty_text(client):
    # Whitespace-only text passes pydantic's min_length but yields no
    # chunks, so the handler rejects it with 400.
    resp = client.post("/ingest", json={"doc_id": "empty", "text": "   "})
    assert resp.status_code == 400


def test_ask_returns_citations(client):
    client.post("/ingest", json={"doc_id": "k8s-guide", "text": K8S_DOC})
    client.post(
        "/ingest",
        json={
            "doc_id": "cooking",
            "text": "Pasta recipes with tomato sauce garlic basil and parmesan.",
        },
    )

    resp = client.post(
        "/ask", json={"question": "How do Kubernetes deployments update?", "top_k": 2}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]
    assert body["citations"], "expected at least one citation"
    # The kubernetes doc must be cited first.
    assert body["citations"][0]["doc_id"] == "k8s-guide"
    assert body["citations"][0]["chunk_index"] == 0
    assert "Kubernetes" in body["citations"][0]["text"]
    assert body["citations"][0]["score"] > 0


def test_ask_with_no_documents_is_graceful(client):
    resp = client.post("/ask", json={"question": "What is Kubernetes?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["citations"] == []
    assert "don't have any documents" in body["answer"]


def test_metrics_endpoint_exposes_counters(client):
    client.post("/ask", json={"question": "hello?"})
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "rag_ask_requests_total" in resp.text
    assert "rag_ingest_requests_total" in resp.text
