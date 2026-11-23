"""Shared pytest fixtures.

The API is exercised through its factory with fake embedding/LLM
clients and the in-memory store, so tests never touch the network or
any real model endpoint.
"""

import pytest
from fastapi.testclient import TestClient

from app.embeddings import FakeEmbeddingClient
from app.llm import FakeLLMClient
from app.main import Settings, create_app
from app.vectorstore import InMemoryVectorStore


@pytest.fixture
def fake_clients():
    return {
        "embeddings": FakeEmbeddingClient(dimensions=128),
        "llm": FakeLLMClient(),
        "store": InMemoryVectorStore(),
    }


@pytest.fixture
def client(fake_clients):
    settings = Settings(
        vector_store="memory",
        chunk_words=100,
        chunk_overlap_words=20,
        default_top_k=4,
    )
    app = create_app(
        settings=settings,
        embedding_client=fake_clients["embeddings"],
        llm_client=fake_clients["llm"],
        vector_store=fake_clients["store"],
    )
    with TestClient(app) as test_client:
        yield test_client
