"""RAG platform API — FastAPI service.

Endpoints:
  POST /ingest  chunk + embed + store documents
  POST /ask     retrieve top-k chunks and answer with citations
  GET  /health  liveness/readiness
  GET  /metrics Prometheus metrics

All external AI calls go through small client interfaces (see
embeddings.py / llm.py) so the service can run against any
OpenAI-compatible endpoint — or against fake clients in tests.
No real API keys are needed to run the service locally or in CI.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest
from pydantic import BaseModel, Field

from .chunking import chunk_text
from .embeddings import (
    EmbeddingClient,
    FakeEmbeddingClient,
    OpenAICompatibleEmbeddingClient,
)
from .llm import FakeLLMClient, LLMClient, OpenAICompatibleLLMClient
from .vectorstore import Chunk, InMemoryVectorStore, QdrantVectorStore, VectorStore

# ---------------------------------------------------------------------------
# Configuration (all overridable with environment variables)
# ---------------------------------------------------------------------------


@dataclass
class Settings:
    vector_store: str = "memory"  # "memory" or "qdrant"
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "rag_docs"
    embedding_base_url: str = ""  # e.g. https://api.openai.com/v1
    embedding_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    llm_base_url: str = ""  # e.g. https://api.openai.com/v1
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    chunk_words: int = 100
    chunk_overlap_words: int = 20
    default_top_k: int = 4


def load_settings() -> Settings:
    """Read settings from the environment, falling back to sane defaults."""
    return Settings(
        vector_store=os.getenv("VECTOR_STORE", "memory"),
        qdrant_url=os.getenv("QDRANT_URL", "http://qdrant:6333"),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "rag_docs"),
        embedding_base_url=os.getenv("EMBEDDING_BASE_URL", ""),
        embedding_api_key=os.getenv("EMBEDDING_API_KEY", ""),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        embedding_dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "1536")),
        llm_base_url=os.getenv("LLM_BASE_URL", ""),
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        chunk_words=int(os.getenv("CHUNK_WORDS", "100")),
        chunk_overlap_words=int(os.getenv("CHUNK_OVERLAP_WORDS", "20")),
        default_top_k=int(os.getenv("DEFAULT_TOP_K", "4")),
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

INGEST_REQUESTS = Counter("rag_ingest_requests_total", "Total /ingest requests")
INGEST_CHUNKS = Counter("rag_ingest_chunks_total", "Total chunks ingested")
ASK_REQUESTS = Counter("rag_ask_requests_total", "Total /ask requests")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    doc_id: str = Field(..., min_length=1, description="Unique document id")
    text: str = Field(..., min_length=1, description="Document text (markdown ok)")
    title: str | None = None
    source: str | None = None


class IngestResponse(BaseModel):
    doc_id: str
    chunks_stored: int
    chunk_ids: list[str]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=4, ge=1, le=20)


class Citation(BaseModel):
    doc_id: str
    chunk_index: int
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]


class HealthResponse(BaseModel):
    status: str
    vector_store: str
    chunks_indexed: int


# ---------------------------------------------------------------------------
# Client wiring
# ---------------------------------------------------------------------------


def build_embedding_client(settings: Settings) -> EmbeddingClient:
    """Real OpenAI-compatible client when configured, fake otherwise.

    The fake client is deterministic and needs no key — it exists so the
    service can run locally and in tests without any external service.
    """
    if settings.embedding_base_url and settings.embedding_api_key:
        return OpenAICompatibleEmbeddingClient(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )
    return FakeEmbeddingClient(dimensions=128)


def build_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_base_url and settings.llm_api_key:
        return OpenAICompatibleLLMClient(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
        )
    return FakeLLMClient()


def build_vector_store(
    settings: Settings, embedding_dimensions: int
) -> VectorStore:
    if settings.vector_store == "qdrant":
        return QdrantVectorStore(
            url=settings.qdrant_url,
            collection=settings.qdrant_collection,
            dimensions=embedding_dimensions,
        )
    return InMemoryVectorStore()


# ---------------------------------------------------------------------------
# App factory (injectable clients make the service testable without fakes
# leaking into production code paths)
# ---------------------------------------------------------------------------


def create_app(
    settings: Settings | None = None,
    embedding_client: EmbeddingClient | None = None,
    llm_client: LLMClient | None = None,
    vector_store: VectorStore | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    embeddings = embedding_client or build_embedding_client(settings)
    llm = llm_client or build_llm_client(settings)
    store = vector_store or build_vector_store(settings, embeddings.dimensions)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Qdrant needs its collection to exist before the first upsert.
        store.ensure_collection()
        yield

    app = FastAPI(title="rag-platform", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.embeddings = embeddings
    app.state.llm = llm
    app.state.store = store

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(req: IngestRequest) -> IngestResponse:
        INGEST_REQUESTS.inc()
        texts = chunk_text(
            req.text,
            chunk_words=settings.chunk_words,
            overlap_words=settings.chunk_overlap_words,
        )
        if not texts:
            raise HTTPException(status_code=400, detail="No chunkable text found")
        vectors = embeddings.embed(texts)
        chunks = [
            Chunk(
                doc_id=req.doc_id,
                chunk_index=i,
                text=text,
                source=req.source,
                title=req.title,
            )
            for i, text in enumerate(texts)
        ]
        chunk_ids = store.upsert(chunks, vectors)
        INGEST_CHUNKS.inc(len(chunks))
        return IngestResponse(
            doc_id=req.doc_id, chunks_stored=len(chunks), chunk_ids=chunk_ids
        )

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        ASK_REQUESTS.inc()
        top_k = req.top_k or settings.default_top_k
        query_vector = embeddings.embed([req.question])[0]
        results = store.search(query_vector, top_k=top_k)
        if not results:
            return AskResponse(
                answer="I don't have any documents indexed yet, so I can't "
                "answer from retrieved context. Ingest some documents first.",
                citations=[],
            )
        context = "\n\n".join(
            f"[Source {i + 1} — {r.chunk.doc_id}]\n{r.chunk.text}"
            for i, r in enumerate(results)
        )
        prompt = (
            "Answer the question using ONLY the context below. "
            "If the context does not contain the answer, say so.\n\n"
            f"Context:\n{context}\n\nQuestion: {req.question}\nAnswer:"
        )
        answer = llm.complete(prompt)
        citations = [
            Citation(
                doc_id=r.chunk.doc_id,
                chunk_index=r.chunk.chunk_index,
                score=round(r.score, 4),
                text=r.chunk.text,
            )
            for r in results
        ]
        return AskResponse(answer=answer, citations=citations)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            vector_store=settings.vector_store,
            chunks_indexed=store.count(),
        )

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


# Default app instance for `uvicorn app.main:app`
app = create_app()
