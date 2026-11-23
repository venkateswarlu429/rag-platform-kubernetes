# RAG Platform on Kubernetes

A retrieval-augmented generation (RAG) service you can run on Kubernetes:
ingest documents, retrieve the most relevant chunks for a question, and
answer with citations to the exact source chunks.

```
                    ┌──────────────┐
  POST /ingest ──►  │  FastAPI     │──embed──► Qdrant (vectors + payload)
  POST /ask    ──►  │  rag-api     │──search─► top-k chunks
                    │              │──LLM────► answer + citations
                    └──────────────┘
  GET /health, /metrics (Prometheus)
```

## Architecture

| Piece | Where | Notes |
|---|---|---|
| API | `app/` (FastAPI) | Chunking, embedding, retrieval, LLM calls, `/metrics` |
| Vector store | Qdrant (in-cluster) | `k8s/qdrant/` StatefulSet + PVC; in-memory store for local dev/tests |
| GitOps | ArgoCD | `argocd/` — one Application for Qdrant, one for the API |
| Infra | Terraform | `terraform/` — VPC, EKS cluster, managed node group, ECR repo |
| CI | GitHub Actions | `.github/workflows/ci.yml` — lint, pytest, docker build, tf fmt/validate |

### How ingestion works

1. `POST /ingest` receives `{doc_id, text, title?, source?}`.
2. The text is split into **word-based chunks** (default 100 words) with
   an exact **20-word overlap** between consecutive chunks, so no word is
   cut and the overlap is deterministic (`app/chunking.py`).
3. Each chunk is embedded through a configurable embedding client and
   stored in the vector store together with its payload
   (`doc_id`, `chunk_index`, `text`, `source`, `title`).

### How retrieval works

1. `POST /ask` embeds the question with the same embedding client.
2. The store returns the **top-k chunks by cosine similarity**.
3. The chunks are pasted into a prompt ("answer using ONLY this
   context") and sent to an LLM client.
4. The response returns the answer **with citations** — each citation
   names the `doc_id`, `chunk_index`, similarity `score`, and chunk text,
   so every claim can be traced back to a source document.
5. With no documents indexed, `/ask` says so instead of hallucinating.

### Clients are swappable

Both AI integrations are small interfaces (`app/embeddings.py`,
`app/llm.py`):

- **Real:** any OpenAI-compatible endpoint — set `EMBEDDING_BASE_URL` /
  `LLM_BASE_URL` + API keys + model names. Works with OpenAI, Azure
  OpenAI, Ollama, vLLM, etc.
- **Fake (default when no keys are set):** deterministic, offline
  clients. The fake embedder hashes words into buckets (vocabulary
  overlap drives similarity); the fake LLM quotes the retrieved context
  back. They exist so the service runs and tests anywhere with zero
  external services — they are **not** real models and must not be
  mistaken for embedding quality.

The vector store is also swappable: `VECTOR_STORE=memory` (default,
ephemeral, used by tests) or `VECTOR_STORE=qdrant` (persistent,
in-cluster). The Qdrant implementation speaks the REST API directly —
no extra client library.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r app/requirements.txt
pip install pytest

# Offline mode (fake clients, in-memory store) — no keys needed:
uvicorn app.main:app --reload --port 8000

# With a real provider (example: OpenAI):
EMBEDDING_BASE_URL=https://api.openai.com/v1 EMBEDDING_API_KEY=$KEY \
LLM_BASE_URL=https://api.openai.com/v1 LLM_API_KEY=$KEY \
VECTOR_STORE=memory uvicorn app.main:app --port 8000
```

```bash
# Ingest
curl -X POST localhost:8000/ingest -H 'Content-Type: application/json' \
  -d '{"doc_id": "k8s-guide", "text": "Kubernetes deployments use a rolling update strategy..."}'

# Ask (answer comes back with citations)
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "How do Kubernetes deployments update?", "top_k": 3}'

# Health & metrics
curl localhost:8000/health
curl localhost:8000/metrics
```

## Run the tests

```bash
python -m pytest -q        # 14 tests, all offline with fake clients
```

Tests cover: chunk sizes + exact overlap, retrieval ranking (a relevant
doc must outrank an irrelevant one), `/ask` returning citations,
`/health`, and Prometheus counters. They use fake embedding/LLM clients
— no real API calls are ever made.

## Deploy

### 1. Infrastructure (Terraform)

```bash
cd terraform
terraform init
terraform apply            # creates VPC, EKS cluster, node group, ECR repo
aws eks update-kubeconfig --name rag-platform-dev --region us-east-1
```

Costs: an EKS cluster + 2× t3.large nodes + one NAT gateway runs
roughly $150–200/month in us-east-1. Destroy when done:
`terraform destroy`.

### 2. Build & push the image

```bash
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker build -t <account>.dkr.ecr.us-east-1.amazonaws.com/rag-platform-api:latest .
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/rag-platform-api:latest
```

Then update the `image:` in `k8s/api/deployment.yaml`.

### 3. Deploy with ArgoCD (recommended) or kubectl

```bash
# Namespace + Qdrant first (the API needs Qdrant up before it starts)
kubectl apply -f k8s/namespace.yaml
kubectl apply -k k8s/qdrant/
kubectl apply -k k8s/api/

# Create the API keys secret (skip for offline/fake mode):
kubectl -n rag create secret generic rag-api-secrets \
  --from-literal=EMBEDDING_API_KEY='...' \
  --from-literal=LLM_API_KEY='...'

# Or let ArgoCD sync both Applications (Qdrant first):
kubectl apply -f argocd/application-qdrant.yaml
kubectl apply -f argocd/application-api.yaml
```

Update the `host:` in `k8s/api/ingress.yaml` and the `repoURL:` in
`argocd/*.yaml` to your own repo/domain before deploying.

## What is verified / what is not

**Verified in this repo (locally, before publishing):**

- All 14 pytest tests pass (chunking, retrieval ranking, API + citations,
  health, metrics) — run with fake clients, zero network calls.
- Every Python file compiles; `ruff` lint runs in CI.
- Every YAML manifest parses (`python -c yaml.safe_load_all`, see below).
- Terraform is `fmt`-clean and `terraform validate` passes (see below).
- The Docker image builds in CI (build-only, no push).

**NOT verified — do not assume these:**

- No live cluster deploy has been performed; the manifests have not been
  applied to a real EKS cluster.
- No real embedding or LLM quality has been measured. The fake clients
  are vocabulary stand-ins, not semantic models — there are **no
  benchmark numbers** in this repo, and retrieval quality against a real
  embedding model is untested.
- No load, latency, or cost measurements exist.
- Qdrant persistence, backup, and multi-node clustering are out of scope
  (single replica, 10Gi PVC).

## Project layout

```
app/                  FastAPI service (main, chunking, embeddings, llm, vectorstore)
tests/                pytest: chunking, retrieval ranking, API, metrics
k8s/
  namespace.yaml      rag namespace
  api/                ConfigMap, Deployment+Service, Ingress, kustomization
  qdrant/             StatefulSet+Service+PVC, kustomization
argocd/               Application manifests (qdrant, api)
terraform/            VPC, EKS, node group, ECR (fmt-clean, validated)
.github/workflows/    CI: lint, pytest, docker build, terraform fmt/validate
```
