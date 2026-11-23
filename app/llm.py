"""LLM clients.

Same pattern as embeddings.py: one small interface, a real
OpenAI-compatible implementation, and a deterministic fake for tests.

The fake does NOT pretend to reason — it echoes the first context
chunk back, which is enough to prove the retrieval -> prompt ->
answer plumbing works without any external service.
"""

from __future__ import annotations

from typing import Protocol

import httpx


class LLMClient(Protocol):
    """Minimal interface the rest of the app depends on."""

    def complete(self, prompt: str) -> str:
        """Return the model's completion for ``prompt``."""
        ...


class OpenAICompatibleLLMClient:
    """Chat completions via any OpenAI-compatible ``/chat/completions``."""

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def complete(self, prompt: str) -> str:
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You answer questions using only the "
                        "context provided. If the answer is not in the "
                        "context, say so.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


class FakeLLMClient:
    """Deterministic stand-in: answers by quoting the first context chunk.

    The /ask prompt embeds retrieved chunks as "[Source 1 — ...]" blocks,
    so quoting the first one back proves end-to-end wiring (retrieve ->
    prompt -> answer) without calling any model.
    """

    def complete(self, prompt: str) -> str:
        marker = "[Source 1 —"
        if marker in prompt:
            chunk_text = prompt.split(marker, 1)[1]
            # Strip the "doc_id]" header line, keep the chunk body.
            body = chunk_text.split("\n", 1)[1] if "\n" in chunk_text else chunk_text
            # Cut off before the next source block / question.
            body = body.split("[Source ", 1)[0].split("\n\nQuestion:", 1)[0].strip()
            return f"Based on the retrieved context: {body[:200]}"
        return "No context was provided, so I cannot answer from documents."
