"""
AI clients for connecting to remote models.

Exposes an OpenAI-compatible REST API for embeddings and chat completions
(e.g., via a local LM Studio server).
"""

from __future__ import annotations

from typing import Optional

import httpx

from .constants import (
    API_FIELD_CHOICES,
    API_FIELD_DATA,
    API_FIELD_EMBEDDING,
    API_FIELD_INDEX,
    API_FIELD_INPUT,
    API_FIELD_MESSAGE,
    API_FIELD_MESSAGES,
    API_FIELD_MODEL,
    API_FIELD_ROLE,
    DEFAULT_API_INDEX,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHAT_URL,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDINGS_URL,
    DEFAULT_TIMEOUT_SECONDS,
    FIELD_CONTENT,
    ROLE_SYSTEM,
    ROLE_USER,
)


class AIClient:
    """Base class for AI clients connecting to remote models."""

    def __init__(self, base_url: str, model: str, timeout: float) -> None:
        self.base_url = base_url
        self.model = model
        self.timeout = timeout


class EmbeddingClient(AIClient):
    """Thin client around LM Studio's `/v1/embeddings` endpoint."""

    def __init__(
        self,
        base_url: str = DEFAULT_EMBEDDINGS_URL,
        model: str = DEFAULT_EMBEDDING_MODEL,
        dim: int = DEFAULT_EMBEDDING_DIM,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__(base_url, model, timeout)
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[Optional[list[float]]]:
        """Embed a batch of texts, returning one vector per text.

        Returns ``None`` for every position in the batch if the remote
        server is unreachable or returns an error. The caller is
        responsible for handling ``None`` values.
        """
        if not texts:
            return []
        try:
            return await self._embed_remote(texts)
        except Exception:  # noqa: BLE001
            return [None] * len(texts)

    async def _embed_remote(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.base_url,
                json={API_FIELD_MODEL: self.model, API_FIELD_INPUT: texts},
                timeout=self.timeout,
            )
        response.raise_for_status()
        payload = response.json()
        # OpenAI-style response: {"data": [{"embedding": [...], "index": 0}, ...]}
        ordered = sorted(
            payload[API_FIELD_DATA],
            key=lambda item: item.get(API_FIELD_INDEX, DEFAULT_API_INDEX),
        )
        return [item[API_FIELD_EMBEDDING] for item in ordered]


class LLMClient(AIClient):
    """Thin client around LM Studio's `/v1/chat/completions` endpoint."""

    def __init__(
        self,
        base_url: str = DEFAULT_CHAT_URL,
        model: str = DEFAULT_CHAT_MODEL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__(base_url, model, timeout)

    async def generate(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> Optional[str]:
        """Generate a text response from the LLM.

        Returns ``None`` if the remote server is unreachable or returns an error.
        """
        messages = []
        if system_prompt:
            messages.append(
                {API_FIELD_ROLE: ROLE_SYSTEM, FIELD_CONTENT: system_prompt}
            )
        messages.append({API_FIELD_ROLE: ROLE_USER, FIELD_CONTENT: prompt})

        try:
            return await self._generate_remote(messages)
        except Exception:  # noqa: BLE001
            return None

    async def _generate_remote(self, messages: list[dict[str, str]]) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.base_url,
                json={
                    API_FIELD_MODEL: self.model,
                    API_FIELD_MESSAGES: messages,
                },
                timeout=self.timeout,
            )
        response.raise_for_status()
        payload = response.json()
        return payload[API_FIELD_CHOICES][0][API_FIELD_MESSAGE][FIELD_CONTENT]
