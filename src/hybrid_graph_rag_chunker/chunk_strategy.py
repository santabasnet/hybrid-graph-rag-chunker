"""
Pluggable chunking strategy — protocol, output contract, and built-in implementations.

Architecture
────────────
Every chunking strategy must satisfy the ``ChunkingStrategy`` Protocol:

    class ChunkingStrategy(Protocol):
        async def chunk(self, text: str, embedder: EmbeddingClient) -> list[TextChunk]: ...

Callers (e.g. ``chunk_builder._build_chunks``) only depend on this protocol,
so any concrete strategy can be injected or swapped at runtime without
blocking the event loop or changing the calling code in ``chunk_builder.py``:

Built-in strategies
───────────────────
``SemanticChunker``   (semantic_chunker.py)
    Embedding-based; groups sentences/paragraphs by cosine similarity.
    Imported here only for re-export convenience — implementation lives in
    semantic_chunker.py to keep the embedding/NLP logic self-contained.

``FixedSizeChunker``  (fixed_size_chunker.py)
    Embedding-free; uses a sliding word-window of configurable size and
    overlap. Suitable when latency or infrastructure constraints make
    embedding calls impractical, or as a fast baseline.

Output contract
───────────────
Both strategies return ``list[TextChunk]`` so ``chunk_builder`` can iterate
them uniformly — the only structural difference being that ``SemanticChunker``
sets ``coherence_score`` while ``FixedSizeChunker`` leaves it ``None``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .ai_client import EmbeddingClient


# ── Output contract ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TextChunk:
    """
    Immutable output unit produced by any ``ChunkingStrategy``.

    Attributes
    ----------
    text:
        The raw chunk text, ready to embed and store.
    coherence_score:
        Mean pairwise cosine similarity among the chunk's constituent
        sentence embeddings (``SemanticChunker``), or ``None`` when the
        strategy has no concept of semantic coherence (``FixedSizeChunker``).
    """

    text: str
    coherence_score: float | None = field(default=None)


# ── Abstract Base Class ──────────────────────────────────────────────────────


class BaseChunker(ABC):
    """
    Abstract base class satisfied by every built-in and user-defined chunking
    strategy. This enables explicit inheritance and a future policy factory.

    The ``embedder`` argument is supplied by the caller even for strategies
    that ignore it, so implementations never have to worry about wiring it
    themselves.  Embedding-free strategies simply declare the parameter and
    do not call it.
    """

    @abstractmethod
    async def chunk(
        self, text: str, embedder: EmbeddingClient
    ) -> list[TextChunk]:
        pass
