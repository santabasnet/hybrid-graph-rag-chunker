"""
Fixed-Size sliding-window chunking strategy.
"""

from __future__ import annotations

from typing import Iterator

from .chunk_strategy import BaseChunker, TextChunk
from .constants import (
    BLANK_CHAR,
    DEFAULT_FIXED_CHUNK_OVERLAP,
    DEFAULT_FIXED_CHUNK_SIZE,
)
from .ai_client import EmbeddingClient


def _word_slices(
    words: list[str], chunk_size: int, step: int
) -> Iterator[list[str]]:
    """Yield successive word-window slices of ``words``."""
    n = len(words)
    for start in range(0, n, step):
        yield words[start : start + chunk_size]
        if start + chunk_size >= n:
            break


class FixedSizeChunker(BaseChunker):
    """
    Embedding-free, sliding-window chunking strategy.

    Splits text into windows of at most ``chunk_size`` words, advancing
    by ``chunk_size - overlap`` words between windows so that each pair of
    adjacent chunks shares ``overlap`` words of context.

    Parameters
    ----------
    chunk_size:
        Maximum number of words per chunk. Defaults to
        ``DEFAULT_FIXED_CHUNK_SIZE`` (from ``constants.py``).
    overlap:
        Number of words shared between adjacent chunks. Must be strictly
        less than ``chunk_size``. Defaults to ``DEFAULT_FIXED_CHUNK_OVERLAP``.

    Examples
    --------
    >>> chunker = FixedSizeChunker(chunk_size=5, overlap=1)
    >>> chunks = chunker.chunk("a b c d e f g h", embedder=None)
    >>> [chunk.text for chunk in chunks]
    ['a b c d e', 'e f g h']
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_FIXED_CHUNK_SIZE,
        overlap: int = DEFAULT_FIXED_CHUNK_OVERLAP,
    ) -> None:
        if overlap < 0:
            raise ValueError(f"overlap must be non-negative, got {overlap}")
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be strictly less than "
                f"chunk_size ({chunk_size})"
            )
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._step = chunk_size - overlap

    # ------------------------------------------------------------------
    # ChunkingStrategy protocol
    # ------------------------------------------------------------------

    async def chunk(
        self, text: str, embedder: EmbeddingClient | None = None
    ) -> list[TextChunk]:
        """
        Split *text* into overlapping fixed-size windows.

        The ``embedder`` parameter is accepted for protocol conformance but
        is never used — ``FixedSizeChunker`` is embedding-free by design.
        """
        words = text.split()
        if not words:
            return []
        return [
            TextChunk(text=BLANK_CHAR.join(window))
            for window in _word_slices(words, self._chunk_size, self._step)
        ]

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"chunk_size={self._chunk_size}, "
            f"overlap={self._overlap})"
        )
