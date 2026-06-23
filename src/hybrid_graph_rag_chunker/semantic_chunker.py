"""
Semantic chunking strategy (PDF §10) — splits a block of leaf-section
text into at most ``MAX_CHUNKS_PER_DOC`` topic-coherent, dynamically
sized chunks, sized for 8B-parameter locally hosted LLMs.

Tail-call-optimized recursive functions (via tail_call.py):

  * ``enforce_max_chunks`` — repeatedly merges the most-similar
    adjacent pair until group count reaches the cap. Each call is a
    self-tail-call on the smaller group list; on pathological input
    (thousands of initial groups) a naive recursive version would
    overflow the call stack.

  * ``split_oversized_group`` — bisects an oversized group at its
    weakest semantic boundary, recursing on each half. In the worst
    case (monotone text with many sentences), splitting depth is
    O(log S) where S = sentence count, which is safe; but for
    explicitness and correctness at very large S we tail_call_optimized it
    anyway. Because this recursion fans out (each call may produce two
    recursive branches), a single tail_call_optimized driver can't unroll it
    directly: instead it returns a pre-order accumulated list via a
    tail-call optimized work-list (``_split_step``).

  * ``validate_token_budgets`` delegates its undersized-group merge to
    ``functools.reduce`` (the tail-recursive accumulate/merge loop it
    replaced is now O(1) stack depth in reduce's C-level loop).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import reduce
from typing import Any, Callable, Sequence

import numpy as np

from .constants import (
    BLANK_CHAR,
    COSINE_EPSILON,
    FULL_COHERENCE,
    GROUP_KEY_CENTROID,
    GROUP_KEY_EMBEDDINGS,
    GROUP_KEY_SENTENCES,
    MAX_CHUNKS_PER_DOC,
    MAX_TOKENS_PER_CHUNK,
    MIN_TOKEN_ESTIMATE,
    MIN_TOKENS_PER_CHUNK,
    MONOTONOUS_SIMILARITY_THRESHOLD,
    SENTENCE_BOUNDARY_PATTERN,
    SIMILARITY_THRESHOLD,
    SPACY_LANGUAGE,
    SPACY_SENTENCIZER_PIPE,
    TOKENS_PER_WORD_RATIO,
)
from .ai_client import EmbeddingClient
from .chunk_strategy import BaseChunker, TextChunk
from .tail_call import TailCall, tail_call_optimized

Vector = np.ndarray
Group = dict  # {"sentences": list[str], "embeddings": list[Vector], "centroid": Vector}

_SENTENCE_BOUNDARY_RE = re.compile(SENTENCE_BOUNDARY_PATTERN)


def cosine_sim(a: Vector, b: Vector) -> float:
    return float(
        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + COSINE_EPSILON)
    )


def estimate_tokens(text: str) -> int:
    return max(
        MIN_TOKEN_ESTIMATE, int(len(text.split()) * TOKENS_PER_WORD_RATIO)
    )


# ── Step 1: Sentence Segmentation ──────────────────────────────────────
def _segment_with_spacy(text: str) -> list[str]:
    import spacy  # type: ignore

    nlp = spacy.blank(SPACY_LANGUAGE)
    nlp.add_pipe(SPACY_SENTENCIZER_PIPE)
    return [s.text.strip() for s in nlp(text).sents if s.text.strip()]


def _segment_with_regex(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_BOUNDARY_RE.split(text) if s.strip()]


def segment_sentences(text: str) -> list[str]:
    try:
        return _segment_with_spacy(text)
    except ImportError:
        return _segment_with_regex(text)


# ── Step 3: Boundary Detection ─────────────────────────────────────────
def detect_boundaries(
    embeddings: Sequence[Vector], threshold: float = SIMILARITY_THRESHOLD
) -> list[int]:
    return [
        i + 1
        for i in range(len(embeddings) - 1)
        if cosine_sim(embeddings[i], embeddings[i + 1]) < threshold
    ]


# ── Step 4: Form Initial Semantic Groups ───────────────────────────────
def form_groups(
    sentences: list[str], embeddings: list[Vector], boundaries: list[int]
) -> list[Group]:
    splits = [0, *boundaries, len(sentences)]
    spans = [
        (start, end) for start, end in zip(splits, splits[1:]) if start < end
    ]
    return [
        {
            GROUP_KEY_SENTENCES: sentences[start:end],
            GROUP_KEY_EMBEDDINGS: embeddings[start:end],
            GROUP_KEY_CENTROID: np.mean(embeddings[start:end], axis=0),
        }
        for start, end in spans
    ]


# ── Step 5: Dynamic Merging — tail-call optimized ──────────────────────
def _merge_two(a: Group, b: Group) -> Group:
    combined = a[GROUP_KEY_EMBEDDINGS] + b[GROUP_KEY_EMBEDDINGS]
    return {
        GROUP_KEY_SENTENCES: a[GROUP_KEY_SENTENCES] + b[GROUP_KEY_SENTENCES],
        GROUP_KEY_EMBEDDINGS: combined,
        GROUP_KEY_CENTROID: np.mean(combined, axis=0),
    }


def merge_most_similar_pair(groups: list[Group]) -> list[Group]:
    if len(groups) <= 1:
        return groups
    similarities = [
        cosine_sim(
            groups[i][GROUP_KEY_CENTROID], groups[i + 1][GROUP_KEY_CENTROID]
        )
        for i in range(len(groups) - 1)
    ]
    idx = int(np.argmax(similarities))
    return [
        *groups[:idx],
        _merge_two(groups[idx], groups[idx + 1]),
        *groups[idx + 2 :],
    ]


def _enforce_max_chunks_step(groups: list[Group], max_chunks: int) -> Any:
    return (
        groups
        if len(groups) <= max_chunks
        else TailCall(
            lambda: _enforce_max_chunks_step(
                merge_most_similar_pair(groups), max_chunks
            )
        )
    )


enforce_max_chunks = tail_call_optimized(_enforce_max_chunks_step)


# ── Step 6: Token-Budget Validation — tail-call optimized split ─────────
def _split_at(group: Group, index: int) -> tuple[Group, Group]:
    left_embs, right_embs = (
        group[GROUP_KEY_EMBEDDINGS][:index],
        group[GROUP_KEY_EMBEDDINGS][index:],
    )
    return (
        {
            GROUP_KEY_SENTENCES: group[GROUP_KEY_SENTENCES][:index],
            GROUP_KEY_EMBEDDINGS: left_embs,
            GROUP_KEY_CENTROID: np.mean(left_embs, axis=0),
        },
        {
            GROUP_KEY_SENTENCES: group[GROUP_KEY_SENTENCES][index:],
            GROUP_KEY_EMBEDDINGS: right_embs,
            GROUP_KEY_CENTROID: np.mean(right_embs, axis=0),
        },
    )


def _weakest_boundary_index(embeddings: list[Vector]) -> int:
    sims = [
        cosine_sim(embeddings[i], embeddings[i + 1])
        for i in range(len(embeddings) - 1)
    ]
    return (
        len(embeddings) // 2
        if min(sims) > MONOTONOUS_SIMILARITY_THRESHOLD
        else int(np.argmin(sims)) + 1
    )


def _split_step(
    pending: list[Group], done: list[Group], max_tokens: int
) -> Any:
    """
    Work-list tail-call optimized traversal: `pending` holds groups still to be
    examined; `done` accumulates already-validated output. Because
    splitting fans out (one group becomes two), a simple tail-recursive
    "split then recurse on right" would still have linear depth on
    pathological monotone text. The work-list keeps the Python call
    stack O(1) regardless.
    """
    if not pending:
        return done
    head, *rest = pending
    text = BLANK_CHAR.join(head[GROUP_KEY_SENTENCES])
    if (
        estimate_tokens(text) <= max_tokens
        or len(head[GROUP_KEY_EMBEDDINGS]) <= 1
    ):
        return TailCall(lambda: _split_step(rest, [*done, head], max_tokens))
    left, right = _split_at(
        head, _weakest_boundary_index(head[GROUP_KEY_EMBEDDINGS])
    )
    return TailCall(lambda: _split_step([left, right, *rest], done, max_tokens))


_split_oversized = tail_call_optimized(_split_step)


def split_oversized_groups(
    groups: list[Group], max_tokens: int = MAX_TOKENS_PER_CHUNK
) -> list[Group]:
    return _split_oversized(groups, [], max_tokens)


def _merge_if_undersized(
    groups: list[Group], candidate: Group, min_tokens: int
) -> list[Group]:
    text = BLANK_CHAR.join(candidate[GROUP_KEY_SENTENCES])
    is_undersized = bool(groups) and estimate_tokens(text) < min_tokens
    return (
        [*groups[:-1], _merge_two(groups[-1], candidate)]
        if is_undersized
        else [*groups, candidate]
    )


def validate_token_budgets(
    groups: list[Group],
    max_tokens: int = MAX_TOKENS_PER_CHUNK,
    min_tokens: int = MIN_TOKENS_PER_CHUNK,
) -> list[Group]:
    expanded = split_oversized_groups(groups, max_tokens)
    return reduce(
        lambda acc, g: _merge_if_undersized(acc, g, min_tokens), expanded, []
    )


def _coherence(embeddings: list[Vector]) -> float:
    return (
        float(
            np.mean(
                [
                    cosine_sim(embeddings[i], embeddings[i + 1])
                    for i in range(len(embeddings) - 1)
                ]
            )
        )
        if len(embeddings) > 1
        else FULL_COHERENCE
    )


def _group_to_chunk(group: Group) -> TextChunk:
    text = BLANK_CHAR.join(group[GROUP_KEY_SENTENCES])
    return TextChunk(
        text=text,
        coherence_score=_coherence(group[GROUP_KEY_EMBEDDINGS]),
    )


def _single_sentence_chunk(
    sentence: str, embedder: EmbeddingClient
) -> TextChunk:
    return TextChunk(
        text=sentence,
        coherence_score=FULL_COHERENCE,
    )


class SemanticChunker(BaseChunker):
    """
    A chunking strategy that semantically groups text units (sentences or paragraphs)
    using embeddings and a similarity threshold.
    """

    def __init__(
        self,
        split_fn: Callable[[str], list[str]] = segment_sentences,
        max_chunks: int = MAX_CHUNKS_PER_DOC,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        max_tokens: int = MAX_TOKENS_PER_CHUNK,
        min_tokens: int = MIN_TOKENS_PER_CHUNK,
    ):
        self.split_fn = split_fn
        self.max_chunks = max_chunks
        self.similarity_threshold = similarity_threshold
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens

    async def chunk(
        self, text: str, embedder: EmbeddingClient
    ) -> list[TextChunk]:
        units = self.split_fn(text)
        if not units:
            return []
        if len(units) == 1:
            return [_single_sentence_chunk(units[0], embedder)]

        raw_embeddings = await embedder.embed(units)

        if (
            not raw_embeddings
            or len(raw_embeddings) != len(units)
            or any(e is None for e in raw_embeddings)
        ):
            raise RuntimeError(
                "Embedding failed: remote server returned empty, mismatched, or None embeddings. "
                "Ensure your embedding server (e.g. LM Studio) is running and reachable."
            )
        embeddings = [np.asarray(e) for e in raw_embeddings]
        boundaries = detect_boundaries(
            embeddings, threshold=self.similarity_threshold
        )
        groups = form_groups(units, embeddings, boundaries)
        groups = enforce_max_chunks(groups, self.max_chunks)
        groups = validate_token_budgets(
            groups, self.max_tokens, self.min_tokens
        )

        return [_group_to_chunk(g) for g in groups]
