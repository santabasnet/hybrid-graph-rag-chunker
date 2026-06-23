"""
Smoke + TCO stress tests for the chunking pipeline.

Run with:   uv run python tests/test_chunking.py

Every stress test sets ``sys.setrecursionlimit(500)`` before calling the
function under test.  If any tail-call optimized function falls back to native
recursion it will raise ``RecursionError`` long before reaching the test
assertion -- the tight limit makes non-optimized paths fail loudly.
"""

from __future__ import annotations

import asyncio
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from hybrid_graph_rag_chunker.chunk_builder import (
    chunk_markdown,
    chunk_plain_text,
)  # noqa: E402
from hybrid_graph_rag_chunker.ai_client import EmbeddingClient  # noqa: E402
from hybrid_graph_rag_chunker.markdown_parser import parse_markdown_outline  # noqa: E402
from hybrid_graph_rag_chunker.models import DocumentStyle, NodeType  # noqa: E402
from hybrid_graph_rag_chunker.semantic_chunker import (  # noqa: E402
    enforce_max_chunks,
    split_oversized_groups,
)

_TIGHT_RECURSION_LIMIT = 500
_DEFAULT_RECURSION_LIMIT = sys.getrecursionlimit()

_rng = np.random.default_rng(42)


# Always mock EmbeddingClient.embed for testing to avoid requiring a running LM Studio server
async def _mock_embed(self, texts: list[str]) -> list[list[float]]:
    # Generate deterministic vectors
    return [[float(x) for x in _rng.normal(size=self.dim)] for _ in texts]


EmbeddingClient.embed = _mock_embed

# ── Correctness tests ───────────────────────────────────────────────────


def test_outline_handles_nested_headings() -> None:
    md = (
        "# Book\n\n## Chapter 1\n\nintro text\n\n### Section 1.1\n\nbody text\n"
    )
    outline = parse_markdown_outline(md)
    assert outline.title == "Document"
    assert len(outline.children) == 1
    chapter = outline.children[0]
    assert chapter.title == "Chapter 1"
    assert chapter.level == 2
    assert chapter.children[0].title == "Section 1.1"
    print("test_outline_handles_nested_headings:              OK")


def test_outline_handles_no_headings() -> None:
    md = "Just a flat paragraph of text with no markdown headings at all."
    outline = parse_markdown_outline(md)
    assert outline.children == []
    assert outline.content == md
    print("test_outline_handles_no_headings:                  OK")


@pytest.mark.asyncio
async def test_chunk_markdown_produces_valid_tree() -> None:
    md = (
        "# Demo Doc\n\n"
        "## Section A\n\nSome content for section A that is reasonably short.\n\n"
        "## Section B\n\nSome content for section B that is also reasonably short.\n"
    )
    embedder = EmbeddingClient()
    root = await chunk_markdown(
        md,
        doc_title="Demo Doc",
        organization="org-test",
        embedder=embedder,
    )

    assert root.node_type is NodeType.DOCUMENT
    assert root.organization == "org-test"

    docs = root.flatten()
    ids = {d["id"] for d in docs}
    assert len(ids) == len(docs), "every flattened node must have a unique id"
    assert all(
        d["parent_id"] in ids for d in docs if d["node_type"] != "document"
    )
    assert all(
        "embedding" in d
        and len(d["embedding"]) > 0
        and "content" in d
        and d["content"]
        for d in docs
        if d["node_type"] == "chunk"
    )
    print(
        f"test_chunk_markdown_produces_valid_tree:           OK  ({len(docs)} nodes)"
    )


def test_fenced_code_not_parsed_as_heading() -> None:
    md = "# Title\n\nSome text.\n\n```python\n# not a heading\ndef f(): pass\n```\n\n## Real Section\n\nmore.\n"
    outline = parse_markdown_outline(md)
    all_titles = [n.title for n in outline.children] + [
        n.title for n in outline.children[0].children
    ]
    assert "not a heading" not in all_titles
    print("test_fenced_code_not_parsed_as_heading:            OK")


# ── TCO stress tests — must pass with a tight recursion limit ───────────


def test_tco_parser_many_siblings() -> None:
    """parse_markdown_outline must handle thousands of same-level headings
    without hitting Python's call stack (tail-call optimized fold over headings)."""
    N = 3000
    md = "\n\n".join(f"## Heading {i}\n\nsome text here." for i in range(N))
    sys.setrecursionlimit(_TIGHT_RECURSION_LIMIT)
    try:
        outline = parse_markdown_outline(md)
    finally:
        sys.setrecursionlimit(_DEFAULT_RECURSION_LIMIT)
    assert len(outline.children) == N
    print(
        f"test_tco_parser_many_siblings:                     OK  ({N} siblings, limit={_TIGHT_RECURSION_LIMIT})"
    )


def test_tco_parser_deep_nesting() -> None:
    """parse_markdown_outline must handle deeply nested headings without
    overflowing the call stack."""
    depth = 6  # markdown max heading level
    md = "\n\n".join(f"{'#' * (i + 1)} Level {i}\n\ntext" for i in range(depth))
    sys.setrecursionlimit(_TIGHT_RECURSION_LIMIT)
    try:
        outline = parse_markdown_outline(md)
    finally:
        sys.setrecursionlimit(_DEFAULT_RECURSION_LIMIT)
    # _collapse_lone_h1_title promotes the single H1 into the root,
    # so root.children[0] is the H2 ("Level 1").
    assert outline.children[0].title == "Level 1"
    print(
        f"test_tco_parser_deep_nesting:                      OK  (depth={depth}, limit={_TIGHT_RECURSION_LIMIT})"
    )


@pytest.mark.asyncio
async def test_tco_flatten_wide_tree() -> None:
    """ChunkNode.flatten() and .walk() must handle trees with thousands of
    sibling nodes without overflowing the call stack."""
    N = 3000
    md = "\n\n".join(f"## Heading {i}\n\nsome text here." for i in range(N))
    root = await chunk_markdown(
        md, doc_title="Wide Doc", embedder=EmbeddingClient()
    )

    sys.setrecursionlimit(_TIGHT_RECURSION_LIMIT)
    try:
        docs = root.flatten()
        node_count = len(list(root.walk()))
    finally:
        sys.setrecursionlimit(_DEFAULT_RECURSION_LIMIT)

    assert len(docs) == node_count
    print(
        f"test_tco_flatten_wide_tree:                        OK  ({len(docs)} docs, limit={_TIGHT_RECURSION_LIMIT})"
    )


def test_tco_enforce_max_chunks() -> None:
    """enforce_max_chunks must handle thousands of initial groups without
    overflowing the call stack (tail-call optimized self-recursive step)."""
    DIM = 4
    N = 2000
    groups = [
        {
            "sentences": [f"s{i}"],
            "embeddings": [_rng.normal(size=DIM)],
            "centroid": _rng.normal(size=DIM),
        }
        for i in range(N)
    ]
    sys.setrecursionlimit(_TIGHT_RECURSION_LIMIT)
    try:
        merged = enforce_max_chunks(groups, 10)
    finally:
        sys.setrecursionlimit(_DEFAULT_RECURSION_LIMIT)
    assert len(merged) <= 10
    print(
        f"test_tco_enforce_max_chunks:                       OK  ({N} groups -> {len(merged)}, limit={_TIGHT_RECURSION_LIMIT})"
    )


def test_tco_split_oversized_groups_splittable() -> None:
    """split_oversized_groups must bisect a large group down to T_max without
    overflowing the call stack (tail-call optimized work-list bisection)."""
    DIM = 4
    short_sent = "word " * 100  # ~75 tokens, well under 512 individually
    big_group = {
        "sentences": [short_sent] * 50,  # 50 × 75 = 3750 tokens >> T_max=512
        "embeddings": [_rng.normal(size=DIM) for _ in range(50)],
        "centroid": np.ones(DIM),
    }
    sys.setrecursionlimit(_TIGHT_RECURSION_LIMIT)
    try:
        split = split_oversized_groups([big_group], max_tokens=512)
    finally:
        sys.setrecursionlimit(_DEFAULT_RECURSION_LIMIT)
    assert all(
        len(" ".join(g["sentences"]).split()) * 0.75 <= 512 for g in split
    )
    print(
        f"test_tco_split_oversized_groups_splittable:        OK  (1 group -> {len(split)}, limit={_TIGHT_RECURSION_LIMIT})"
    )


def test_tco_split_oversized_groups_unsplittable() -> None:
    """A single-sentence group that exceeds T_max must be returned as-is
    (unsplittable by design: 1 embedding = no internal boundary to split at)."""
    DIM = 4
    long_sent = "word " * 800  # ~600 tokens > 512, but only 1 sentence
    unsplittable = {
        "sentences": [long_sent],
        "embeddings": [_rng.normal(size=DIM)],
        "centroid": np.ones(DIM),
    }
    sys.setrecursionlimit(_TIGHT_RECURSION_LIMIT)
    try:
        result = split_oversized_groups([unsplittable], max_tokens=512)
    finally:
        sys.setrecursionlimit(_DEFAULT_RECURSION_LIMIT)
    assert len(result) == 1, (
        "single-sentence oversized group must be returned unchanged"
    )
    print(
        f"test_tco_split_oversized_groups_unsplittable:      OK  (limit={_TIGHT_RECURSION_LIMIT})"
    )


# ── Flat plain-text correctness tests ──────────────────────────────────


@pytest.mark.asyncio
async def test_plain_text_single_paragraph() -> None:
    """A single-paragraph plain-text document (no blank lines) falls back to
    chunk_markdown's no-headings path; the root must be DOCUMENT and
    document_style must be 'flat' in all Solr docs."""
    text = (
        "This is a single paragraph with no blank lines whatsoever. "
        "It contains multiple sentences but no heading structure."
    )
    root = await chunk_plain_text(
        text, doc_title="Single Para", embedder=EmbeddingClient()
    )
    assert root.node_type is NodeType.DOCUMENT
    assert root.document_style == DocumentStyle.FLAT
    docs = root.flatten()
    assert all(
        d.get("document_style") == DocumentStyle.FLAT.value for d in docs
    ), "every Solr doc from a flat-text root must carry document_style='flat'"
    print("test_plain_text_single_paragraph:                  OK")


@pytest.mark.asyncio
async def test_plain_text_multi_paragraph() -> None:
    """A multi-paragraph plain-text document must produce a DOCUMENT root
    with CHUNK children (no SECTION intermediary) and document_style='flat'."""
    text = (
        "Retrieval-augmented generation combines parametric and non-parametric memory.\n\n"
        "Large language models are trained on vast corpora of text.\n\n"
        "Semantic chunking splits documents at topic boundaries.\n\n"
        "Graph-augmented retrieval encodes explicit relationships between chunks."
    )
    root = await chunk_plain_text(
        text, doc_title="Multi Para Doc", embedder=EmbeddingClient()
    )
    assert root.node_type is NodeType.DOCUMENT
    assert root.document_style == DocumentStyle.FLAT
    assert len(root.children) >= 1

    chunk_types = {c.node_type for c in root.children}
    assert chunk_types == {NodeType.CHUNK}, (
        "flat-text children must all be CHUNK nodes (no SECTION wrappers)"
    )

    docs = root.flatten()
    chunk_docs = [d for d in docs if d["node_type"] == "chunk"]
    assert all(
        d.get("document_style") == DocumentStyle.FLAT.value for d in docs
    )
    assert all("embedding" in d and d["embedding"] for d in chunk_docs)
    assert all("content" in d and d["content"] for d in chunk_docs)
    print(
        f"test_plain_text_multi_paragraph:                   OK  ({len(chunk_docs)} chunks)"
    )


@pytest.mark.asyncio
async def test_plain_text_valid_ids() -> None:
    """Parent/child ID integrity for the flat-text tree (mirrors the
    corresponding test for chunk_markdown)."""
    text = "\n\n".join(
        f"Paragraph {i}: some content about topic {i}." for i in range(6)
    )
    root = await chunk_plain_text(
        text,
        doc_title="ID Test Doc",
        organization="org-test",
        embedder=EmbeddingClient(),
    )
    docs = root.flatten()
    ids = {d["id"] for d in docs}
    assert len(ids) == len(docs), "every flattened node must have a unique id"
    assert all(
        d["parent_id"] in ids for d in docs if d["node_type"] != "document"
    ), "every non-root node's parent_id must reference a node in the flat list"
    assert root.organization == "org-test"
    print(
        f"test_plain_text_valid_ids:                         OK  ({len(docs)} nodes)"
    )


# ── Flat plain-text TCO stress test ────────────────────────────────────


@pytest.mark.asyncio
async def test_tco_plain_text_many_paragraphs() -> None:
    """chunk_plain_text must process hundreds of paragraphs without hitting
    Python's call stack (enforce_max_chunks + validate_token_budgets are
    already optimized; this test confirms the whole pipeline stays O(1))."""
    N = 500
    text = "\n\n".join(
        f"Paragraph {i} discusses topic {i} in detail, providing context."
        for i in range(N)
    )
    sys.setrecursionlimit(_TIGHT_RECURSION_LIMIT)
    try:
        root = await chunk_plain_text(
            text, doc_title="TCO Flat Doc", embedder=EmbeddingClient()
        )
    finally:
        sys.setrecursionlimit(_DEFAULT_RECURSION_LIMIT)
    assert root.node_type is NodeType.DOCUMENT
    assert root.document_style == DocumentStyle.FLAT
    assert len(root.children) >= 1
    print(
        f"test_tco_plain_text_many_paragraphs:               OK  "
        f"({N} paragraphs -> {len(root.children)} chunks, limit={_TIGHT_RECURSION_LIMIT})"
    )


async def main():
    # Correctness
    test_outline_handles_nested_headings()
    test_outline_handles_no_headings()
    await test_chunk_markdown_produces_valid_tree()
    test_fenced_code_not_parsed_as_heading()
    # TCO stress
    test_tco_parser_many_siblings()
    test_tco_parser_deep_nesting()
    await test_tco_flatten_wide_tree()
    test_tco_enforce_max_chunks()
    test_tco_split_oversized_groups_splittable()
    test_tco_split_oversized_groups_unsplittable()
    # Flat plain-text correctness
    await test_plain_text_single_paragraph()
    await test_plain_text_multi_paragraph()
    await test_plain_text_valid_ids()
    # Flat plain-text TCO stress
    await test_tco_plain_text_many_paragraphs()


if __name__ == "__main__":
    asyncio.run(main())
