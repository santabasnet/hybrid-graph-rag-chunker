"""
Builds the recursive Solr-schema ``ChunkNode`` tree (models.py) from a
parsed Markdown outline (markdown_parser.py), running the semantic
chunking algorithm (semantic_chunker.py) on each section's own text.

Mapping to PDF §2.1 / §4.2:

    MarkdownSection(level=0)      -> ChunkNode(node_type=DOCUMENT)
    MarkdownSection(level=1..6)   -> ChunkNode(node_type=SECTION)
    text directly under a heading -> one or more ChunkNode(node_type=CHUNK)

Tail-call-optimized tree build (via tail_call.py):

    ``_build_node`` converts a ``MarkdownSection`` subtree into a
    ``ChunkNode`` subtree. The naive recursive formulation recurses
    once per section in the document outline. A document with many
    deeply nested headings would exhaust the call stack; we tail_call_optimized
    the build via an explicit bottom-up work-list
    (``_build_node_step``) that accumulates finished ``ChunkNode``
    children before constructing their parent, keeping Python
    call-stack depth O(1).

Flat plain-text path (``chunk_plain_text``):

    When the input has no ATX heading structure, ``chunk_plain_text``
    splits the text at blank-line boundaries (``split_paragraphs``),
    embeds each paragraph, and feeds them through the same semantic
    grouping pipeline used for heading sections (boundary detection,
    enforce_max_chunks, validate_token_budgets). The resulting
    CHUNK leaf nodes are hung directly under a single DOCUMENT root,
    and every node in the tree carries ``document_style="flat"``.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import replace
from typing import Any

from .constants import (
    CHUNK_PATH_PREFIX,
    DEFAULT_SLUG,
    DEFAULT_UNTITLED_DOCUMENT_TITLE,
    DOCUMENT_LEVEL,
    PART_TITLE_SEPARATOR,
    PATH_SEPARATOR,
    SLUG_PATTERN,
    SLUG_SEPARATOR,
)
from .ai_client import EmbeddingClient
from .markdown_parser import (
    MarkdownSection,
    parse_markdown_outline,
    split_paragraphs,
)
from .models import ChunkNode, DocumentStyle, NodeType
from .semantic_chunker import SemanticChunker, segment_sentences
from .chunk_strategy import BaseChunker
from .tail_call import AsyncTailCall, async_tail_call_optimized

_SLUG_RE = re.compile(SLUG_PATTERN)


def slugify(text: str) -> str:
    return (
        _SLUG_RE.sub(SLUG_SEPARATOR, text.lower().strip()).strip(SLUG_SEPARATOR)
        or DEFAULT_SLUG
    )


def _join_path(prefix: str, slug: str) -> str:
    return slug if not prefix else f"{prefix}{PATH_SEPARATOR}{slug}"


def _chunk_path(section_path: str, i: int) -> str:
    return f"{section_path}{PATH_SEPARATOR}{CHUNK_PATH_PREFIX}{i}"


def _chunk_title(title: str, i: int, multi: bool) -> str:
    return f"{title}{PART_TITLE_SEPARATOR}{i + 1}" if multi else title


def _fmt_coherence_score(score: float | None) -> str | None:
    """Format a coherence score as a 2-decimal string, or return None."""
    return f"{score:.2f}" if score is not None else None


async def _build_chunks(
    section: MarkdownSection,
    embedder: EmbeddingClient,
    doc_id: str,
    section_path: str,
    parent_id: str,
    level: int,
    strategy: BaseChunker,
    document_style: DocumentStyle = DocumentStyle.STRUCTURED,
) -> list[ChunkNode]:
    """Chunk `section`'s own text into leaf ChunkNode(s) using the provided strategy.

    Parameters
    ----------
    document_style:
        Passed through to every ``ChunkNode`` produced.  Use
        ``DocumentStyle.FLAT`` for the heading-free plain-text path so that
        every output node is correctly tagged; defaults to ``STRUCTURED``.
    """
    if not section.content.strip():
        return []
    text_chunks = await strategy.chunk(section.content, embedder)
    if not text_chunks:
        return []
    chunk_embeddings = await embedder.embed(
        [chunk.text for chunk in text_chunks]
    )
    has_multiple_chunks = len(text_chunks) > 1
    return [
        ChunkNode(
            id=str(uuid.uuid4()),
            doc_id=doc_id,
            node_type=NodeType.CHUNK,
            title=_chunk_title(section.title, i, has_multiple_chunks),
            content=chunk.text,
            level=level,
            section_path=_chunk_path(section_path, i),
            embedding=embedding,
            chunk_order=i,
            chunk_group_id=i,
            group_coherence_score=_fmt_coherence_score(chunk.coherence_score),
            parent_id=parent_id,
            document_style=document_style,
        )
        for i, (chunk, embedding) in enumerate(
            zip(text_chunks, chunk_embeddings)
        )
    ]


# ── Work-list frame used during bottom-up build ─────────────────────────
# Each pending entry is one of:
#   ("section", MarkdownSection, section_path, parent_id, doc_id, org) — not yet converted
#   ("node",    ChunkNode)                                               — already converted
#
# The algorithm runs in two passes per section:
#   1. Push children first (so they are converted before their parent).
#   2. When all children of a section have been converted and sit on the
#      done-stack, pop them and assemble the parent ChunkNode.
#
# To know when a section's children are ready we track how many children
# each section expects via a ("wait", expected_children, section, ...) frame.

_FRAME_WAIT = "wait"
_FRAME_SECTION = "section"
_FRAME_NODE = "node"


async def _build_step(
    pending: list[tuple],
    done: list[ChunkNode],
    embedder: EmbeddingClient,
    strategy: BaseChunker,
) -> Any:
    if not pending:
        return done

    frame = pending[0]
    rest = pending[1:]

    if frame[0] == _FRAME_NODE:
        return AsyncTailCall(
            lambda: _build_step(rest, [*done, frame[1]], embedder, strategy)
        )

    if frame[0] == _FRAME_SECTION:
        _, section, section_path, parent_id, doc_id, org = frame
        child_frames = [
            (
                _FRAME_SECTION,
                child,
                _join_path(section_path, slugify(child.title)),
                None,
                doc_id,
                org,
            )
            for child in section.children
        ]
        wait_frame = (
            _FRAME_WAIT,
            len(section.children),
            section,
            section_path,
            parent_id,
            doc_id,
            org,
        )
        return AsyncTailCall(
            lambda: _build_step(
                [*child_frames, wait_frame, *rest], done, embedder, strategy
            )
        )

    # _FRAME_WAIT — all children have been pushed; the last N items in `done`
    # are the finished child nodes (in the same left-to-right order as children).
    _, n_children, section, section_path, parent_id, doc_id, org = frame
    child_nodes, earlier = (
        done[-n_children:] if n_children else [],
        done[:-n_children] if n_children else done,
    )

    node_id = str(uuid.uuid4())
    node_type = (
        NodeType.DOCUMENT
        if section.level == DOCUMENT_LEVEL
        else NodeType.SECTION
    )
    chunk_children = await _build_chunks(
        section,
        embedder,
        doc_id,
        section_path,
        node_id,
        section.level + 1,
        strategy,
    )

    wired_children = chunk_children + [
        replace(child_node, parent_id=node_id, doc_id=doc_id)
        for child_node in child_nodes
    ]

    node = ChunkNode(
        id=node_id,
        doc_id=doc_id,
        node_type=node_type,
        title=section.title,
        level=section.level,
        section_path=section_path,
        organization=org,
        children=wired_children,
    )
    return AsyncTailCall(
        lambda: _build_step(rest, [*earlier, node], embedder, strategy)
    )


_build_step_tree = async_tail_call_optimized(_build_step)


async def _build_node(
    root_section: MarkdownSection,
    embedder: EmbeddingClient,
    doc_id: str,
    organization: str | None,
    doc_slug: str,
    strategy: BaseChunker,
) -> ChunkNode:
    initial = [
        (_FRAME_SECTION, root_section, doc_slug, None, doc_id, organization)
    ]
    result = await _build_step_tree(initial, [], embedder, strategy)
    return result[0]


async def chunk_markdown(
    markdown_text: str,
    doc_title: str = DEFAULT_UNTITLED_DOCUMENT_TITLE,
    organization: str | None = None,
    doc_id: str | None = None,
    embedder: EmbeddingClient | None = None,
    strategy: BaseChunker | None = None,
) -> ChunkNode:
    """
    End-to-end entrypoint: parse `markdown_text`'s heading hierarchy,
    chunk each section's content, embed every chunk, and
    return the root ``ChunkNode`` of the resulting recursive tree.

    Call ``.flatten()`` on the result to get the flat list of
    independent Solr documents (schema §4.1) ready for indexing.
    """
    embedder = embedder or EmbeddingClient()
    doc_id = doc_id or str(uuid.uuid4())
    strategy = strategy or SemanticChunker(split_fn=segment_sentences)
    outline = replace(parse_markdown_outline(markdown_text), title=doc_title)
    return await _build_node(
        outline, embedder, doc_id, organization, slugify(doc_title), strategy
    )


async def chunk_plain_text(
    text: str,
    doc_title: str = DEFAULT_UNTITLED_DOCUMENT_TITLE,
    organization: str | None = None,
    doc_id: str | None = None,
    embedder: EmbeddingClient | None = None,
    strategy: BaseChunker | None = None,
) -> ChunkNode:
    """
    End-to-end entrypoint for flat, heading-free plain-text documents.

    Uses the provided chunking strategy to break the text into pieces.
    If no strategy is provided, it defaults to `SemanticChunker(split_fn=split_paragraphs)`
    (falling back to `segment_sentences` if the text has only 1 paragraph).

    Every node in the returned tree carries ``document_style="flat"`` so
    downstream code (and Solr queries) can distinguish this path from the
    structured Markdown path.
    """
    embedder = embedder or EmbeddingClient()
    doc_id = doc_id or str(uuid.uuid4())
    doc_slug = slugify(doc_title)

    if strategy is None:
        paragraphs = split_paragraphs(text)
        strategy = (
            SemanticChunker(split_fn=segment_sentences)
            if len(paragraphs) <= 1
            else SemanticChunker(split_fn=split_paragraphs)
        )

    node_id = str(uuid.uuid4())
    # Reuse _build_chunks: wrap the raw text in a minimal MarkdownSection so
    # the shared helper can chunk, embed, and assemble the leaf ChunkNodes.
    flat_section = MarkdownSection(
        level=DOCUMENT_LEVEL,
        title=doc_title,
        content=text,
    )
    chunk_children = await _build_chunks(
        flat_section,
        embedder,
        doc_id,
        doc_slug,
        node_id,
        DOCUMENT_LEVEL + 1,
        strategy,
        document_style=DocumentStyle.FLAT,
    )

    return ChunkNode(
        id=node_id,
        doc_id=doc_id,
        node_type=NodeType.DOCUMENT,
        title=doc_title,
        level=DOCUMENT_LEVEL,
        section_path=doc_slug,
        organization=organization,
        children=chunk_children,
        document_style=DocumentStyle.FLAT,
    )
