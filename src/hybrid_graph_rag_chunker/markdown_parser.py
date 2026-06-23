"""
Markdown -> recursive hierarchy parser.

Walks a Markdown document's ATX headings (`#`, `##`, ... `######`) and
builds a recursive outline tree -- the same parent/child shape shown
for structured documents in PDF §2.1:

    Book
    ├── Chapter 1: Introduction
    │   ├── Section 1.1 — Motivation
    │   └── Section 1.2 — Problem Formulation
    └── Chapter 2: Methods
        ├── Section 2.1 — Embeddings
        └── Section 2.2 — Graph Retrieval

This module only builds the *outline* (titles + raw content per
heading). Turning leaf/section content into semantically-chunked
text and embeddings, and turning the outline into Solr-schema
``ChunkNode`` objects, happens in ``chunk_builder.py``.

The tree is built via a single left-to-right *fold* over an explicit,
immutable "open sections" stack -- a tail-recursive reformulation of
the classic stack-based heading-outline algorithm, optimized via tail-calls (see
tail_call.py) so it isn't bounded by Python's call-stack depth. A
naive recursive "parse this heading, then recurse into the rest of
its siblings" formulation chains one nested Python call per sibling
heading, so a document with thousands of same-level headings would
overflow the stack; folding over an explicit stack with a tail_call_optimized
keeps Python call-stack growth O(1) regardless of heading count.

``split_paragraphs`` is a thin helper for the flat plain-text path in
``chunk_builder.py``: it splits raw prose at blank lines so paragraph
breaks can seed the semantic-grouping algorithm without requiring any
ATX heading structure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any

from .constants import (
    BLANK_CHAR,
    CODE_FENCE_DELIMITER,
    DEFAULT_ROOT_TITLE,
    DOCUMENT_LEVEL,
    FRAME_KEY_CHILDREN,
    FRAME_KEY_CONTENT,
    FRAME_KEY_LEVEL,
    FRAME_KEY_TITLE,
    HEADING_GROUP_HASHES,
    HEADING_GROUP_TITLE,
    MAX_HEADING_LEVEL,
    NEWLINE,
    PARAGRAPH_BOUNDARY_PATTERN,
    TOP_HEADING_LEVEL,
)
from .tail_call import TailCall, tail_call_optimized

_PARAGRAPH_BOUNDARY_RE = re.compile(PARAGRAPH_BOUNDARY_PATTERN)

_HEADER_RE = re.compile(
    rf"^(?P<{HEADING_GROUP_HASHES}>#{{{1},{MAX_HEADING_LEVEL}}})\s+(?P<{HEADING_GROUP_TITLE}>.*?)\s*#*\s*$",
    re.MULTILINE,
)
_ESCAPED_FENCE = re.escape(CODE_FENCE_DELIMITER)
_FENCE_RE = re.compile(rf"{_ESCAPED_FENCE}.*?{_ESCAPED_FENCE}", re.DOTALL)


@dataclass
class MarkdownSection:
    """One heading-delimited block of a Markdown document."""

    level: int  # 0 = synthetic document root, 1..6 = '#'..'######'
    title: str
    content: str  # raw text directly under this heading, excluding subsections
    children: list[MarkdownSection] = field(default_factory=list)


# ── mask fenced code blocks so '#' inside them isn't mistaken for a heading ──
def _blank_line(line: str) -> str:
    return BLANK_CHAR * len(line)


def _blank_block(fence_match: re.Match) -> str:
    return NEWLINE.join(
        _blank_line(line) for line in fence_match.group(0).split(NEWLINE)
    )


def _mask_fenced_code(text: str) -> str:
    return _FENCE_RE.sub(_blank_block, text)


def _heading_level(heading_match: re.Match) -> int:
    return len(heading_match.group(HEADING_GROUP_HASHES))


def _section_span(
    markdown_text: str, matches: list[re.Match], index: int
) -> tuple[int, int]:
    start = matches[index].end()
    end = (
        matches[index + 1].start()
        if index + 1 < len(matches)
        else len(markdown_text)
    )
    return start, end


# ── Open-section stack: each frame is a plain dict (not yet a MarkdownSection,
# since its children are still being collected):
#     {"level": int, "title": str, "content": str, "children": tuple[MarkdownSection, ...]}
def _close_top(stack: tuple[dict, ...]) -> tuple[dict, ...]:
    """Pop the innermost open frame, turning it into a finished MarkdownSection,
    and file it under the new top frame's children."""
    closed = stack[-1]
    closed_section = MarkdownSection(
        level=closed[FRAME_KEY_LEVEL],
        title=closed[FRAME_KEY_TITLE],
        content=closed[FRAME_KEY_CONTENT],
        children=list(closed[FRAME_KEY_CHILDREN]),
    )
    parent = stack[-2]
    reparented = {
        **parent,
        FRAME_KEY_CHILDREN: (*parent[FRAME_KEY_CHILDREN], closed_section),
    }
    return (*stack[:-2], reparented)


def _close_while_step(stack: tuple[dict, ...], level: int) -> Any:
    return (
        stack
        if stack[-1][FRAME_KEY_LEVEL] < level
        else TailCall(lambda: _close_while_step(_close_top(stack), level))
    )


_close_while = tail_call_optimized(_close_while_step)


def _close_to_root_step(stack: tuple[dict, ...]) -> Any:
    return (
        stack
        if len(stack) == 1
        else TailCall(lambda: _close_to_root_step(_close_top(stack)))
    )


_close_to_root = tail_call_optimized(_close_to_root_step)


def _fold_headings_step(
    stack: tuple[dict, ...],
    matches: list[re.Match],
    markdown_text: str,
    index: int,
) -> Any:
    if index >= len(matches):
        return stack

    heading_level = _heading_level(matches[index])
    closed_stack = _close_while(
        stack, heading_level
    )  # close any open frames this heading doesn't nest under
    start, end = _section_span(markdown_text, matches, index)
    new_frame = {
        FRAME_KEY_LEVEL: heading_level,
        FRAME_KEY_TITLE: matches[index].group(HEADING_GROUP_TITLE).strip(),
        FRAME_KEY_CONTENT: markdown_text[start:end].strip(),
        FRAME_KEY_CHILDREN: (),
    }
    return TailCall(
        lambda: _fold_headings_step(
            (*closed_stack, new_frame), matches, markdown_text, index + 1
        )
    )


_fold_headings = tail_call_optimized(_fold_headings_step)


def _collapse_lone_h1_title(root: MarkdownSection) -> MarkdownSection:
    """
    Common Markdown pattern: a single top-level `#` heading is used as
    the *document title*, with everything else nested under it as
    `##`+ sections, e.g.::

        # Service Agreement Contract A
        ...preamble...
        ## Definitions
        ## Payment Terms

    Without this step the synthetic level-0 root and that lone level-1
    heading would both end up representing "the document", producing a
    redundant document -> section wrapper with a duplicated title and
    section_path. When the root has no preamble content of its own and
    exactly one level-1 child, that child's content/children are
    promoted directly onto the root (a new, immutable root is returned).
    """
    is_lone_h1_title = (
        not root.content.strip()
        and len(root.children) == 1
        and root.children[0].level == TOP_HEADING_LEVEL
    )
    return (
        replace(
            root,
            content=root.children[0].content,
            children=root.children[0].children,
        )
        if is_lone_h1_title
        else root
    )


def parse_markdown_outline(markdown_text: str) -> MarkdownSection:
    """
    Parse `markdown_text` into a recursive ``MarkdownSection`` tree.

    A synthetic level-0 root is always created to hold any preamble
    text that appears before the first heading. If the document has
    no headings at all, the root simply contains the whole text as a
    single (flat, non-hierarchical) section -- matching the
    "weak/inconsistent structure" case from PDF §2.3.
    """
    matches = list(_HEADER_RE.finditer(_mask_fenced_code(markdown_text)))

    if not matches:
        return MarkdownSection(
            level=DOCUMENT_LEVEL,
            title=DEFAULT_ROOT_TITLE,
            content=markdown_text.strip(),
        )

    preamble = markdown_text[: matches[0].start()].strip()
    root_frame = {
        FRAME_KEY_LEVEL: DOCUMENT_LEVEL,
        FRAME_KEY_TITLE: DEFAULT_ROOT_TITLE,
        FRAME_KEY_CONTENT: preamble,
        FRAME_KEY_CHILDREN: (),
    }

    opened_stack = _fold_headings((root_frame,), matches, markdown_text, 0)
    (closed_root_frame,) = _close_to_root(opened_stack)

    root = MarkdownSection(
        level=closed_root_frame[FRAME_KEY_LEVEL],
        title=closed_root_frame[FRAME_KEY_TITLE],
        content=closed_root_frame[FRAME_KEY_CONTENT],
        children=list(closed_root_frame[FRAME_KEY_CHILDREN]),
    )
    return _collapse_lone_h1_title(root)


def split_paragraphs(text: str) -> list[str]:
    """
    Split `text` at blank-line boundaries into non-empty paragraphs.

    Used by the flat plain-text pipeline in ``chunk_builder.chunk_plain_text``
    to give the semantic-grouping algorithm paragraph-level granularity
    even when the input has no ATX heading structure.

    Returns an empty list when `text` is blank, and a single-element list
    when there are no blank-line separators.
    """
    return [p.strip() for p in _PARAGRAPH_BOUNDARY_RE.split(text) if p.strip()]
