"""
Data structures for the Hybrid Hierarchical + Graph-Augmented RAG schema.

These mirror the Apache Solr schema defined in Section 4 of
"hybrid_hierarchical_graph_rag_solr_implementation.pdf" (v3.0).

Tail-call-optimized operations (via tail_call.depth_first_search):

  ``flatten()`` — pre-order DFS over an arbitrary-depth ChunkNode tree,
  producing one Solr document dict per node. A deeply nested document
  (say, document → many chapter levels → many section levels → many
  chunks) would exhaust Python's call stack with plain recursion.

  ``walk()`` — same traversal as flatten but yields raw ChunkNodes;
  used by ``count_by_type()``.

Both delegate to ``tail_call.depth_first_search``, which reduces the tree to a flat
list in O(1) Python call-stack frames via a work-list tail_call.
"""

from __future__ import annotations

import uuid
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterator, Optional

from .constants import (
    FIELD_CHILD_IDS,
    FIELD_CHUNK_GROUP_ID,
    FIELD_CHUNK_ORDER,
    FIELD_CONTENT,
    FIELD_CREATED_AT,
    FIELD_DOC_ID,
    FIELD_DOCUMENT_STYLE,
    FIELD_EMBEDDING,
    FIELD_GROUP_COHERENCE_SCORE,
    FIELD_GROUP_LABEL,
    FIELD_ID,
    FIELD_LEVEL,
    FIELD_NODE_TYPE,
    FIELD_ORGANIZATION,
    FIELD_PARENT_ID,
    FIELD_REFERENCE_COUNT,
    FIELD_REFERENCE_IDS,
    FIELD_SECTION_PATH,
    FIELD_SUMMARY,
    FIELD_TITLE,
    ISO_TIMESTAMP_FORMAT,
)
from .tail_call import depth_first_search


class NodeType(str, Enum):
    DOCUMENT = "document"
    SECTION = "section"
    CHUNK = "chunk"
    REFERENCE_TARGET = "reference_target"


class DocumentStyle(str, Enum):
    FLAT = "flat"
    STRUCTURED = "structured"


def _new_id() -> str:
    return str(uuid.uuid4())


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime(ISO_TIMESTAMP_FORMAT)


@dataclass
class ChunkNode:
    # --- Identity & multi-tenancy (schema §4.1) ---
    node_type: NodeType
    title: str
    id: str = field(default_factory=_new_id)
    doc_id: str = ""
    organization: Optional[str] = None

    # --- Graph structure ---
    parent_id: Optional[str] = None
    reference_ids: list[str] = field(default_factory=list)
    reference_count: int = 0

    # --- Hierarchy ---
    section_path: str = ""
    level: int = 0

    # --- Content ---
    content: str = ""
    summary: Optional[str] = None
    chunk_order: Optional[int] = None

    # --- Embedding (DenseVectorField, 1024-dim, cosine — schema §4.1) ---
    embedding: Optional[list[float]] = None

    # --- Semantic chunking metadata (§10.6) ---
    chunk_group_id: Optional[int] = None
    group_label: Optional[str] = None
    group_coherence_score: Optional[str] = None

    # --- Timestamps ---
    created_at: str = field(default_factory=_now_iso)

    # --- Document style flag (schema §4.1 `document_style`) ---
    document_style: DocumentStyle = DocumentStyle.STRUCTURED

    # --- Recursive hierarchy (NOT a Solr field) ---
    children: list["ChunkNode"] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------
    # Tree-building helper (immutable)
    # ------------------------------------------------------------------
    def with_child(self, child: "ChunkNode") -> "ChunkNode":
        wired = replace(child, parent_id=self.id, doc_id=self.doc_id)
        return replace(self, children=[*self.children, wired])

    @property
    def child_ids(self) -> list[str]:
        return [child.id for child in self.children]

    # ------------------------------------------------------------------
    # Schema projection
    # ------------------------------------------------------------------
    def to_solr_doc(self) -> dict[str, Any]:
        required: dict[str, Any] = {
            FIELD_ID: self.id,
            FIELD_DOC_ID: self.doc_id,
            FIELD_NODE_TYPE: self.node_type.value,
            FIELD_TITLE: self.title,
            FIELD_SECTION_PATH: self.section_path,
            FIELD_LEVEL: self.level,
            FIELD_CREATED_AT: self.created_at,
            FIELD_DOCUMENT_STYLE: self.document_style.value,
        }
        optional: dict[str, Any] = {
            FIELD_ORGANIZATION: (self.organization or None)
            if self.node_type is NodeType.DOCUMENT
            else None,
            FIELD_PARENT_ID: self.parent_id,
            FIELD_CONTENT: self.content or None,
            FIELD_SUMMARY: self.summary or None,
            FIELD_CHUNK_ORDER: self.chunk_order,
            FIELD_CHILD_IDS: self.child_ids or None,
            FIELD_EMBEDDING: self.embedding,
            FIELD_REFERENCE_IDS: self.reference_ids or None,
            FIELD_REFERENCE_COUNT: self.reference_count
            if self.reference_ids
            else None,
            FIELD_CHUNK_GROUP_ID: self.chunk_group_id,
            FIELD_GROUP_LABEL: self.group_label or None,
            FIELD_GROUP_COHERENCE_SCORE: self.group_coherence_score,
        }
        return {
            **required,
            **{k: v for k, v in optional.items() if v is not None},
        }

    # ------------------------------------------------------------------
    # Tail-call optimized tree traversals (O(1) Python call-stack depth)
    # ------------------------------------------------------------------
    def flatten(self) -> list[dict[str, Any]]:
        """Pre-order DFS over the whole subtree, returning one Solr doc per node."""
        return depth_first_search(
            lambda n: n.children, lambda n: n.to_solr_doc()
        )([self], [])

    def walk(self) -> Iterator["ChunkNode"]:
        """Pre-order DFS iterator over this node and all descendants."""
        return iter(
            depth_first_search(lambda n: n.children, lambda n: n)([self], [])
        )

    def count_by_type(self) -> dict[str, int]:
        return dict(Counter(n.node_type.value for n in self.walk()))
