# hybrid-graph-rag-chunker

A focused, **chunking-only** implementation of the Markdown ingestion stage from
*Hybrid Hierarchical + Graph-Augmented RAG — Apache Solr Implementation* (v3.0).

It turns a Markdown (or plain-text) document into a recursive tree of typed nodes
whose flattened, JSON-serialized shape matches the Solr schema defined in **§4**
of that document — ready to hand off to an indexing step.

> This project does **not** talk to Solr, extract cross-references, or do
> retrieval/querying. It only does one thing: turn text into schema-shaped,
> embedded chunks.

---

## Project layout

```
hybrid-graph-rag-chunker/
├── main.py                          # console entrypoint & demo documents
├── pyproject.toml
├── Makefile
├── src/hybrid_graph_rag_chunker/
│   ├── __init__.py                  # public API re-exports
│   ├── constants.py                 # every literal used across the pipeline
│   ├── models.py                    # ChunkNode dataclass + NodeType / DocumentStyle enums
│   ├── tail_call.py                 # manual TCO (TailCall / AsyncTailCall loops)
│   ├── ai_client.py                 # EmbeddingClient + LLMClient (LM Studio / OpenAI-compat)
│   ├── chunk_strategy.py            # BaseChunker ABC + TextChunk output contract
│   ├── fixed_size_chunker.py        # sliding-window, embedding-free strategy
│   ├── semantic_chunker.py          # embedding-based, topic-boundary strategy
│   ├── markdown_parser.py           # ATX-heading parser → MarkdownSection tree
│   └── chunk_builder.py             # assembles everything → ChunkNode tree
└── tests/test_chunking.py
```

---

## Two pipeline paths

### Path A — Structured Markdown (`chunk_markdown`)

```
chunk_markdown(markdown_text, doc_title, organization, doc_id, embedder, strategy)
│
├─ markdown_parser.parse_markdown_outline(markdown_text)
│     Masks fenced code blocks so '#' inside them is ignored.
│     Folds ATX headings left-to-right via a tail-recursive optimized stack.
│     Collapses a lone H1 title into the synthetic level-0 root.
│     Returns MarkdownSection tree.
│
└─ chunk_builder._build_node(root_section, embedder, doc_id, org, doc_slug, strategy)
      │
      └─ _build_step_tree(...)   [AsyncTailCall TCO loop, O(1) stack depth]
            For each MarkdownSection (bottom-up, children before parent):
            │
            ├─ _build_chunks(section, embedder, doc_id, section_path,
            │                parent_id, level, strategy, document_style)
            │     strategy.chunk(section.content, embedder) → list[TextChunk]
            │     embedder.embed([chunk.text, ...])          → list[list[float]]
            │     Returns list[ChunkNode(node_type=CHUNK)]
            │
            └─ ChunkNode(node_type=DOCUMENT | SECTION, children=[chunk_children + section_children])
```

**Output:** a single `ChunkNode(node_type=DOCUMENT)` root whose `.children`
recursively contain `SECTION` and `CHUNK` nodes with `document_style="structured"`.

---

### Path B — Flat Plain Text (`chunk_plain_text`)

```
chunk_plain_text(text, doc_title, organization, doc_id, embedder, strategy)
│
├─ markdown_parser.split_paragraphs(text)     (if strategy is None)
│     Splits at blank-line boundaries → list[str]
│     Chooses split_fn: split_paragraphs (≥2 paras) or segment_sentences (≤1)
│     Instantiates SemanticChunker(split_fn=...)
│
└─ _build_chunks(flat_section, embedder, doc_id, doc_slug,
                 node_id, DOCUMENT_LEVEL+1, strategy,
                 document_style=DocumentStyle.FLAT)
      Reuses the same shared helper as Path A.
      Returns list[ChunkNode(node_type=CHUNK, document_style=FLAT)]
│
└─ ChunkNode(node_type=DOCUMENT, document_style=FLAT,
             children=[chunk_children])
```

**Output:** a `ChunkNode(node_type=DOCUMENT)` root with only `CHUNK` direct
children (no intermediate `SECTION` nodes) and `document_style="flat"` on every
node.

---

## Call graph (module level)

```
main.py
 ├── chunk_builder.chunk_markdown / chunk_plain_text
 │    ├── markdown_parser.parse_markdown_outline
 │    │    ├── tail_call.tail_call_optimized   (tail-call driver)
 │    │    │    └── _close_while, _close_to_root, _fold_headings
 │    │    └── returns MarkdownSection tree
 │    │
 │    ├── chunk_builder._build_node
 │    │    └── tail_call.async_tail_call_optimized(_build_step)
 │    │         └── chunk_builder._build_chunks   ← shared by both paths
 │    │              ├── BaseChunker.chunk(text, embedder)
 │    │              │    ├── SemanticChunker.chunk
 │    │              │    │    ├── segment_sentences / split_paragraphs (split_fn)
 │    │              │    │    ├── ai_client.EmbeddingClient.embed   (sentence embeddings)
 │    │              │    │    ├── semantic_chunker.detect_boundaries
 │    │              │    │    ├── semantic_chunker.form_groups
 │    │              │    │    ├── semantic_chunker.enforce_max_chunks  [TCO]
 │    │              │    │    └── semantic_chunker.validate_token_budgets
 │    │              │    └── FixedSizeChunker.chunk (no embedding calls)
 │    │              │
 │    │              └── ai_client.EmbeddingClient.embed  (chunk-level embeddings)
 │    │
 │    └── models.ChunkNode tree
 │         ├── .flatten() → list[dict]   [TCO via tail_call.depth_first_search]
 │         ├── .walk()    → Iterator[ChunkNode]  [TCO]
 │         └── .count_by_type() → dict[str, int]
 │
 └── ai_client.LLMClient.generate   (optional LLM test call)
```

---

## Pluggable chunking strategy

Both `chunk_markdown` and `chunk_plain_text` accept an optional `strategy`
parameter of type `BaseChunker`:

```python
from hybrid_graph_rag_chunker.chunk_strategy import BaseChunker, TextChunk
from hybrid_graph_rag_chunker.ai_client import EmbeddingClient

class BaseChunker(ABC):
    @abstractmethod
    async def chunk(self, text: str, embedder: EmbeddingClient) -> list[TextChunk]:
        ...
```

### Built-in strategies

| Strategy | Module | Embedding calls | Notes |
|---|---|---|---|
| `SemanticChunker` | `semantic_chunker.py` | Yes (sentence + chunk level) | Default for both paths |
| `FixedSizeChunker` | `fixed_size_chunker.py` | No | Sliding word-window; fast baseline |

### `TextChunk` — the strategy output contract

```python
@dataclass(frozen=True)
class TextChunk:
    text: str               # raw chunk text, ready to embed and store
    coherence_score: float | None  # mean pairwise cosine sim (SemanticChunker)
                                   # or None (FixedSizeChunker)
```

---

## `SemanticChunker` internal algorithm

```
text
 │
 ├─ 1. split_fn(text)                     → list[str]  (sentences or paragraphs)
 ├─ 2. EmbeddingClient.embed(units)       → list[Vector]
 ├─ 3. detect_boundaries(embeddings, τ)   → list[int]   (cosine sim < τ = boundary)
 ├─ 4. form_groups(units, embeddings, boundaries)  → list[Group]
 ├─ 5. enforce_max_chunks(groups, n)      → list[Group]  [TCO — merges most-similar pair]
 ├─ 6. validate_token_budgets(groups)
 │       ├─ split_oversized_groups(…)     [TCO — bisects at weakest boundary]
 │       └─ reduce(_merge_if_undersized)
 └─ 7. _group_to_chunk(group)             → TextChunk(text, coherence_score)
```

Key constants (`constants.py`):

| Constant | Default | Meaning |
|---|---|---|
| `SIMILARITY_THRESHOLD` | `0.45` | Cosine sim below which a boundary is detected |
| `MAX_CHUNKS_PER_DOC` | `20` | Hard cap on groups after merging |
| `MAX_TOKENS_PER_CHUNK` | `350` | Group is split if it exceeds this |
| `MIN_TOKENS_PER_CHUNK` | `80` | Group is merged into previous if below this |

---

## Data structure — `ChunkNode`

Every node in the tree is a single `ChunkNode` dataclass instance
(`models.py`). The tree is **immutable**: all children are wired via
constructor arguments, never mutated after creation.

```python
@dataclass
class ChunkNode:
    # ── Identity ────────────────────────────────
    node_type:    NodeType          # DOCUMENT | SECTION | CHUNK
    title:        str
    id:           str               # uuid4, unique per node
    doc_id:       str               # shared by every node in one document
    organization: str | None        # set only on the DOCUMENT root

    # ── Graph structure ──────────────────────────
    parent_id:      str | None
    reference_ids:  list[str]       # populated by the indexing stage (out of scope)
    reference_count: int

    # ── Hierarchy ───────────────────────────────
    section_path: str               # e.g. "contract_a/payment_terms/chunk_0"
    level:        int               # 0 = document root, +1 per heading depth

    # ── Content ─────────────────────────────────
    content:      str               # only on CHUNK nodes
    summary:      str | None
    chunk_order:  int | None        # 0-based position among siblings

    # ── Embedding ───────────────────────────────
    embedding:    list[float] | None  # 1024-dim (or model-dim) cosine vector

    # ── Semantic chunking metadata ───────────────
    chunk_group_id:        int | None
    group_label:           str | None
    group_coherence_score: str | None  # formatted "0.00"–"1.00"

    # ── Timestamps ──────────────────────────────
    created_at:    str              # ISO-8601 UTC, e.g. "2025-01-01T00:00:00Z"

    # ── Style flag ──────────────────────────────
    document_style: DocumentStyle   # STRUCTURED | FLAT

    # ── In-memory only (not in Solr schema) ──────
    children: list[ChunkNode]
```

### `NodeType` mapping

| `node_type` | Source | `content` | `embedding` | `children` |
|---|---|---|---|---|
| `DOCUMENT` | Synthetic root (level 0) | ✗ | ✗ | SECTION and/or CHUNK |
| `SECTION` | ATX heading (level 1–6) | ✗ | ✗ | SECTION and/or CHUNK |
| `CHUNK` | Leaf text window | ✓ | ✓ | — |

### `DocumentStyle` flag

| Value | Set when |
|---|---|
| `STRUCTURED` | Input has ATX headings (`chunk_markdown`) |
| `FLAT` | Heading-free plain text (`chunk_plain_text`) |

---

## Solr document shape — `.flatten()`

Calling `.flatten()` on the root node runs a tail-call optimized pre-order DFS
and returns a `list[dict]`, one dict per node.  Only non-`None` optional
fields are included.

```jsonc
// DOCUMENT node
{
  "id": "a1b2...",
  "doc_id": "a1b2...",
  "node_type": "document",
  "title": "Service Agreement Contract A",
  "section_path": "service_agreement_contract_a",
  "level": 0,
  "organization": "org-demo-0001",
  "child_ids": ["c3d4...", "e5f6..."],
  "created_at": "2025-06-22T18:00:00Z",
  "document_style": "structured"
}

// SECTION node
{
  "id": "c3d4...",
  "doc_id": "a1b2...",
  "node_type": "section",
  "title": "Payment Terms",
  "section_path": "service_agreement_contract_a/payment_terms",
  "level": 2,
  "parent_id": "a1b2...",
  "child_ids": ["g7h8..."],
  "created_at": "...",
  "document_style": "structured"
}

// CHUNK node
{
  "id": "g7h8...",
  "doc_id": "a1b2...",
  "node_type": "chunk",
  "title": "Payment Terms — Part 1",
  "section_path": "service_agreement_contract_a/payment_terms/chunk_0",
  "level": 3,
  "parent_id": "c3d4...",
  "content": "This section governs all payment obligations...",
  "embedding": [0.012, -0.034, ...],   // 1024-dim
  "chunk_order": 0,
  "chunk_group_id": 0,
  "group_coherence_score": "0.87",
  "created_at": "...",
  "document_style": "structured"
}
```

---

## Tail-call optimization (TCO) layer — [tail_call.py](file:///home/santa/WiseYak/work/wiseai/wiseai-chunking/hybrid-graph-rag-chunker/src/hybrid_graph_rag_chunker/tail_call.py)

CPython does not support native Tail-Call Optimization (TCO), meaning deeply nested documents or large sentence/paragraph counts would trigger a `RecursionError`. To overcome this, this project implements a **tail-call optimization (TCO)** mechanism:

- A **tail-call recursive wrapper** is a programming pattern used to execute recursive functions in $O(1)$ stack space.
- Instead of recursing directly (which pushes a new frame onto the call stack), the function returns a **thunk** (a deferred recursive step wrapped in a `TailCall` or `AsyncTailCall` object).
- The **tail-call driver** (invoked via `@tail_call_optimized` or `@async_tail_call_optimized`) runs a flat `while` loop that repeatedly invokes the next thunk until a final non-thunk result is returned.

```python
# Pattern used throughout the codebase
def _foo_step(state, ...):
    if done:
        return result
    # Return a deferred TailCall thunk instead of recursing directly
    return TailCall(lambda: _foo_step(next_state, ...))

foo = tail_call_optimized(_foo_step)  # Driver unrolls TailCall thunks in a flat while loop
```

| Function | File | TCO type |
|---|---|---|
| `_fold_headings` | `markdown_parser.py` | sync `TailCall` |
| `_close_while`, `_close_to_root` | `markdown_parser.py` | sync `TailCall` |
| `enforce_max_chunks` | `semantic_chunker.py` | sync `TailCall` |
| `_split_oversized` | `semantic_chunker.py` | sync `TailCall` (work-list) |
| `_build_step_tree` | `chunk_builder.py` | async `AsyncTailCall` |
| `flatten`, `walk` | `models.py` | sync `TailCall` via `depth_first_search` |

---

## AI clients — `ai_client.py`

Both clients are thin wrappers over `httpx.AsyncClient` calling an
OpenAI-compatible REST API (LM Studio by default).

```
EmbeddingClient.embed(texts: list[str]) → list[list[float] | None]
  POST /v1/embeddings  { "model": "...", "input": ["..."] }
  Response: { "data": [{ "embedding": [...], "index": 0 }, ...] }
  Returns None per position on network/server error.

LLMClient.generate(prompt, system_prompt) → str | None
  POST /v1/chat/completions  { "model": "...", "messages": [...] }
  Response: { "choices": [{ "message": { "content": "..." } }] }
  Returns None on error.
```

Default endpoints (overridable via constructor):

| Client | Default URL | Constant |
|---|---|---|
| `EmbeddingClient` | `http://localhost:1234/v1/embeddings` | `DEFAULT_EMBEDDINGS_URL` |
| `LLMClient` | `http://localhost:1234/v1/chat/completions` | `DEFAULT_CHAT_URL` |

> [!NOTE]
> **Embedding Dimension Note:** The locally hosted embedding model in LM Studio (e.g. `nomic-embed-text`) yields **768-dimensional** embeddings by default. However, the production Solr database schema defined in the RAG specification expects **1024-dimensional** dense vectors. When preparing for production deployment, ensure the model dimension matches the dimension defined in the Solr schema.

---

## Running

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
# Install deps + run the demo pipeline
make start

# Run only the sync/TCO tests (no LM Studio required)
uv run python -m pytest tests/test_chunking.py -q \
  -k "not (test_chunk_markdown or test_tco_flatten or test_plain_text or test_tco_plain)"

# Run all tests (requires LM Studio running on localhost:1234)
uv run python -m pytest tests/test_chunking.py -q
```

---

## Using it on your own documents

### Markdown (structured)

```python
import asyncio
from hybrid_graph_rag_chunker import chunk_markdown
from hybrid_graph_rag_chunker.ai_client import EmbeddingClient

async def main():
    root = await chunk_markdown(
        my_markdown_text,
        doc_title="My Document",
        organization="org-1234",
        embedder=EmbeddingClient(),        # default: localhost:1234
    )
    solr_docs = root.flatten()             # list[dict], one per Solr document
    print(root.count_by_type())            # {'document': 1, 'section': N, 'chunk': M}

asyncio.run(main())
```

### Plain text (flat / no headings)

```python
from hybrid_graph_rag_chunker import chunk_plain_text

root = await chunk_plain_text(
    my_prose_text,
    doc_title="Research Notes",
)
# Every node has document_style="flat"
```

### Custom chunking strategy

```python
from hybrid_graph_rag_chunker import chunk_markdown, FixedSizeChunker

root = await chunk_markdown(
    my_markdown_text,
    strategy=FixedSizeChunker(chunk_size=200, overlap=30),
)
```
