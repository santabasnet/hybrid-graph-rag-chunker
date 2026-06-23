"""
Console entrypoint.

Runs the Markdown -> recursive hierarchy -> semantic chunking -> embedding
pipeline against a bundled demo document, prints a tree summary of the
resulting nodes, and writes the flat list of Solr-schema documents (PDF
§4.1) to ``chunked_output.json``.

Also runs the flat plain-text path (``chunk_plain_text``) against a
heading-free prose document, printing a second tree and writing
``chunked_output_flat.json`` (every entry has ``document_style: "flat"``).

Tail-call-optimized tree render (via tail_call.depth_first_search):

    ``_render_tree`` builds the indented tree summary as a flat list
    of strings in pre-order DFS, using the shared ``depth_first_search`` combinator
    so the renderer isn't bounded by Python's call-stack depth.
    Instead of keeping an indent string per ``ChunkNode``, it passes
    indent depth as a parallel value alongside each node via a
    ``(node, depth)`` wrapper so ``depth_first_search``'s single ``transform``
    argument can access it.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass

from hybrid_graph_rag_chunker.chunk_builder import (
    chunk_markdown,
    chunk_plain_text,
)
from hybrid_graph_rag_chunker.constants import (
    DEMO_DOC_TITLE,
    DEMO_ORGANIZATION,
    DEMO_PLAIN_DOC_TITLE,
    EMBEDDING_DIM_SUFFIX,
    FIXED_DEMO_CHUNK_SIZE,
    FIXED_DEMO_OVERLAP,
    JSON_INDENT,
    OUTPUT_FILE_PATH,
    OUTPUT_FILE_PATH_FLAT,
    OUTPUT_FILE_PATH_FIXED,
    PATH_BRACKET_CLOSE,
    PATH_BRACKET_OPEN,
    SEPARATOR_WIDTH,
    TREE_INDENT_UNIT,
)
from hybrid_graph_rag_chunker.models import ChunkNode, DocumentStyle
from hybrid_graph_rag_chunker.ai_client import EmbeddingClient, LLMClient
from hybrid_graph_rag_chunker.fixed_size_chunker import FixedSizeChunker
from hybrid_graph_rag_chunker.tail_call import depth_first_search

DEMO_MARKDOWN = """\
# Service Agreement Contract A

This Service Agreement ("Agreement") is entered into between the Buyer and the \
Seller for the provision of professional services as described in the attached \
statement of work. By signing below, both parties agree to be bound by the terms \
set out in this document.

## Definitions

"Buyer" means the party purchasing services under this Agreement. "Seller" means \
the party providing services under this Agreement. "Confidential Information" \
means any non-public business, technical, or financial information disclosed by \
either party. All capitalized terms used herein shall have the meanings ascribed \
to them in this section unless otherwise defined elsewhere in this Agreement.

## Payment Terms

This section governs all payment obligations between Buyer and Seller, including \
remittance timing, accepted payment methods, and consequences of late payment.

### Remittance Window

The Buyer shall remit payment within 30 days of invoice receipt. Late payments \
incur a 1.5% monthly penalty on the outstanding balance. All payments must be made \
in the currency specified on the original purchase order. Disputed invoices must \
be flagged in writing within 10 business days of receipt, or they are deemed \
accepted.

### Currency and Method

Payments shall be made via bank transfer to the account designated by the Seller \
in writing. The Seller may update its banking details with 10 days advance notice \
to the Buyer. Wire transfer fees are borne by the Buyer. Payments in any currency \
other than the one specified on the purchase order require prior written consent \
from both parties.

## Liability

Neither party shall be liable to the other for indirect, incidental, special, or \
consequential damages arising out of or relating to this Agreement, regardless of \
the legal theory under which such damages are sought. Total liability under this \
Agreement shall not exceed the total fees paid by the Buyer in the twelve months \
preceding the event giving rise to the claim. Nothing in this section limits \
liability for gross negligence, willful misconduct, or breach of confidentiality \
obligations.

## Termination Conditions

Either party may terminate this Agreement upon 60 days written notice to the other \
party. Upon termination, all outstanding invoices become immediately due and \
payable in full. The confidentiality obligations set out in this Agreement survive \
termination indefinitely. Either party may also terminate immediately for material \
breach that remains uncured 15 days after written notice of the breach.

## Large Section Test

This section is intentionally long to test the chunking algorithm's ability to handle \
large continuous blocks of text. The chunker should split this into multiple chunks. \
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam in dui mauris. \
Vivamus hendrerit arcu sed erat molestie vehicula. Sed auctor neque eu tellus \
rhoncus ut eleifend nibh porttitor. Ut in nulla enim. Phasellus molestie magna \
non est bibendum non venenatis nisl tempor. Suspendisse dictum feugiat nisl ut \
dapibus. Mauris iaculis porttitor posuere. Praesent id metus massa, ut blandit \
odio. Proin quis tortor orci. Etiam at risus et justo dignissim congue. Donec \
congue lacinia dui, a porttitor lectus condimentum laoreet. Nunc eu ullamcorper \
orci. Quisque eget odio ac lectus vestibulum faucibus eget in metus. In \
pellentesque faucibus vestibulum. Nulla at nulla justo, eget luctus tortor. \
Nulla facilisi. Duis aliquet egestas purus in blandit. Curabitur vulputate, ligula \
lacinia scelerisque tempor, lacus lacus ornare ante, ac egestas est urna sit amet \
arcu. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per \
inceptos himenaeos. Sed molestie augue sit amet leo consequat posuere. Vestibulum \
ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; \
Proin vel ante a orci tempus eleifend ut et magna. Lorem ipsum dolor sit amet, \
consectetur adipiscing elit. Vivamus luctus urna sed urna ultricies ac tempor dui \
sagittis. In condimentum facilisis porta. Sed nec diam eu diam mattis viverra. \
Nulla fringilla, orci ac euismod semper, magna diam porttitor mauris, quis \
sollicitudin sapien justo in libero. Vestibulum mollis mauris enim. Morbi euismod \
magna ac lorem rutrum elementum. Donec viverra auctor lobortis. Pellentesque \
habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. \
Integer congue consequat lectus, pellentesque aliquet urna gravida pellentesque. \
Nunc ut convallis purus. Pellentesque habitant morbi tristique senectus et netus \
et malesuada fames ac turpis egestas. Mauris tempor elit sed leo rhoncus porta. \
Nulla facilisi. Integer vel nisl nec nisi commodo ullamcorper vel ut felis. \
Aliquam in interdum dolor. Aenean in odio ut nisl consequat ullamcorper vel sit \
amet ligula. Ut tristique, nisl et iaculis viverra, dui felis varius nulla, non \
fringilla massa nisl sit amet sapien. Praesent volutpat, quam quis porta elementum, \
urna sem dapibus diam, non ultrices nunc est id tellus. In tincidunt, eros nec \
dapibus cursus, neque arcu rhoncus lorem, ac tristique mi odio at urna. Nam \
aliquet, ante sed tincidunt dapibus, libero ipsum accumsan felis, eget \
ullamcorper sem sapien sed risus. Aenean cursus metus sed diam ultrices tristique. \
Sed varius vulputate eros ut euismod. Phasellus accumsan lacinia nisl, id cursus \
augue egestas at. Ut feugiat vehicula tellus, at fermentum quam feugiat in. \
Pellentesque mattis magna id mi dignissim porta. Proin vel congue neque. Praesent \
sodales pretium sem. Morbi blandit orci eu mi facilisis iaculis. Quisque tempus \
turpis in ligula tincidunt aliquet. Maecenas tristique orci ac mauris bibendum \
sollicitudin. Nunc et dui ante. Donec scelerisque odio sem. Ut tempus, neque non \
elementum eleifend, tellus metus scelerisque mauris, eget gravida nibh nisi vitae \
massa. Nullam sagittis egestas eros. Fusce facilisis urna sed magna lacinia sed \
facilisis metus consequat. Suspendisse potenti. Nam non erat eros. Sed mattis \
lacinia imperdiet. Nulla in arcu turpis, et rhoncus ante. Donec hendrerit ipsum \
vel leo tincidunt imperdiet. Phasellus feugiat erat vel lacus vehicula \
pellentesque. Nunc porta suscipit nunc, sit amet mattis ligula fermentum vel. \
Aenean ut nisi non nisi viverra feugiat id sed est. Aliquam sem purus, faucibus \
sit amet sodales id, varius sed sem. Integer ut tristique lorem. Suspendisse \
ultrices nisi ex. Nulla eget congue sapien. Cras in turpis elit. Nulla venenatis, \
tortor sed dignissim imperdiet, nisi nunc facilisis libero, sed iaculis ligula \
velit a quam. Mauris condimentum, lectus ac consequat ultrices, felis est varius \
est, eu fermentum dui leo vitae sapien. Fusce commodo diam a leo suscipit dictum. \
Donec condimentum, ante a lacinia mollis, arcu lorem vulputate lectus, et \
sollicitudin lacus risus et nulla.
"""

DEMO_PLAIN_TEXT = """\
Artificial intelligence has transformed the way we interact with technology. \
From voice assistants to recommendation systems, AI-powered tools are now \
embedded in everyday life. The pace of progress shows no sign of slowing down, \
and researchers continue to push the boundaries of what machines can understand \
and generate.

Large language models represent one of the most significant recent advances in \
AI. These models are trained on vast corpora of text and learn to predict the \
next token in a sequence. Through this seemingly simple objective, they acquire \
surprisingly broad world knowledge and can perform complex reasoning tasks when \
prompted appropriately.

Retrieval-augmented generation combines the strengths of parametric and \
non-parametric memory. A retrieval component fetches relevant passages from an \
external corpus, and a generative model conditions its output on both the query \
and the retrieved context. This approach reduces hallucination and allows the \
system to cite its sources.

Semantic chunking is a critical preprocessing step in retrieval-augmented \
pipelines. Splitting documents at topic boundaries rather than arbitrary token \
counts ensures that each chunk is topically coherent and that related information \
is not split across chunk boundaries. This directly improves retrieval precision.

Graph-augmented retrieval extends the flat vector search paradigm by encoding \
explicit relationships between chunks. Cross-references, citations, and \
hierarchical containment relationships can all be captured in the graph, allowing \
retrieval algorithms to hop between nodes and assemble multi-faceted answers that \
would be impossible to retrieve from a single chunk.
"""


@dataclass(frozen=True)
class _DepthNode:
    """Wraps a ChunkNode with its current indentation depth so the
    ``depth_first_search`` transform can produce a correctly indented line per node
    without external mutable state."""

    node: ChunkNode
    depth: int


def _depth_children(dn: _DepthNode) -> list[_DepthNode]:
    return [
        _DepthNode(child_node, dn.depth + 1) for child_node in dn.node.children
    ]


def _render_line(dn: _DepthNode) -> str:
    line = (
        f"{TREE_INDENT_UNIT * dn.depth}"
        f"- ({dn.node.node_type.value}) {dn.node.title}  "
        f"{PATH_BRACKET_OPEN}{dn.node.section_path}{PATH_BRACKET_CLOSE}"
    )
    if dn.node.embedding:
        line += f" [{len(dn.node.embedding)}{EMBEDDING_DIM_SUFFIX}]"
    return line


_render_tree_dfs = depth_first_search(_depth_children, _render_line)


def _render_tree(root: ChunkNode) -> str:
    lines = _render_tree_dfs([_DepthNode(root, 0)], [])
    return "\n".join(lines)


def _write_json(path: str, data: object) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=JSON_INDENT)


async def main() -> int:
    embedder = EmbeddingClient()
    llm = LLMClient()

    print("=" * SEPARATOR_WIDTH)
    print("Hybrid Hierarchical + Graph-Augmented RAG — Markdown Chunker")
    print("=" * SEPARATOR_WIDTH)
    print(f"Embedding endpoint : {embedder.base_url}")
    print(f"Embedding model    : {embedder.model}")
    print(f"Embedding dim      : {embedder.dim}")
    print(f"LLM endpoint       : {llm.base_url}")
    print(f"LLM model          : {llm.model}")
    print()

    # ── LLM Test Call ────────────────────────────────────────────────────
    print("Testing LLM generation...")
    # NOTE: LM Studio must be running for this to work, and if it's not, it'll return None.
    # We will print the result regardless.
    llm_response = await llm.generate(
        "Hello, are you there?", system_prompt="You are a helpful assistant."
    )
    print(f"LLM Response       : {llm_response}")
    print()

    # ── Structured Markdown demo ─────────────────────────────────────────
    root = await chunk_markdown(
        DEMO_MARKDOWN,
        doc_title=DEMO_DOC_TITLE,
        organization=DEMO_ORGANIZATION,
        embedder=embedder,
    )

    print("Recursive chunk hierarchy (Markdown):\n")
    print(_render_tree(root))

    solr_docs = root.flatten()

    print()
    print(f"Node counts by type : {root.count_by_type()}")
    print(f"Total Solr documents: {len(solr_docs)}")

    _write_json(OUTPUT_FILE_PATH, solr_docs)

    print(f"\nFull Solr-document payload written to ./{OUTPUT_FILE_PATH}")
    print("(Each entry matches the field names/types from PDF schema §4.1 —")
    print(" ready to POST to /solr/docs/update.)")

    # ── Flat plain-text demo ─────────────────────────────────────────────
    print()
    print("-" * SEPARATOR_WIDTH)
    print("Flat Plain-Text Demo (heading-free prose → document_style: 'flat')")
    print("-" * SEPARATOR_WIDTH)

    flat_root = await chunk_plain_text(
        DEMO_PLAIN_TEXT,
        doc_title=DEMO_PLAIN_DOC_TITLE,
        organization=DEMO_ORGANIZATION,
        embedder=embedder,
    )

    print("\nFlat chunk hierarchy:\n")
    print(_render_tree(flat_root))

    flat_solr_docs = flat_root.flatten()

    print()
    print(f"Node counts by type : {flat_root.count_by_type()}")
    print(f"Total Solr documents: {len(flat_solr_docs)}")
    is_flat_count = sum(
        1
        for d in flat_solr_docs
        if d.get("document_style") == DocumentStyle.FLAT.value
    )
    print(
        f"Docs with document_style='flat': {is_flat_count}/{len(flat_solr_docs)}"
    )

    _write_json(OUTPUT_FILE_PATH_FLAT, flat_solr_docs)

    print(f"\nFlat Solr-document payload written to ./{OUTPUT_FILE_PATH_FLAT}")

    # ── Fixed-Size chunker demo (pluggable strategy) ──────────────────────────
    print()
    print("-" * SEPARATOR_WIDTH)
    print("Fixed-Size Chunker Demo (pluggable strategy, no embeddings needed)")
    print("-" * SEPARATOR_WIDTH)
    print(
        f"Strategy : FixedSizeChunker(chunk_size={FIXED_DEMO_CHUNK_SIZE}, overlap={FIXED_DEMO_OVERLAP})"
    )

    fixed_strategy = FixedSizeChunker(
        chunk_size=FIXED_DEMO_CHUNK_SIZE, overlap=FIXED_DEMO_OVERLAP
    )
    fixed_root = await chunk_markdown(
        DEMO_MARKDOWN,
        doc_title=DEMO_DOC_TITLE,
        organization=DEMO_ORGANIZATION,
        embedder=embedder,
        strategy=fixed_strategy,
    )

    print("\nChunk hierarchy (fixed-size strategy):\n")
    print(_render_tree(fixed_root))

    fixed_solr_docs = fixed_root.flatten()

    print()
    print(f"Node counts by type : {fixed_root.count_by_type()}")
    print(f"Total Solr documents: {len(fixed_solr_docs)}")

    _write_json(OUTPUT_FILE_PATH_FIXED, fixed_solr_docs)

    print(f"\nFixed-size payload written to ./{OUTPUT_FILE_PATH_FIXED}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
