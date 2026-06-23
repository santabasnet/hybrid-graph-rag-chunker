"""
Single source of truth for every literal constant used across the
chunking pipeline -- Solr schema field names (PDF §4.1), hierarchy
levels, semantic-chunking thresholds (PDF §10.3), embedding API
contract keys, and console-app defaults.

Centralizing these avoids "magic" strings/numbers scattered through
the modules and means any literal only ever needs to change in one
place.
"""

from __future__ import annotations

# ── Hierarchy levels (PDF §4.1 `level` field) ──────────────────────────
DOCUMENT_LEVEL = 0
TOP_HEADING_LEVEL = 1

# ── Markdown heading parsing ────────────────────────────────────────────
MAX_HEADING_LEVEL = 6
DEFAULT_ROOT_TITLE = "Document"
CODE_FENCE_DELIMITER = "```"
BLANK_CHAR = " "
NEWLINE = "\n"

# ATX heading regex named-group identifiers
HEADING_GROUP_HASHES = "hashes"
HEADING_GROUP_TITLE = "title"

# Open-section stack frame dict keys
FRAME_KEY_LEVEL = "level"
FRAME_KEY_TITLE = "title"
FRAME_KEY_CONTENT = "content"
FRAME_KEY_CHILDREN = "children"

# ── section_path / slug construction (PDF §13.1) ───────────────────────
PATH_SEPARATOR = "/"
CHUNK_PATH_PREFIX = "chunk_"
PART_TITLE_SEPARATOR = " — Part "
SLUG_SEPARATOR = "_"
DEFAULT_SLUG = "section"
SLUG_PATTERN = r"[^a-z0-9]+"
DEFAULT_UNTITLED_DOCUMENT_TITLE = "Untitled Document"

# ── Solr schema field names (PDF §4.1) ──────────────────────────────────
FIELD_ID = "id"
FIELD_DOC_ID = "doc_id"
FIELD_ORGANIZATION = "organization"
FIELD_NODE_TYPE = "node_type"
FIELD_PARENT_ID = "parent_id"
FIELD_CHILD_IDS = "child_ids"
FIELD_REFERENCE_IDS = "reference_ids"
FIELD_REFERENCE_COUNT = "reference_count"
FIELD_SECTION_PATH = "section_path"
FIELD_LEVEL = "level"
FIELD_TITLE = "title"
FIELD_CONTENT = "content"
FIELD_SUMMARY = "summary"
FIELD_CHUNK_ORDER = "chunk_order"
FIELD_EMBEDDING = "embedding"
FIELD_CREATED_AT = "created_at"
FIELD_CHUNK_GROUP_ID = "chunk_group_id"
FIELD_GROUP_LABEL = "group_label"
FIELD_GROUP_COHERENCE_SCORE = "group_coherence_score"
FIELD_DOCUMENT_STYLE = "document_style"
ISO_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# ── Semantic chunking algorithm (PDF §10.3) ─────────────────────────────
MAX_CHUNKS_PER_DOC = 20  # n — raised from 10; allows more natural topic splits
SIMILARITY_THRESHOLD = 0.45  # tau — boundary detection threshold (unchanged)
MAX_TOKENS_PER_CHUNK = 350  # was 512 — tighter chunks → better precision
MIN_TOKENS_PER_CHUNK = 80  # was 50  — avoid 2-sentence micro-chunks
TOKENS_PER_WORD_RATIO = 0.75
MIN_TOKEN_ESTIMATE = 10  # Fallback for very short string inputs

# ── Fixed Size chunking algorithm ────────────────────────────────
DEFAULT_FIXED_CHUNK_SIZE = 350
DEFAULT_FIXED_CHUNK_OVERLAP = 50

# ── Flat text parsing / grouping ─────────────────────────────────────────
MONOTONOUS_SIMILARITY_THRESHOLD = 0.8
FULL_COHERENCE = 1.0
COSINE_EPSILON = 1e-9
SPACY_LANGUAGE = "en"
SPACY_SENTENCIZER_PIPE = "sentencizer"
SENTENCE_BOUNDARY_PATTERN = r"(?<=[.!?])\s+"

# Semantic group dict keys (used in form_groups / _merge_two / _split_at etc.)
GROUP_KEY_SENTENCES = "sentences"
GROUP_KEY_EMBEDDINGS = "embeddings"
GROUP_KEY_CENTROID = "centroid"

# ── Flat text parsing / grouping ─────────────────────────────────────────
PARAGRAPH_SIMILARITY_THRESHOLD = (
    0.55  # lowered from 0.6; slightly more paragraph grouping
)
PARAGRAPH_BOUNDARY_PATTERN = r"\n\s*\n"

# ── Embedding client (LM Studio, OpenAI-compatible API) ────────────────
DEFAULT_EMBEDDINGS_URL = "http://localhost:1234/v1/embeddings"
DEFAULT_EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"
DEFAULT_EMBEDDING_DIM = 1024
DEFAULT_TIMEOUT_SECONDS = 30.0
API_FIELD_MODEL = "model"
API_FIELD_INPUT = "input"
API_FIELD_DATA = "data"
API_FIELD_EMBEDDING = "embedding"
API_FIELD_INDEX = "index"
DEFAULT_API_INDEX = 0

# ── Chat client (LM Studio, OpenAI-compatible API) ───────────────────────
DEFAULT_CHAT_URL = "http://localhost:1234/v1/chat/completions"
DEFAULT_CHAT_MODEL = "local-model"
API_FIELD_MESSAGES = "messages"
API_FIELD_CHOICES = "choices"
API_FIELD_MESSAGE = "message"
API_FIELD_ROLE = "role"
ROLE_USER = "user"
ROLE_SYSTEM = "system"


# ── Console app / demo ───────────────────────────────────────────────────
DEMO_ORGANIZATION = "org-demo-0001"
DEMO_DOC_TITLE = "Service Agreement Contract A"
DEMO_PLAIN_DOC_TITLE = "AI and Retrieval-Augmented Generation"
OUTPUT_FILE_PATH = "chunked_output.json"
OUTPUT_FILE_PATH_FLAT = "chunked_output_flat.json"
OUTPUT_FILE_PATH_FIXED = "chunked_output_fixed.json"
JSON_INDENT = 2
TREE_INDENT_UNIT = "    "
EMBEDDING_DIM_SUFFIX = "-dim embedding"
PATH_BRACKET_OPEN = "«"
PATH_BRACKET_CLOSE = "»"
SEPARATOR_WIDTH = 78

# ── Fixed-size chunker demo parameters ──────────────────────────────────
FIXED_DEMO_CHUNK_SIZE = 120
FIXED_DEMO_OVERLAP = 20
