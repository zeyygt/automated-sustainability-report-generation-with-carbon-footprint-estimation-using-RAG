"""Core data contracts shared by ingestion, parsing, chunking, and retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ElementType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    IMAGE = "image"


class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    MIXED = "mixed"


@dataclass(frozen=True, slots=True)
class BoundingBox:
    page: int
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True, slots=True)
class DocumentInput:
    doc_id: str
    path: Path
    filename: str
    mime_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ParsedElement:
    element_id: str
    doc_id: str
    page: int
    element_type: ElementType
    text: str
    section_path: tuple[str, ...] = ()
    bbox: BoundingBox | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    doc_id: str
    filename: str
    elements: list[ParsedElement]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    element_ids: tuple[str, ...]
    section_path: tuple[str, ...]
    page_start: int
    page_end: int
    chunk_type: ChunkType
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Query:
    raw_text: str
    normalized_text: str
    expanded_text: str
    terms: tuple[str, ...]
    expanded_terms: tuple[str, ...]
    phrases: tuple[str, ...]
    numbers: tuple[str, ...]
    years: tuple[str, ...]
    scope_terms: tuple[str, ...]
    intents: tuple[str, ...]
    source_hints: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    chunk: Chunk
    score: float
    vector_score: float = 0.0
    keyword_score: float = 0.0
    rerank_score: float = 0.0
    matched_terms: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BuildStats:
    session_id: str
    document_count: int
    element_count: int
    chunk_count: int
    embedding_count: int
    elapsed_seconds: float
