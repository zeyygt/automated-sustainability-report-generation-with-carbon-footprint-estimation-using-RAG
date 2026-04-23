"""Configuration objects for the retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ParserConfig:
    """Controls the layout-aware parser."""

    enable_ocr_fallback: bool = True
    max_ocr_pages: int = 5
    min_extracted_chars_for_no_ocr: int = 80
    detect_tables: bool = True
    table_strategy: str | None = None


@dataclass(slots=True)
class ChunkingConfig:
    """Controls structure-aware chunk construction."""

    max_tokens: int = 450
    min_tokens: int = 40
    overlap_tokens: int = 40
    max_table_rows_inline: int = 40
    spreadsheet_table_rows_inline: int = 12
    include_section_path: bool = True


@dataclass(slots=True)
class EmbeddingConfig:
    """Controls embedding model selection and batching."""

    provider: str = "auto"
    model_name: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = 64
    normalize: bool = True
    fallback_dimension: int = 384


@dataclass(slots=True)
class RetrievalConfig:
    """Controls candidate retrieval, fusion, reranking, and final context."""

    vector_top_k: int = 40
    keyword_top_k: int = 40
    rerank_top_k: int = 24
    final_top_k: int = 8
    rrf_k: int = 60
    numeric_exact_match_boost: float = 0.08
    table_numeric_boost: float = 0.05
    table_intent_boost: float = 0.18
    source_hint_boost: float = 0.10
    expanded_term_weight: float = 0.35
    max_chunks_per_section: int = 3
