"""Hybrid retrieval, reranking, and final context selection."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Protocol

from .config import RetrievalConfig
from .embeddings import Embedder
from .index import BM25Index, InMemoryVectorIndex
from .models import Chunk, ChunkType, Query, RetrievalHit
from .text import normalize_for_search, search_tokens


SUSTAINABILITY_EXPANSIONS: tuple[tuple[tuple[str, ...], tuple[str, ...], str], ...] = (
    (
        ("carbon footprint", "greenhouse gas", "ghg", "emissions", "co2", "environmental impact"),
        ("natural gas", "dogalgaz", "energy", "consumption", "fossil fuel", "activity data", "sustainability"),
        "implicit_energy",
    ),
    (
        ("fossil fuel", "fuel usage", "gas usage", "gas consumption"),
        ("natural gas", "dogalgaz", "energy", "consumption", "carbon", "emissions"),
        "implicit_energy",
    ),
    (
        ("energy efficiency", "baseline", "sustainability reporting"),
        ("energy", "consumption", "natural gas", "dogalgaz", "district", "year", "month"),
        "implicit_energy",
    ),
    (
        ("monthly", "month", "ay no", "ay"),
        ("monthly", "month", "ay", "ay no"),
        "monthly",
    ),
    (
        ("annual", "yearly", "year", "yil", "yıl"),
        ("annual", "year", "yil", "yıl"),
        "annual",
    ),
    (
        ("district", "ilce", "ilçe"),
        ("district", "ilce", "ilçe"),
        "district",
    ),
)

ANALYTICAL_TERMS = {
    "compare",
    "comparison",
    "trend",
    "change",
    "changed",
    "increase",
    "increasing",
    "decrease",
    "decreasing",
    "highest",
    "lowest",
    "rank",
    "ranking",
    "unusually",
    "outlier",
    "baseline",
}

GENERIC_QUERY_TERMS = {
    "a",
    "an",
    "and",
    "annual",
    "across",
    "ay",
    "both",
    "consumption",
    "data",
    "district",
    "districts",
    "dogalgaz",
    "excel",
    "figures",
    "for",
    "from",
    "gas",
    "how",
    "in",
    "m3",
    "miktari",
    "month",
    "monthly",
    "natural",
    "no",
    "pdf",
    "report",
    "spreadsheet",
    "table",
    "the",
    "to",
    "tuketim",
    "value",
    "values",
    "which",
    "with",
    "workbook",
    "year",
}


class QueryProcessor:
    """Normalizes queries while preserving exact numeric and scope terms."""

    def process(self, text: str) -> Query:
        normalized = _normalize(text)
        phrases = tuple(normalize_for_search(match.group(1)) for match in re.finditer(r'"([^"]+)"', text))
        terms = tuple(search_tokens(normalized))
        expansion_terms, expansion_intents = self._expand(normalized)
        expanded_terms = tuple(dict.fromkeys([*terms, *expansion_terms]))
        expanded_text = " ".join([normalized, *expansion_terms]).strip()
        numbers = tuple(dict.fromkeys(re.findall(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+(?:[.,]\d+)?\b", normalized)))
        years = tuple(dict.fromkeys(match.group(0) for match in re.finditer(r"\b(?:20\d{2}|19\d{2})\b", normalized)))
        scope_terms = tuple(
            phrase
            for phrase in ("scope 1", "scope 2", "scope 3", "scope1", "scope2", "scope3")
            if phrase in normalized
        )
        intents = set(expansion_intents)
        if numbers:
            intents.add("numeric")
        if any(term in terms for term in ANALYTICAL_TERMS):
            intents.add("analytical")
        if {"table", "row", "rows", "column", "columns"} & set(terms):
            intents.add("table")
        if {"pdf", "report", "annual"} & set(terms):
            intents.add("pdf")
        if {"excel", "spreadsheet", "workbook", "monthly"} & set(terms):
            intents.add("spreadsheet")
        source_hints = self._source_hints(normalized, intents)
        return Query(
            text,
            normalized,
            expanded_text,
            terms,
            expanded_terms,
            phrases,
            numbers,
            years,
            scope_terms,
            tuple(sorted(intents)),
            source_hints,
        )

    def _expand(self, normalized: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        expansion_terms: list[str] = []
        intents: list[str] = []
        for triggers, additions, intent in SUSTAINABILITY_EXPANSIONS:
            if any(trigger in normalized for trigger in triggers):
                expansion_terms.extend(additions)
                intents.append(intent)
        return tuple(dict.fromkeys(search_tokens(" ".join(expansion_terms)))), tuple(dict.fromkeys(intents))

    def _source_hints(self, normalized: str, intents: set[str]) -> tuple[str, ...]:
        hints: list[str] = []
        mentions_annual_series = all(year in normalized for year in ("2022", "2023", "2024"))
        if "pdf" in intents or "annual" in intents or mentions_annual_series or "annual table" in normalized or "report" in normalized:
            hints.append("pdf")
        if "spreadsheet" in intents or "monthly" in intents or "excel" in normalized or "workbook" in normalized:
            hints.append("spreadsheet")
        if "both" in normalized or "across both" in normalized or ("pdf" in normalized and "spreadsheet" in normalized):
            hints.extend(["pdf", "spreadsheet"])
        return tuple(dict.fromkeys(hints))


class Reranker(Protocol):
    def rerank(self, query: Query, candidates: list[RetrievalHit]) -> list[RetrievalHit]:
        """Return candidates sorted by descending relevance."""


class HeuristicReranker:
    """Fast reranker for session workloads and restricted environments."""

    def __init__(self, config: RetrievalConfig | None = None) -> None:
        self.config = config or RetrievalConfig()

    def rerank(self, query: Query, candidates: list[RetrievalHit]) -> list[RetrievalHit]:
        reranked: list[RetrievalHit] = []
        query_terms = set(query.terms)
        expanded_terms = set(query.expanded_terms)
        for hit in candidates:
            text = _normalize(hit.chunk.text)
            chunk_terms = set(search_tokens(text))
            overlap = len(query_terms & chunk_terms) / max(len(query_terms), 1)
            expansion_overlap = len((expanded_terms - query_terms) & chunk_terms) / max(len(expanded_terms), 1)
            phrase_score = sum(0.12 for phrase in query.phrases if phrase in text)
            numeric_score = sum(0.10 for number in query.numbers if number in text)
            scope_score = sum(0.12 for scope in query.scope_terms if scope in text)
            entity_score = self._specific_term_score(query_terms, chunk_terms)
            table_score = self._table_score(query, hit.chunk)
            source_score = self._source_score(query, hit.chunk)
            row_score = self._row_score(query, hit.chunk)
            rerank_score = overlap + (self.config.expanded_term_weight * expansion_overlap) + entity_score + phrase_score + numeric_score + scope_score + table_score + source_score + row_score
            reranked.append(
                RetrievalHit(
                    chunk=hit.chunk,
                    score=hit.score + rerank_score,
                    vector_score=hit.vector_score,
                    keyword_score=hit.keyword_score,
                    rerank_score=rerank_score,
                    matched_terms=tuple(sorted(expanded_terms & chunk_terms)),
                )
            )
        reranked.sort(key=lambda hit: hit.score, reverse=True)
        return reranked

    def _specific_term_score(self, query_terms: set[str], chunk_terms: set[str]) -> float:
        specific_terms = query_terms - GENERIC_QUERY_TERMS
        if not specific_terms:
            return 0.0
        return 0.28 * (len(specific_terms & chunk_terms) / len(specific_terms))

    def _table_score(self, query: Query, chunk: Chunk) -> float:
        if chunk.chunk_type != ChunkType.TABLE:
            return 0.0
        score = 0.0
        if query.numbers:
            score += 0.10
        if {"analytical", "table", "implicit_energy"} & set(query.intents):
            score += self.config.table_intent_boost
        if "monthly" in query.intents and chunk.metadata.get("source_type") == "spreadsheet":
            score += 0.12
        if "annual" in query.intents and chunk.metadata.get("source_type") != "spreadsheet":
            score += 0.10
        return score

    def _source_score(self, query: Query, chunk: Chunk) -> float:
        source_type = chunk.metadata.get("source_type")
        filename = str(chunk.metadata.get("filename", ""))
        score = 0.0
        if "spreadsheet" in query.source_hints and source_type == "spreadsheet":
            score += self.config.source_hint_boost
        if "pdf" in query.source_hints and filename.lower().endswith(".pdf"):
            score += self.config.source_hint_boost
        return score

    def _row_score(self, query: Query, chunk: Chunk) -> float:
        if chunk.chunk_type != ChunkType.TABLE:
            return 0.0
        row_count = chunk.metadata.get("row_count")
        if not isinstance(row_count, int) or row_count <= 0:
            return 0.0
        if query.numbers or {"monthly", "analytical"} & set(query.intents):
            return min(0.08, 0.24 / row_count)
        return 0.0


class RetrievalPipeline:
    """End-to-end retrieval over an already-built session index."""

    def __init__(
        self,
        vector_index: InMemoryVectorIndex,
        keyword_index: BM25Index,
        embedder: Embedder,
        config: RetrievalConfig | None = None,
        reranker: Reranker | None = None,
        query_processor: QueryProcessor | None = None,
    ) -> None:
        self.vector_index = vector_index
        self.keyword_index = keyword_index
        self.embedder = embedder
        self.config = config or RetrievalConfig()
        self.reranker = reranker or HeuristicReranker(self.config)
        self.query_processor = query_processor or QueryProcessor()

    def search(self, query_text: str, top_k: int | None = None) -> list[RetrievalHit]:
        query = self.query_processor.process(query_text)
        query_vector = self.embedder.embed_texts([query.expanded_text])[0]
        vector_hits = self.vector_index.search(query_vector, self.config.vector_top_k)
        keyword_hits = self.keyword_index.search(query.expanded_text, self.config.keyword_top_k)

        fused = self._fuse(query, vector_hits, keyword_hits)
        candidates = sorted(fused.values(), key=lambda hit: hit.score, reverse=True)[: self.config.rerank_top_k]
        reranked = self.reranker.rerank(query, candidates)
        return self._select_context(reranked, top_k or self.config.final_top_k, query)

    def _fuse(
        self,
        query: Query,
        vector_hits: list[tuple[str, float]],
        keyword_hits: list[tuple[str, float]],
    ) -> dict[str, RetrievalHit]:
        scores: dict[str, float] = defaultdict(float)
        vector_scores: dict[str, float] = {}
        keyword_scores: dict[str, float] = {}

        for rank, (chunk_id, score) in enumerate(vector_hits, start=1):
            scores[chunk_id] += 1.0 / (self.config.rrf_k + rank)
            vector_scores[chunk_id] = score
        for rank, (chunk_id, score) in enumerate(keyword_hits, start=1):
            scores[chunk_id] += 1.0 / (self.config.rrf_k + rank)
            keyword_scores[chunk_id] = score

        hits: dict[str, RetrievalHit] = {}
        for chunk_id, score in scores.items():
            chunk = self.vector_index.chunks[chunk_id]
            boosted_score = score + self._metadata_boost(query, chunk)
            hits[chunk_id] = RetrievalHit(
                chunk=chunk,
                score=boosted_score,
                vector_score=vector_scores.get(chunk_id, 0.0),
                keyword_score=keyword_scores.get(chunk_id, 0.0),
            )
        return hits

    def _metadata_boost(self, query: Query, chunk: Chunk) -> float:
        text = _normalize(chunk.text)
        boost = 0.0
        for value in (*query.years, *query.scope_terms):
            if value and value in text:
                boost += self.config.numeric_exact_match_boost
        if query.numbers and chunk.chunk_type == ChunkType.TABLE:
            boost += self.config.table_numeric_boost
        if chunk.chunk_type == ChunkType.TABLE and {"analytical", "table", "implicit_energy"} & set(query.intents):
            boost += self.config.table_intent_boost
        if "spreadsheet" in query.source_hints and chunk.metadata.get("source_type") == "spreadsheet":
            boost += self.config.source_hint_boost
        if "pdf" in query.source_hints and str(chunk.metadata.get("filename", "")).lower().endswith(".pdf"):
            boost += self.config.source_hint_boost
        return boost

    def _select_context(self, hits: list[RetrievalHit], top_k: int, query: Query) -> list[RetrievalHit]:
        selected: list[RetrievalHit] = []
        section_counts: dict[tuple[str, tuple[str, ...]], int] = defaultdict(int)
        for hit in hits:
            key = (hit.chunk.doc_id, hit.chunk.section_path)
            if section_counts[key] >= self.config.max_chunks_per_section:
                continue
            selected.append(hit)
            section_counts[key] += 1
            if len(selected) >= top_k:
                break
        if len(set(query.source_hints)) > 1:
            selected = self._ensure_source_coverage(hits, selected, query.source_hints, top_k)
        return selected

    def _ensure_source_coverage(
        self,
        hits: list[RetrievalHit],
        selected: list[RetrievalHit],
        source_hints: tuple[str, ...],
        top_k: int,
    ) -> list[RetrievalHit]:
        covered = {hint for hint in source_hints if any(self._matches_source_hint(hit, hint) for hit in selected)}
        for source_hint in source_hints:
            if source_hint in covered:
                continue
            candidate = self._first_source_hit(hits, source_hint, selected)
            if not candidate:
                continue
            if len(selected) < top_k:
                selected.append(candidate)
            elif selected:
                selected[-1] = candidate
            covered.add(source_hint)
        return selected

    def _first_source_hit(self, hits: list[RetrievalHit], source_hint: str, selected: list[RetrievalHit]) -> RetrievalHit | None:
        selected_ids = {hit.chunk.chunk_id for hit in selected}
        for hit in hits:
            if hit.chunk.chunk_id in selected_ids:
                continue
            if self._matches_source_hint(hit, source_hint):
                return hit
        return None

    @staticmethod
    def _matches_source_hint(hit: RetrievalHit, source_hint: str) -> bool:
        filename = str(hit.chunk.metadata.get("filename", "")).lower()
        source_type = hit.chunk.metadata.get("source_type")
        if source_hint == "spreadsheet":
            return source_type == "spreadsheet"
        if source_hint == "pdf":
            return filename.endswith(".pdf")
        return False


def _normalize(text: str) -> str:
    return normalize_for_search(text)
