"""In-memory dense and keyword indexes for temporary session retrieval."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable

from .models import Chunk
from .text import search_tokens


class InMemoryVectorIndex:
    """Simple cosine-search vector index for per-session corpora."""

    def __init__(self) -> None:
        self.chunks: dict[str, Chunk] = {}
        self.vectors: dict[str, list[float]] = {}

    def add(self, chunks: Iterable[Chunk], vectors: Iterable[list[float]]) -> None:
        for chunk, vector in zip(chunks, vectors):
            self.chunks[chunk.chunk_id] = chunk
            self.vectors[chunk.chunk_id] = vector

    def search(self, query_vector: list[float], top_k: int) -> list[tuple[str, float]]:
        scored = [
            (chunk_id, _dot(query_vector, vector))
            for chunk_id, vector in self.vectors.items()
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def clear(self) -> None:
        self.chunks.clear()
        self.vectors.clear()


class BM25Index:
    """Small BM25 implementation optimized for short-lived session indexes."""

    def __init__(self, k1: float = 1.4, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_terms: dict[str, Counter[str]] = {}
        self.doc_lengths: dict[str, int] = {}
        self.inverted: dict[str, set[str]] = defaultdict(set)
        self.avgdl = 0.0

    def add(self, chunks: Iterable[Chunk]) -> None:
        for chunk in chunks:
            terms = Counter(_tokens(chunk.text))
            self.doc_terms[chunk.chunk_id] = terms
            self.doc_lengths[chunk.chunk_id] = sum(terms.values())
            for term in terms:
                self.inverted[term].add(chunk.chunk_id)
        self.avgdl = sum(self.doc_lengths.values()) / max(len(self.doc_lengths), 1)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        query_terms = Counter(_tokens(query))
        scores: dict[str, float] = defaultdict(float)
        total_docs = max(len(self.doc_terms), 1)

        for term, query_weight in query_terms.items():
            matching_docs = self.inverted.get(term, set())
            if not matching_docs:
                continue
            df = len(matching_docs)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            for chunk_id in matching_docs:
                freq = self.doc_terms[chunk_id][term]
                doc_len = self.doc_lengths[chunk_id]
                denom = freq + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1e-9))
                scores[chunk_id] += query_weight * idf * (freq * (self.k1 + 1)) / denom

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

    def clear(self) -> None:
        self.doc_terms.clear()
        self.doc_lengths.clear()
        self.inverted.clear()
        self.avgdl = 0.0


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _tokens(text: str) -> list[str]:
    return search_tokens(text)
