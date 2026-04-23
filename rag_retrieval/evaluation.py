"""Retrieval-only evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True, slots=True)
class RetrievalExample:
    query: str
    relevant_chunk_ids: frozenset[str]


def recall_at_k(ranked_chunk_ids: list[str], relevant_chunk_ids: set[str] | frozenset[str], k: int) -> float:
    if not relevant_chunk_ids:
        return 0.0
    retrieved = set(ranked_chunk_ids[:k])
    return len(retrieved & set(relevant_chunk_ids)) / len(relevant_chunk_ids)


def precision_at_k(ranked_chunk_ids: list[str], relevant_chunk_ids: set[str] | frozenset[str], k: int) -> float:
    if k <= 0:
        return 0.0
    retrieved = ranked_chunk_ids[:k]
    if not retrieved:
        return 0.0
    return len(set(retrieved) & set(relevant_chunk_ids)) / min(k, len(retrieved))


def reciprocal_rank(ranked_chunk_ids: list[str], relevant_chunk_ids: set[str] | frozenset[str]) -> float:
    relevant = set(relevant_chunk_ids)
    for index, chunk_id in enumerate(ranked_chunk_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / index
    return 0.0


def evaluate_rankings(
    predictions: dict[str, list[str]],
    examples: list[RetrievalExample],
    k_values: tuple[int, ...] = (3, 5, 10),
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"recall@{k}"] = mean(
            recall_at_k(predictions.get(example.query, []), example.relevant_chunk_ids, k)
            for example in examples
        )
        metrics[f"precision@{k}"] = mean(
            precision_at_k(predictions.get(example.query, []), example.relevant_chunk_ids, k)
            for example in examples
        )
    metrics["mrr"] = mean(
        reciprocal_rank(predictions.get(example.query, []), example.relevant_chunk_ids)
        for example in examples
    )
    return metrics

