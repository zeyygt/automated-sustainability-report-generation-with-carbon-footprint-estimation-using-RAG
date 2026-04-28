"""RAGAS-based retrieval evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable
import warnings

from .evaluation import RetrievalExample, evaluate_rankings
from .models import Chunk
from .session import RetrievalSession


@dataclass(frozen=True, slots=True)
class ChunkReferenceSpec:
    label: str
    filename: str
    chunk_type: str | None = None
    contains_all: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RagasQueryExample:
    query: str
    reference_labels: tuple[str, ...]


DEFAULT_FILES = ("test1.pdf", "rawtext.pdf")

DEFAULT_REFERENCE_SPECS = (
    ChunkReferenceSpec(
        label="test1_intro",
        filename="test1.pdf",
        chunk_type="text",
        contains_all=("annual natural gas consumption figures", "15 districts in istanbul"),
    ),
    ChunkReferenceSpec(
        label="test1_table",
        filename="test1.pdf",
        chunk_type="table",
        contains_all=("headers: district name", "2024 consumption (m³)"),
    ),
    ChunkReferenceSpec(
        label="rawtext_notes",
        filename="rawtext.pdf",
        chunk_type="text",
        contains_all=("mixed sustainability notes for district energy review",),
    ),
)

DEFAULT_EXAMPLES = (
    RagasQueryExample(
        query="Bakirkoy 2024 natural gas consumption",
        reference_labels=("test1_table", "rawtext_notes"),
    ),
    RagasQueryExample(
        query="Kadikoy electricity consumption in 2024",
        reference_labels=("rawtext_notes",),
    ),
    RagasQueryExample(
        query="Adalar carbon footprint in 2024",
        reference_labels=("rawtext_notes",),
    ),
    RagasQueryExample(
        query="District-based natural gas consumption report 2022 2023 2024",
        reference_labels=("test1_intro", "test1_table"),
    ),
    RagasQueryExample(
        query="Which table presents annual natural gas consumption figures for 15 districts in Istanbul?",
        reference_labels=("test1_intro", "test1_table"),
    ),
    RagasQueryExample(
        query="Besiktas natural gas consumption in 2024",
        reference_labels=("test1_table", "rawtext_notes"),
    ),
    RagasQueryExample(
        query="Bakirkoy natural gas consumption in 2023 written in Turkish",
        reference_labels=("test1_table", "rawtext_notes"),
    ),
    RagasQueryExample(
        query="Annual district gas table",
        reference_labels=("test1_intro", "test1_table"),
    ),
)


def run_ragas_retrieval_evaluation(
    file_paths: Iterable[str | Path] | None = None,
    top_k: int = 3,
) -> dict:
    try:
        os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics._context_precision import IDBasedContextPrecision, NonLLMContextPrecisionWithReference
        from ragas.metrics._context_recall import IDBasedContextRecall, NonLLMContextRecall
    except ImportError as exc:  # pragma: no cover - dependency gated
        raise ImportError(
            "RAGAS evaluation requires optional dependencies. Install with: pip install '.[evaluation]'"
        ) from exc

    session = RetrievalSession()
    paths = _resolve_paths(file_paths)
    stats = session.build_index(paths)
    references = resolve_reference_chunks(session, DEFAULT_REFERENCE_SPECS)

    dataset_rows = []
    predictions: dict[str, list[str]] = {}
    retrieval_examples: list[RetrievalExample] = []

    for example in DEFAULT_EXAMPLES:
        hits = session.search(example.query, top_k=top_k)
        retrieved_chunks = [hit.chunk for hit in hits]
        retrieved_ids = [chunk.chunk_id for chunk in retrieved_chunks]
        reference_chunks = [references[label] for label in example.reference_labels]
        reference_ids = [chunk.chunk_id for chunk in reference_chunks]

        predictions[example.query] = retrieved_ids
        retrieval_examples.append(RetrievalExample(example.query, frozenset(reference_ids)))
        dataset_rows.append(
            {
                "user_input": example.query,
                "retrieved_context_ids": retrieved_ids,
                "reference_context_ids": reference_ids,
                "retrieved_contexts": [_compact_text(chunk.text) for chunk in retrieved_chunks],
                "reference_contexts": [_compact_text(chunk.text) for chunk in reference_chunks],
            }
        )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")
        warnings.filterwarnings("ignore", category=ResourceWarning, module="ragas")
        ragas_result = evaluate(
            Dataset.from_list(dataset_rows),
            metrics=[
                IDBasedContextRecall(),
                IDBasedContextPrecision(),
                NonLLMContextRecall(),
                NonLLMContextPrecisionWithReference(),
            ],
            show_progress=False,
        )
    classical = evaluate_rankings(predictions, retrieval_examples, k_values=(1, 2, 3))
    per_query = ragas_result.to_pandas().to_dict(orient="records")

    return {
        "files": [str(path) for path in paths],
        "build": {
            "elapsed_seconds": round(stats.elapsed_seconds, 4),
            "document_count": stats.document_count,
            "element_count": stats.element_count,
            "chunk_count": stats.chunk_count,
            "embedding_count": stats.embedding_count,
        },
        "reference_chunks": {
            label: {
                "chunk_id": chunk.chunk_id,
                "filename": chunk.metadata.get("filename"),
                "chunk_type": chunk.chunk_type.value,
                "text_preview": _compact_text(chunk.text, limit=220),
            }
            for label, chunk in references.items()
        },
        "ragas_metrics": {key: float(value) for key, value in ragas_result._repr_dict.items()},
        "classical_metrics": {key: float(value) for key, value in classical.items()},
        "per_query": per_query,
    }


def resolve_reference_chunks(
    session: RetrievalSession,
    specs: Iterable[ChunkReferenceSpec],
) -> dict[str, Chunk]:
    resolved: dict[str, Chunk] = {}
    for spec in specs:
        matches = [
            chunk
            for chunk in session.chunks.values()
            if _matches_reference_spec(chunk, spec)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one chunk for reference '{spec.label}', found {len(matches)}."
            )
        resolved[spec.label] = matches[0]
    return resolved


def _matches_reference_spec(chunk: Chunk, spec: ChunkReferenceSpec) -> bool:
    filename = str(chunk.metadata.get("filename", ""))
    if filename != spec.filename:
        return False
    if spec.chunk_type and chunk.chunk_type.value != spec.chunk_type:
        return False
    normalized = _compact_text(chunk.text).lower()
    return all(term in normalized for term in spec.contains_all)


def _resolve_paths(file_paths: Iterable[str | Path] | None) -> list[Path]:
    candidates = list(file_paths) if file_paths is not None else [Path(name) for name in DEFAULT_FILES]
    paths = [Path(value) for value in candidates]
    existing = [path for path in paths if path.exists()]
    if not existing:
        raise FileNotFoundError("No evaluation files found for RAGAS retrieval evaluation.")
    return existing


def _compact_text(text: str, limit: int = 1000) -> str:
    return " ".join(str(text).split())[:limit]
