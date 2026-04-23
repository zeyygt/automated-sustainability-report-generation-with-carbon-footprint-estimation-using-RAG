"""End-to-end retrieval/data-engine evaluation for local session uploads."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from rag_retrieval import RetrievalSession, handle_query


DEFAULT_FILES = ("test1.pdf", "rawtext.pdf", "testexcel.xlsx")
DEFAULT_DISTRICTS = ("Adalar", "Bakirkoy", "Besiktas", "Kadikoy", "Uskudar")
DEFAULT_QUERIES = (
    "Bakirkoy 2024 natural gas consumption numeric analysis",
    "Kadikoy electricity consumption trend",
    "Besiktas carbon footprint report context",
    "Adalar sustainability context",
)


def main(argv: list[str] | None = None) -> int:
    paths = _resolve_paths(argv or sys.argv[1:])
    if not paths:
        print(json.dumps({"error": "No evaluation files found."}, indent=2))
        return 1

    session = RetrievalSession()
    stats = session.build_index(paths)

    report = {
        "files": [str(path) for path in paths],
        "build": {
            "elapsed_seconds": round(stats.elapsed_seconds, 4),
            "document_count": stats.document_count,
            "element_count": stats.element_count,
            "chunk_count": stats.chunk_count,
            "embedding_count": stats.embedding_count,
        },
        "documents": _document_reports(session),
        "queries": [_query_report(session, query) for query in DEFAULT_QUERIES],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    return 0


def _resolve_paths(args: list[str]) -> list[Path]:
    candidates = args or list(DEFAULT_FILES)
    return [path for value in candidates if (path := Path(value)).exists()]


def _document_reports(session: RetrievalSession) -> list[dict[str, Any]]:
    reports = []
    for doc_id, document in session.documents.items():
        table_dataframe = session.table_dataframes.get(doc_id)
        fact_dataframe = session.fact_dataframes.get(doc_id)
        engine = session.data_engines.get(doc_id)
        document_report = {
            "doc_id": doc_id,
            "filename": document.filename,
            "parser": document.metadata.get("parser"),
            "element_count": len(document.elements),
            "chunk_count": sum(1 for chunk in session.chunks.values() if chunk.doc_id == doc_id),
            "table_rows": _row_count(table_dataframe),
            "fact_rows": _row_count(fact_dataframe),
            "has_data_engine": bool(engine),
        }
        if engine:
            document_report["detected_columns"] = {
                "district": engine.district_column,
                "value": engine.value_column,
                "electricity": engine.electricity_column,
                "natural_gas": engine.gas_column,
                "direct_emissions": engine.emissions_column,
                "metric": engine.metric_column,
                "unit": engine.unit_column,
                "time": engine.time_column,
            }
            document_report["sample_results"] = _sample_results(engine)
        reports.append(document_report)
    return reports


def _query_report(session: RetrievalSession, query: str) -> dict[str, Any]:
    result = handle_query(query, session)
    return {
        "query": query,
        "route": result["route"],
        "retrieval_count": len(result["retrieval_context"]),
        "structured_count": len(result["structured_results"]),
        "sources": result["sources"],
        "warnings": result["warnings"],
        "retrieval_context": [
            {
                "rank": item.get("rank"),
                "filename": item.get("filename"),
                "source_type": item.get("source_type"),
                "score": item.get("score"),
                "text_preview": _preview(item.get("text", "")),
            }
            for item in result["retrieval_context"]
        ],
        "structured_results": result["structured_results"],
    }


def _sample_results(engine) -> list[dict[str, Any]]:
    results = []
    for district in DEFAULT_DISTRICTS:
        result = engine.analyze_district(district)
        if result:
            results.append(result)
    return results


def _row_count(dataframe) -> int:
    if dataframe is None:
        return 0
    return int(getattr(dataframe, "shape", (0, 0))[0])


def _preview(text: str, limit: int = 240) -> str:
    normalized = " ".join(str(text).split())
    return normalized[:limit]


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
