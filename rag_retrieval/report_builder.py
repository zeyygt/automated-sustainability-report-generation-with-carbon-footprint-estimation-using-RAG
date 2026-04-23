"""Build generation-ready report inputs from a retrieval session."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from .pipeline import handle_query
from .report_models import ReportInput
from .session import RetrievalSession


DEFAULT_REPORT_QUERIES = (
    "district-level natural gas consumption and emissions trend",
    "highest emission districts and sustainability priorities",
    "carbon footprint and per capita emission context",
    "energy consumption reduction opportunities",
)


def build_report_input(
    session: RetrievalSession,
    queries: Iterable[str] | None = None,
    title: str = "Sustainability Analysis Report",
    language: str = "English",
) -> ReportInput:
    query_results = [handle_query(query, session) for query in (tuple(queries) if queries else DEFAULT_REPORT_QUERIES)]
    structured_results = _unique_by_key(
        [
            *_flatten_unique(query_results, "structured_results", _structured_key),
            *_all_structured_results(session),
        ],
        _structured_key,
    )
    retrieval_context = _flatten_unique(query_results, "retrieval_context", _retrieval_key)
    sources = _unique_by_key(
        [
            *_flatten_unique(query_results, "sources", _source_key),
            *_structured_sources(structured_results),
        ],
        _source_key,
    )
    warnings = _unique(
        warning
        for warning in [
            *(
                warning
                for result in query_results
                for warning in result.get("warnings", [])
            ),
            *_structured_warnings(structured_results),
        ]
        if _report_warning(warning)
    )

    return ReportInput(
        title=title,
        language=language,
        session_id=session.session_id,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        documents=_document_summaries(session),
        query_results=query_results,
        structured_results=structured_results,
        retrieval_context=retrieval_context,
        sources=sources,
        warnings=warnings,
    )


def _document_summaries(session: RetrievalSession) -> list[dict]:
    summaries = []
    for doc_id, document in session.documents.items():
        table_dataframe = session.table_dataframes.get(doc_id)
        fact_dataframe = session.fact_dataframes.get(doc_id)
        engine = session.data_engines.get(doc_id)
        summaries.append(
            {
                "doc_id": doc_id,
                "filename": document.filename,
                "parser": document.metadata.get("parser"),
                "element_count": len(document.elements),
                "chunk_count": sum(1 for chunk in session.chunks.values() if chunk.doc_id == doc_id),
                "table_rows": _row_count(table_dataframe),
                "fact_rows": _row_count(fact_dataframe),
                "has_data_engine": bool(engine),
                "detected_columns": _detected_columns(engine),
            }
        )
    return summaries


def _detected_columns(engine) -> dict:
    if not engine:
        return {}
    return {
        "district": engine.district_column,
        "value": engine.value_column,
        "electricity": engine.electricity_column,
        "natural_gas": engine.gas_column,
        "direct_emissions": engine.emissions_column,
        "metric": engine.metric_column,
        "unit": engine.unit_column,
        "time": engine.time_column,
    }


def _all_structured_results(session: RetrievalSession) -> list[dict]:
    results = []
    for doc_id, engine in session.data_engines.items():
        if not engine:
            continue
        document = session.documents.get(doc_id)
        for district in engine.districts():
            data = engine.analyze_district(district)
            if not data:
                continue
            results.append(
                {
                    "doc_id": doc_id,
                    "filename": document.filename if document else doc_id,
                    "source_type": _source_type(document),
                    "parser": document.metadata.get("parser") if document else None,
                    "data": data,
                }
            )
    return results


def _structured_sources(structured_results: list[dict]) -> list[dict]:
    return [
        {
            "doc_id": item.get("doc_id"),
            "filename": item.get("filename"),
            "source_type": item.get("source_type"),
            "parser": item.get("parser"),
            "usage": ["structured"],
        }
        for item in structured_results
    ]


def _structured_warnings(structured_results: list[dict]) -> list[str]:
    warnings = []
    for item in structured_results:
        filename = item.get("filename") or item.get("doc_id") or "unknown_source"
        for warning in item.get("data", {}).get("warnings", []) or []:
            warnings.append(f"{filename}: {warning}")
    return warnings


def _report_warning(value: str) -> bool:
    return value not in {"district_not_detected_for_structured_analysis", "no_structured_results"}


def _source_type(document) -> str:
    if not document:
        return "unknown"
    if document.metadata.get("parser") == "spreadsheet":
        return "spreadsheet"
    return "pdf"


def _flatten_unique(items: list[dict], field: str, key_fn) -> list[dict]:
    seen = set()
    flattened = []
    for item in items:
        for value in item.get(field, []):
            key = key_fn(value)
            if key in seen:
                continue
            seen.add(key)
            flattened.append(value)
    return flattened


def _unique_by_key(items: list[dict], key_fn) -> list[dict]:
    seen = set()
    unique_items = []
    for item in items:
        key = key_fn(item)
        if key in seen:
            continue
        seen.add(key)
        unique_items.append(item)
    return unique_items


def _structured_key(item: dict) -> tuple:
    data = item.get("data", {})
    return (
        item.get("doc_id"),
        item.get("filename"),
        item.get("source_type"),
        data.get("district"),
        data.get("total_emission"),
        data.get("growth"),
    )


def _retrieval_key(item: dict) -> tuple:
    return (item.get("chunk_id"), item.get("doc_id"), item.get("filename"), item.get("rank"))


def _source_key(item: dict) -> tuple:
    return (item.get("doc_id"), item.get("filename"), item.get("source_type"))


def _row_count(dataframe) -> int:
    if dataframe is None:
        return 0
    return int(getattr(dataframe, "shape", (0, 0))[0])


def _unique(values) -> list:
    seen = set()
    unique_values = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values
