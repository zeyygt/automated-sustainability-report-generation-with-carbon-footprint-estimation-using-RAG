"""Query execution pipeline that routes between retrieval and data analysis."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from .router import route_query
from .text import normalize_for_search

if TYPE_CHECKING:
    from .models import Query
    from .session import RetrievalSession


_DISTRICT_ALIAS_OVERRIDES = {
    "eyup": "Eyupsultan",
    "eyup sultan": "Eyupsultan",
    "gaziosman pasa": "Gaziosmanpasa",
    "kucuk cekmece": "Kucukcekmece",
    "buyuk cekmece": "Buyukcekmece",
}
KNOWN_DISTRICTS = {}


def handle_query(query_text: str, session: "RetrievalSession") -> dict:
    query = session.retrieval.query_processor.process(query_text)
    route = route_query(query)
    district = extract_district(query)

    if route == "excel":
        structured_results = analyze_data_engines(session, district, query.source_hints)
        return build_query_response(query_text, route, query, [], structured_results)

    if route == "pdf":
        hits = session.search(query_text)
        retrieval_context = format_retrieval_hits(hits)
        return build_query_response(query_text, route, query, retrieval_context, [])

    hits = session.search(query_text)
    retrieval_context = format_retrieval_hits(hits)
    structured_results = analyze_data_engines(session, district, query.source_hints)
    return build_query_response(query_text, route, query, retrieval_context, structured_results)


def extract_district(query: "Query") -> str | None:
    aliases = known_districts()
    terms = tuple(query.terms or ())
    for candidate in _district_candidates(terms):
        if candidate in aliases:
            return aliases[candidate]
    return None


def known_districts() -> dict[str, str]:
    """Return normalized district aliases loaded from reference data."""
    return dict(_known_district_aliases())


def analyze_data_engines(session: "RetrievalSession", district: str | None, source_hints=()) -> list[dict]:
    if not district:
        return []

    source_filter = _source_filter(source_hints)
    results = []
    for doc_id, engine in getattr(session, "data_engines", {}).items():
        if not engine:
            continue

        source = _source_info(session, doc_id)
        if source_filter and source["source_type"] not in source_filter:
            continue

        data = engine.analyze_district(district)
        if not data:
            continue

        results.append(
            {
                "doc_id": doc_id,
                "filename": source["filename"],
                "source_type": source["source_type"],
                "parser": source["parser"],
                "data": data,
            }
        )
    return results


def build_query_response(
    query_text: str,
    route: str,
    query: "Query",
    retrieval_context: list[dict],
    structured_results: list[dict],
) -> dict:
    warnings = _response_warnings(route, query, retrieval_context, structured_results)
    response = {
        "query": query_text,
        "route": route,
        "retrieval_context": retrieval_context,
        "structured_results": structured_results,
        "sources": _collect_sources(retrieval_context, structured_results),
        "warnings": warnings,
    }

    # Backwards-compatible aliases for existing callers while generation code
    # migrates to the standardized contract above.
    response["type"] = route
    response["context"] = retrieval_context
    response["data"] = structured_results
    return response


def format_retrieval_hits(hits) -> list[dict]:
    formatted = []
    for rank, hit in enumerate(hits or [], start=1):
        chunk = getattr(hit, "chunk", None)
        if chunk is None:
            formatted.append({"rank": rank, "text": str(hit)})
            continue

        metadata = dict(getattr(chunk, "metadata", {}) or {})
        formatted.append(
            {
                "rank": rank,
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "filename": metadata.get("filename"),
                "source_type": _source_type_from_metadata(metadata),
                "chunk_type": str(chunk.chunk_type),
                "text": chunk.text,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "section_path": list(chunk.section_path),
                "score": float(getattr(hit, "score", 0.0)),
                "vector_score": float(getattr(hit, "vector_score", 0.0)),
                "keyword_score": float(getattr(hit, "keyword_score", 0.0)),
                "rerank_score": float(getattr(hit, "rerank_score", 0.0)),
                "matched_terms": list(getattr(hit, "matched_terms", ()) or ()),
                "metadata": metadata,
            }
        )
    return formatted


def _source_filter(source_hints) -> set[str] | None:
    hints = set(source_hints or ())
    if "pdf" in hints and "spreadsheet" not in hints:
        return {"pdf"}
    if "spreadsheet" in hints and "pdf" not in hints:
        return {"spreadsheet"}
    return None


def _source_info(session: "RetrievalSession", doc_id: str) -> dict:
    document = getattr(session, "documents", {}).get(doc_id)
    parser = document.metadata.get("parser") if document else None
    if parser == "spreadsheet":
        source_type = "spreadsheet"
    elif parser:
        source_type = "pdf"
    else:
        source_type = "unknown"

    return {
        "filename": document.filename if document else doc_id,
        "parser": parser,
        "source_type": source_type,
    }


def _response_warnings(route: str, query: "Query", retrieval_context: list[dict], structured_results: list[dict]) -> list[str]:
    warnings: list[str] = []
    if route in {"excel", "hybrid"} and not extract_district(query):
        warnings.append("district_not_detected_for_structured_analysis")
    if route in {"excel", "hybrid"} and not structured_results:
        warnings.append("no_structured_results")
    if route in {"pdf", "hybrid"} and not retrieval_context:
        warnings.append("no_retrieval_context")

    for result in structured_results:
        filename = result.get("filename") or result.get("doc_id") or "unknown_source"
        for warning in result.get("data", {}).get("warnings", []) or []:
            warnings.append(f"{filename}: {warning}")
    return warnings


def _collect_sources(retrieval_context: list[dict], structured_results: list[dict]) -> list[dict]:
    sources: dict[tuple, dict] = {}

    for item in retrieval_context:
        key = (item.get("doc_id"), item.get("filename"), item.get("source_type"))
        source = sources.setdefault(
            key,
            {
                "doc_id": item.get("doc_id"),
                "filename": item.get("filename"),
                "source_type": item.get("source_type"),
                "parser": item.get("parser"),
                "usage": set(),
            },
        )
        if not source.get("parser") and item.get("parser"):
            source["parser"] = item.get("parser")
        source["usage"].add("retrieval")

    for item in structured_results:
        key = (item.get("doc_id"), item.get("filename"), item.get("source_type"))
        source = sources.setdefault(
            key,
            {
                "doc_id": item.get("doc_id"),
                "filename": item.get("filename"),
                "source_type": item.get("source_type"),
                "parser": item.get("parser"),
                "usage": set(),
            },
        )
        if not source.get("parser") and item.get("parser"):
            source["parser"] = item.get("parser")
        source["usage"].add("structured")

    return [
        {
            **source,
            "usage": sorted(source["usage"]),
        }
        for source in sources.values()
        if source.get("doc_id") or source.get("filename")
    ]


def _source_type_from_metadata(metadata: dict) -> str:
    if metadata.get("source_type") == "spreadsheet":
        return "spreadsheet"
    filename = str(metadata.get("filename", "")).lower()
    if filename.endswith(".pdf"):
        return "pdf"
    return metadata.get("source_type") or "unknown"


@lru_cache(maxsize=1)
def _known_district_aliases() -> dict[str, str]:
    path = Path(__file__).resolve().parent / "reference_Data.json"
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    aliases: dict[str, str] = {}
    for district in data.get("districts", {}):
        aliases.update(_district_aliases_for_name(str(district)))
    for alias, district in _DISTRICT_ALIAS_OVERRIDES.items():
        aliases[normalize_for_search(alias)] = district
    return aliases


def _district_aliases_for_name(district: str) -> dict[str, str]:
    normalized = normalize_for_search(district)
    compact = normalized.replace(" ", "")
    tokens = tuple(token for token in normalized.split() if token)
    values = {
        normalized: district,
        compact: district,
    }
    if len(tokens) > 1:
        values[" ".join(tokens)] = district
    return values


def _district_candidates(terms: tuple[str, ...]) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        normalized = normalize_for_search(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            candidates.append(normalized)
        compact = normalized.replace(" ", "")
        if compact and compact not in seen:
            seen.add(compact)
            candidates.append(compact)

    for size in range(min(3, len(terms)), 0, -1):
        for index in range(len(terms) - size + 1):
            phrase = " ".join(terms[index : index + size])
            add(phrase)
    return candidates


KNOWN_DISTRICTS = known_districts()
