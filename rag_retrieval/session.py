"""Session lifecycle API for building and querying temporary indexes."""

from __future__ import annotations

import os
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .chunking import StructureAwareChunker
from .config import ChunkingConfig, EmbeddingConfig, ParserConfig, RetrievalConfig
from .data_engine import DataEngine, _load_reference_data
from .embeddings import Embedder, build_embedder
from .fact_extractor import FactExtractor
from .formula_extractor import FormulaExtractor
from .llm_formula_extractor import LLMFormulaExtractor
from .index import BM25Index, InMemoryVectorIndex
from .ingestion import DocumentIngestor
from .metric_discovery import merge_metric_overrides, normalize_metric_override, summarize_metric_for_api
from .metric_interpreter import suggest_metric_overrides
from .models import BuildStats, Chunk, ParsedDocument, RetrievalHit
from .parsing import DocumentParser, RuntimeDocumentParser
from .retrieval import RetrievalPipeline
from .router import combine_dataframes, parsed_tables_to_dataframe


class RetrievalSession:
    """Owns all temporary data for one live user session."""

    def __init__(
        self,
        session_id: str | None = None,
        parser: DocumentParser | None = None,
        chunker: StructureAwareChunker | None = None,
        embedder: Embedder | None = None,
        parser_config: ParserConfig | None = None,
        chunking_config: ChunkingConfig | None = None,
        embedding_config: EmbeddingConfig | None = None,
        retrieval_config: RetrievalConfig | None = None,
    ) -> None:
        self.session_id = session_id or str(uuid.uuid4())
        self.ingestor = DocumentIngestor()
        self.parser = parser or RuntimeDocumentParser(parser_config)
        self.chunker = chunker or StructureAwareChunker(chunking_config)
        self.embedder = embedder or build_embedder(embedding_config)
        self.vector_index = InMemoryVectorIndex()
        self.keyword_index = BM25Index()
        self.retrieval = RetrievalPipeline(self.vector_index, self.keyword_index, self.embedder, retrieval_config)
        self.documents: dict[str, ParsedDocument] = {}
        self.pdf_documents: dict[str, ParsedDocument] = {}
        self.excel_documents: dict[str, ParsedDocument] = {}
        self.spreadsheet_dataframes: dict[str, object] = {}
        self.table_dataframes: dict[str, object] = {}
        self.fact_dataframes: dict[str, object] = {}
        self.data_engines: dict[str, DataEngine | None] = {}
        self.chunks: dict[str, Chunk] = {}
        self.document_emission_factors: dict[str, dict[str, float]] = {}
        self.document_formula_candidates: dict[str, object] = {}
        self.custom_formula = None  # ExtractedFormula | None
        self.formula_extraction_method: str = "default"  # "direct" | "rag_assisted" | "default"
        self.custom_formula_user_inputs: dict[str, dict[str, object]] = {}
        self.custom_formula_status: str = "default"
        self.custom_formula_missing_variables: list[str] = []
        self.custom_formula_validation_by_document: list[dict[str, object]] = []
        self.factor_override_keys: list[str] = []
        self.formula_input_columns: list[dict[str, object]] = []
        self.formula_conflicts: list[dict[str, object]] = []
        self.factor_conflicts: list[dict[str, object]] = []
        self.methodology_resolution: dict[str, object] = {"formula_doc_id": None, "factor_doc_ids": {}}
        self.methodology_status: str = "clear"
        self.methodology_warnings: list[str] = []
        self.metric_user_overrides: dict[str, dict[str, object]] = {}
        self.metric_system_overrides: dict[str, dict[str, object]] = {}
        self.detected_metrics: list[dict[str, object]] = []
        self.calculation_audit: dict[str, object] = {}
        self.has_structured_data: bool = False
        self.structured_document_count: int = 0
        self.structured_district_count: int = 0
        self.report_generation_status: str = "ready"
        self.report_generation_warnings: list[str] = []

    def build_index(self, file_paths: Iterable[str | Path]) -> BuildStats:
        """Parse, chunk, embed, and index runtime uploads for this session."""

        start = time.perf_counter()
        inputs = self.ingestor.ingest_paths(file_paths)
        parsed_documents = [self.parser.parse(document) for document in inputs]

        excel_documents = [
            document for document in parsed_documents if document.metadata.get("parser") == "spreadsheet"
        ]
        pdf_documents = [
            document for document in parsed_documents if document.metadata.get("parser") != "spreadsheet"
        ]

        chunks = self.chunker.chunk_documents(pdf_documents)
        vectors = self.embedder.embed_texts([chunk.text for chunk in chunks]) if chunks else []

        self.vector_index.add(chunks, vectors)
        self.keyword_index.add(chunks)
        fact_extractor = FactExtractor()
        formula_extractor = FormulaExtractor()
        for document in parsed_documents:
            self.documents[document.doc_id] = document
        for document in pdf_documents:
            self.pdf_documents[document.doc_id] = document
        for document in excel_documents:
            self.excel_documents[document.doc_id] = document

        # Pass 1: parse dataframes and collect emission factors from every document.
        # Factors are gathered session-wide so a formula in a PDF methodology doc
        # applies to consumption DataEngines from separate spreadsheet uploads.
        table_dfs: dict[str, object] = {}
        fact_dfs: dict[str, object] = {}

        for document in parsed_documents:
            try:
                table_dataframe = parsed_tables_to_dataframe(document)
            except ImportError:
                table_dataframe = None

            fact_dataframe = None
            if document.metadata.get("parser") != "spreadsheet":
                try:
                    fact_dataframe = fact_extractor.to_dataframe(document)
                except ImportError:
                    fact_dataframe = None

            table_dfs[document.doc_id] = table_dataframe
            fact_dfs[document.doc_id] = fact_dataframe
            self.table_dataframes[document.doc_id] = table_dataframe
            self.fact_dataframes[document.doc_id] = fact_dataframe
            if document.metadata.get("parser") == "spreadsheet":
                self.spreadsheet_dataframes[document.doc_id] = table_dataframe

            doc_factors = formula_extractor.extract(document, table_dataframe)
            if doc_factors:
                self.document_emission_factors[document.doc_id] = doc_factors

        llm_extractor = LLMFormulaExtractor()
        self._extract_document_formula_candidates(llm_extractor)
        self._resolve_active_methodology(chunks, llm_extractor)

        # Pass 2: create or refresh DataEngines for every document in the session.
        self._rebuild_data_engines()
        self._summarize_detected_metrics()
        if self._refresh_metric_interpretations():
            self._rebuild_data_engines()
            self._summarize_detected_metrics()
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk
        self._summarize_formula_validation()
        self._summarize_formula_input_columns()
        self._summarize_report_readiness()
        self._summarize_calculation_audit()

        elapsed = time.perf_counter() - start
        return BuildStats(
            session_id=self.session_id,
            document_count=len(parsed_documents),
            element_count=sum(len(document.elements) for document in parsed_documents),
            chunk_count=len(chunks),
            embedding_count=len(vectors),
            elapsed_seconds=elapsed,
        )

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalHit]:
        return self.retrieval.search(query, top_k=top_k)

    def clear(self) -> None:
        self.vector_index.clear()
        self.keyword_index.clear()
        self.documents.clear()
        self.pdf_documents.clear()
        self.excel_documents.clear()
        self.spreadsheet_dataframes.clear()
        self.table_dataframes.clear()
        self.fact_dataframes.clear()
        self.data_engines.clear()
        self.chunks.clear()
        self.document_emission_factors.clear()
        self.document_formula_candidates.clear()
        self.custom_formula = None
        self.formula_extraction_method = "default"
        self.custom_formula_user_inputs.clear()
        self.custom_formula_status = "default"
        self.custom_formula_missing_variables.clear()
        self.custom_formula_validation_by_document.clear()
        self.factor_override_keys.clear()
        self.formula_input_columns.clear()
        self.formula_conflicts.clear()
        self.factor_conflicts.clear()
        self.methodology_resolution = {"formula_doc_id": None, "factor_doc_ids": {}}
        self.methodology_status = "clear"
        self.methodology_warnings.clear()
        self.metric_user_overrides.clear()
        self.metric_system_overrides.clear()
        self.detected_metrics = []
        self.calculation_audit = {}
        self.has_structured_data = False
        self.structured_document_count = 0
        self.structured_district_count = 0
        self.report_generation_status = "ready"
        self.report_generation_warnings.clear()

    def update_custom_formula_inputs(self, variables: dict[str, dict[str, object]]) -> None:
        if self.custom_formula is None:
            raise ValueError("No custom formula is currently available to resolve.")

        cleaned: dict[str, dict[str, object]] = {}
        for variable, payload in (variables or {}).items():
            name = str(variable).strip()
            if not name:
                continue
            kind = str(payload.get("type", "")).strip().lower()
            if kind == "constant":
                try:
                    value = float(payload.get("value"))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Variable '{name}' requires a numeric constant value.") from exc
                cleaned[name] = {"type": "constant", "value": value}
            elif kind == "column":
                column = str(payload.get("column", "") or "").strip()
                if not column:
                    raise ValueError(f"Variable '{name}' requires a column selection.")
                cleaned[name] = {"type": "column", "column": column}
            else:
                raise ValueError(f"Variable '{name}' has an unsupported input type: {payload.get('type')!r}.")

        self.custom_formula_user_inputs.update(cleaned)
        self._rebuild_data_engines()
        self._summarize_detected_metrics()
        self._summarize_formula_validation()
        self._summarize_formula_input_columns()
        self._summarize_report_readiness()
        self._summarize_calculation_audit()

    def update_methodology_resolution(
        self,
        *,
        formula_doc_id: str | None = None,
        factor_doc_ids: dict[str, str] | None = None,
    ) -> None:
        if formula_doc_id is not None:
            if formula_doc_id and formula_doc_id not in self.document_formula_candidates:
                raise ValueError(f"Unknown formula source document: {formula_doc_id}")
            self.methodology_resolution["formula_doc_id"] = formula_doc_id or None

        if factor_doc_ids:
            updated = dict(self.methodology_resolution.get("factor_doc_ids", {}) or {})
            candidates = self._factor_candidates_by_key()
            for metric_key, doc_id in factor_doc_ids.items():
                valid_doc_ids = {entry["doc_id"] for entry in candidates.get(metric_key, [])}
                if doc_id not in valid_doc_ids:
                    raise ValueError(f"Unknown factor source for {metric_key}: {doc_id}")
                updated[metric_key] = doc_id
            self.methodology_resolution["factor_doc_ids"] = updated

        self._resolve_active_methodology()
        self._rebuild_data_engines()
        self._summarize_detected_metrics()
        self._summarize_formula_validation()
        self._summarize_formula_input_columns()
        self._summarize_report_readiness()
        self._summarize_calculation_audit()

    def update_metric_overrides(self, metrics: dict[str, dict[str, object]]) -> None:
        cleaned: dict[str, dict[str, object]] = {}
        for metric_key, payload in (metrics or {}).items():
            key = str(metric_key).strip()
            if not key:
                continue
            normalized = normalize_metric_override(key, payload or {})
            normalized["classification_source"] = "user"
            cleaned[key] = normalized

        self.metric_user_overrides = merge_metric_overrides(self.metric_user_overrides, cleaned)
        self._rebuild_data_engines()
        self._summarize_detected_metrics()
        self._summarize_formula_validation()
        self._summarize_formula_input_columns()
        self._summarize_report_readiness()
        self._summarize_calculation_audit()

    def _summarize_formula_validation(self) -> None:
        self.custom_formula_validation_by_document = []
        self.custom_formula_missing_variables = []
        if self.custom_formula is None:
            self.custom_formula_status = "default"
            return

        seen_missing: set[str] = set()
        statuses: list[str] = []
        for doc_id, engine in self.data_engines.items():
            if engine is None:
                continue
            document = self.documents.get(doc_id)
            missing = list(getattr(engine, "custom_formula_missing_variables", []) or [])
            for name in missing:
                if name not in seen_missing:
                    seen_missing.add(name)
                    self.custom_formula_missing_variables.append(name)
            status = getattr(engine, "custom_formula_status", "default")
            statuses.append(status)
            self.custom_formula_validation_by_document.append(
                {
                    "doc_id": doc_id,
                    "filename": document.filename if document else doc_id,
                    "status": status,
                    "missing_variables": missing,
                    "bound_variables": dict(getattr(engine, "custom_formula_bound_variables", {}) or {}),
                }
            )

        if not statuses:
            self.custom_formula_status = "valid"
        elif all(status == "ready" for status in statuses):
            self.custom_formula_status = "valid"
        elif all(status in {"incomplete", "invalid"} for status in statuses):
            self.custom_formula_status = "incomplete"
        else:
            self.custom_formula_status = "partial"

    def _rebuild_data_engines(self) -> None:
        session_factors = self._session_emission_factors()
        metric_overrides = self._effective_metric_overrides()
        for doc_id in self.documents:
            try:
                dataframe = combine_dataframes(
                    self.table_dataframes.get(doc_id),
                    self.fact_dataframes.get(doc_id),
                )
            except ImportError:
                dataframe = None

            if dataframe is not None and not dataframe.empty:
                self.data_engines[doc_id] = DataEngine(
                    dataframe,
                    emission_factors=session_factors or None,
                    custom_formula=self.custom_formula,
                    custom_formula_inputs=self.custom_formula_user_inputs,
                    metric_overrides=metric_overrides,
                )
            else:
                self.data_engines[doc_id] = None

    def _session_emission_factors(self) -> dict[str, float]:
        return self._active_factor_overrides()

    def _effective_metric_overrides(self) -> dict[str, dict[str, object]]:
        return merge_metric_overrides(self.metric_system_overrides, self.metric_user_overrides)

    def _refresh_metric_interpretations(self) -> bool:
        enabled = str(os.getenv("ENABLE_LLM_METRIC_INTERPRETATION", "")).strip().lower()
        if enabled not in {"1", "true", "yes", "on"}:
            if self.metric_system_overrides:
                self.metric_system_overrides = {}
                return True
            return False
        try:
            suggestions = suggest_metric_overrides(self.detected_metrics)
        except Exception:
            suggestions = {}
        suggestions = self._sanitize_metric_interpretations(suggestions)
        changed = suggestions != self.metric_system_overrides
        if not changed:
            return False
        self.metric_system_overrides = suggestions
        return True

    def _sanitize_metric_interpretations(
        self,
        suggestions: dict[str, dict[str, object]],
    ) -> dict[str, dict[str, object]]:
        current_metrics = {
            str(item.get("metric_key")): item
            for item in (self.detected_metrics or [])
            if item.get("metric_key")
        }
        sanitized: dict[str, dict[str, object]] = {}
        for metric_key, override in (suggestions or {}).items():
            current = current_metrics.get(metric_key)
            if current is None:
                sanitized[metric_key] = override
                continue

            # Preserve strong deterministic classifications for already-recognized
            # sustainability metrics such as tree_count, dam_occupancy, recycling_rate,
            # and similar context/resource indicators. LLM suggestions remain useful for
            # ambiguous "other" metrics but should not silently move a known signal into
            # the wrong section of the report.
            has_strong_deterministic_classification = (
                current.get("classification_source") in {"heuristic", "registry"}
                and bool(current.get("sustainability_related"))
                and str(current.get("category") or "").strip().lower() not in {"", "other", "climate"}
            )
            if has_strong_deterministic_classification:
                preserved = dict(override or {})
                preserved["category"] = current.get("category")
                preserved["role"] = current.get("role")
                preserved["report_section"] = current.get("report_section")
                preserved["sustainability_related"] = current.get("sustainability_related")
                preserved["classification_source"] = current.get("classification_source")
                sanitized[metric_key] = preserved
            else:
                sanitized[metric_key] = override
        return sanitized

    def _summarize_formula_input_columns(self) -> None:
        columns: dict[str, dict[str, object]] = {}
        for doc_id, engine in self.data_engines.items():
            if engine is None or not hasattr(engine, "formula_candidate_columns"):
                continue
            document = self.documents.get(doc_id)
            filename = document.filename if document else doc_id
            for column in engine.formula_candidate_columns():
                entry = columns.setdefault(
                    column,
                    {
                        "value": column,
                        "label": column.replace("_", " "),
                        "documents": [],
                    },
                )
                documents = entry["documents"]
                if filename not in documents:
                    documents.append(filename)

        self.formula_input_columns = sorted(columns.values(), key=lambda item: str(item["label"]))

    def _summarize_report_readiness(self) -> None:
        district_names: set[str] = set()
        self.structured_document_count = 0
        self.structured_district_count = 0
        self.has_structured_data = False
        self.report_generation_status = "ready"
        self.report_generation_warnings = list(self.methodology_warnings)

        for engine in self.data_engines.values():
            if engine is None or not hasattr(engine, "districts"):
                continue
            districts = [district for district in engine.districts() if str(district).strip()]
            if not districts:
                continue
            self.structured_document_count += 1
            district_names.update(str(district).strip() for district in districts)

        self.structured_district_count = len(district_names)
        self.has_structured_data = self.structured_district_count > 0

        if self.methodology_status == "needs_resolution":
            self.report_generation_status = "blocked_methodology_conflict"
        elif self.custom_formula and not self.has_structured_data:
            self.report_generation_status = "blocked_missing_structured_data"
            self.report_generation_warnings.append("custom_formula_detected_but_no_structured_data")
        elif self.custom_formula_status in {"incomplete", "partial"} and self.custom_formula_missing_variables:
            self.report_generation_status = "blocked_missing_formula_inputs"
            self.report_generation_warnings.extend(
                [f"custom_formula_missing_variable_definition:{name}" for name in self.custom_formula_missing_variables]
            )

    def _extract_document_formula_candidates(self, llm_extractor: LLMFormulaExtractor) -> None:
        for doc_id, document in self.pdf_documents.items():
            formula = llm_extractor.extract_from_documents([document])
            if formula is not None:
                self.document_formula_candidates[doc_id] = formula

    def _resolve_active_methodology(
        self,
        chunks: list[Chunk] | None = None,
        llm_extractor: LLMFormulaExtractor | None = None,
    ) -> None:
        self.factor_conflicts = self._detect_factor_conflicts()
        self.formula_conflicts = self._detect_formula_conflicts()
        self.methodology_warnings = []
        self.methodology_status = "clear"

        active_formula, formula_method = self._selected_formula_candidate()
        active_factors = self._active_factor_overrides()

        if self.factor_conflicts or self.formula_conflicts:
            unresolved_factor_keys = {
                conflict["metric_key"]
                for conflict in self.factor_conflicts
                if not conflict.get("selected_doc_id")
            }
            unresolved_formula = any(not conflict.get("selected_doc_id") for conflict in self.formula_conflicts)
            if unresolved_factor_keys or unresolved_formula:
                self.methodology_status = "needs_resolution"
                if unresolved_formula:
                    self.methodology_warnings.append("methodology_formula_conflict_unresolved")
                for key in sorted(unresolved_factor_keys):
                    self.methodology_warnings.append(f"methodology_factor_conflict_unresolved:{key}")

        self.custom_formula = active_formula
        self.formula_extraction_method = formula_method

        if self.custom_formula is None and chunks and self.methodology_status == "clear":
            llm_extractor = llm_extractor or LLMFormulaExtractor()
            rag_formula = self._rag_formula_candidate(chunks, llm_extractor)
            if rag_formula is not None:
                self.custom_formula = rag_formula
                self.formula_extraction_method = "rag_assisted"

        if self.custom_formula is None:
            self.formula_extraction_method = "default"
            self.custom_formula_status = "default"

        self.factor_override_keys = sorted(active_factors)

    def _detect_factor_conflicts(self) -> list[dict[str, object]]:
        conflicts: list[dict[str, object]] = []
        selected_docs = dict(self.methodology_resolution.get("factor_doc_ids", {}) or {})
        for metric_key, entries in self._factor_candidates_by_key().items():
            unique_values = {round(float(entry["value"]), 12) for entry in entries}
            if len(unique_values) <= 1:
                continue
            selected_doc_id = selected_docs.get(metric_key)
            conflicts.append(
                {
                    "metric_key": metric_key,
                    "selected_doc_id": selected_doc_id if selected_doc_id in {entry["doc_id"] for entry in entries} else None,
                    "candidates": entries,
                }
            )
        return conflicts

    def _detect_formula_conflicts(self) -> list[dict[str, object]]:
        grouped: dict[tuple, list[dict[str, object]]] = defaultdict(list)
        for doc_id, formula in self.document_formula_candidates.items():
            grouped[self._formula_signature(formula)].append(self._formula_candidate_entry(doc_id, formula))

        if len(grouped) <= 1:
            return []

        candidates = [entry for entries in grouped.values() for entry in entries]
        selected_doc_id = self.methodology_resolution.get("formula_doc_id")
        if selected_doc_id not in {entry["doc_id"] for entry in candidates}:
            selected_doc_id = None
        return [{"selected_doc_id": selected_doc_id, "candidates": candidates}]

    def _selected_formula_candidate(self) -> tuple[object | None, str]:
        candidates = list(self.document_formula_candidates.items())
        if not candidates:
            return None, "default"

        if not self.formula_conflicts:
            return candidates[0][1], "direct"

        selected_doc_id = self.methodology_resolution.get("formula_doc_id")
        if selected_doc_id and selected_doc_id in self.document_formula_candidates:
            return self.document_formula_candidates[selected_doc_id], "direct"
        return None, "default"

    def _active_factor_overrides(self) -> dict[str, float]:
        active: dict[str, float] = {}
        selected_docs = dict(self.methodology_resolution.get("factor_doc_ids", {}) or {})
        candidates_by_key = self._factor_candidates_by_key()

        for metric_key, entries in candidates_by_key.items():
            unique_values = {round(float(entry["value"]), 12) for entry in entries}
            if len(unique_values) == 1:
                active[metric_key] = float(entries[0]["value"])
                continue
            selected_doc_id = selected_docs.get(metric_key)
            if not selected_doc_id:
                continue
            for entry in entries:
                if entry["doc_id"] == selected_doc_id:
                    active[metric_key] = float(entry["value"])
                    break
        return active

    def _factor_candidates_by_key(self) -> dict[str, list[dict[str, object]]]:
        candidates: dict[str, list[dict[str, object]]] = defaultdict(list)
        for doc_id, factors in self.document_emission_factors.items():
            document = self.documents.get(doc_id)
            for metric_key, value in factors.items():
                candidates[metric_key].append(
                    {
                        "doc_id": doc_id,
                        "filename": document.filename if document else doc_id,
                        "value": float(value),
                    }
                )
        return dict(candidates)

    def _rag_formula_candidate(self, chunks: list[Chunk], llm_extractor: LLMFormulaExtractor):
        _formula_queries = [
            "emission formula calculation CO2 total carbon footprint",
            "CO2 equals electricity natural gas factor formula",
        ]
        rag_texts: list[str] = []
        for _q in _formula_queries:
            for hit in self.retrieval.search(_q, top_k=3):
                chunk_text = getattr(getattr(hit, "chunk", None), "text", None)
                if chunk_text and chunk_text not in rag_texts:
                    rag_texts.append(chunk_text)
        if not rag_texts:
            return None
        return llm_extractor.extract_from_text("\n\n".join(rag_texts))

    def _summarize_calculation_audit(self) -> None:
        selected_formula_doc_id = self.methodology_resolution.get("formula_doc_id")
        if self.custom_formula is not None and not selected_formula_doc_id:
            for doc_id, formula in self.document_formula_candidates.items():
                if self._formula_signature(formula) == self._formula_signature(self.custom_formula):
                    selected_formula_doc_id = doc_id
                    break

        selected_formula_entry = (
            self._formula_candidate_entry(selected_formula_doc_id, self.custom_formula)
            if self.custom_formula is not None and selected_formula_doc_id
            else None
        )

        self.calculation_audit = {
            "formula": {
                "status": self.custom_formula_status,
                "extraction_method": self.formula_extraction_method,
                "selected": selected_formula_entry,
                "candidates": [
                    self._formula_candidate_entry(doc_id, formula)
                    for doc_id, formula in self.document_formula_candidates.items()
                ],
                "conflicts": list(self.formula_conflicts),
                "user_inputs": dict(self.custom_formula_user_inputs),
                "fallback": self.custom_formula is None,
            },
            "factors": {
                "selected": self._selected_factor_audit_entries(),
                "candidates": [
                    {
                        "doc_id": doc_id,
                        "filename": (self.documents.get(doc_id).filename if self.documents.get(doc_id) else doc_id),
                        "factors": factors,
                    }
                    for doc_id, factors in self.document_emission_factors.items()
                ],
                "conflicts": list(self.factor_conflicts),
            },
            "methodology": {
                "status": self.methodology_status,
                "warnings": list(self.methodology_warnings),
                "resolution": dict(self.methodology_resolution),
            },
            "report_generation": {
                "status": self.report_generation_status,
                "warnings": list(self.report_generation_warnings),
            },
            "metrics": {
                "detected": list(self.detected_metrics),
                "user_overrides": dict(self.metric_user_overrides),
            },
        }

    def _summarize_detected_metrics(self) -> None:
        combined: dict[str, dict[str, object]] = {}
        effective_overrides = self._effective_metric_overrides()

        for doc_id, engine in self.data_engines.items():
            if engine is None:
                continue
            document = self.documents.get(doc_id)
            filename = document.filename if document else doc_id
            bound_variables = dict(getattr(engine, "custom_formula_bound_variables", {}) or {})
            for metric_key, metric in dict(getattr(engine, "discovered_metrics", {}) or {}).items():
                current = combined.setdefault(
                    metric_key,
                    {
                        "metric": metric,
                        "documents": set(),
                        "used_in_calculation": False,
                    },
                )
                current["documents"].add(filename)
                if metric.numeric_availability > current["metric"].numeric_availability:
                    current["metric"] = metric
                bound_to_metric_column = metric.source_column and any(
                    label == f"column:{metric.source_column}"
                    for label in bound_variables.values()
                )
                if metric_key in {"electricity", "natural_gas"} or metric_key in bound_variables or bound_to_metric_column:
                    current["used_in_calculation"] = True

        detected = []
        for metric_key, payload in combined.items():
            metric = payload["metric"]
            override = effective_overrides.get(metric_key, {})
            classification_source = override.get("classification_source", metric.classification_source)
            metric_payload = summarize_metric_for_api(
                metric,
                documents=sorted(payload["documents"]),
                used_in_calculation=bool(payload["used_in_calculation"]),
                used_in_narrative=bool(metric.sustainability_related),
            )
            metric_payload["classification_source"] = classification_source
            detected.append(metric_payload)

        self.detected_metrics = sorted(
            detected,
            key=lambda item: (
                not bool(item.get("is_known_metric")),
                not bool(item.get("sustainability_related")),
                str(item.get("display_name") or item.get("metric_key") or "").casefold(),
            ),
        )

    def _selected_factor_audit_entries(self) -> dict[str, dict[str, object]]:
        reference_factors = _load_reference_data().get("emission_factors", {})
        active = {**reference_factors, **self._active_factor_overrides()}
        selected_docs = dict(self.methodology_resolution.get("factor_doc_ids", {}) or {})
        entries: dict[str, dict[str, object]] = {}
        candidates_by_key = self._factor_candidates_by_key()
        for metric_key, value in active.items():
            selected_doc_id = selected_docs.get(metric_key)
            candidate_entries = candidates_by_key.get(metric_key, [])
            source = "reference"
            if not selected_doc_id and candidate_entries:
                selected_doc_id = candidate_entries[0]["doc_id"]
                source = "document"
            elif selected_doc_id:
                source = "document"
            document = self.documents.get(selected_doc_id) if selected_doc_id else None
            entries[metric_key] = {
                "value": float(value),
                "source": source,
                "doc_id": selected_doc_id,
                "filename": document.filename if document else selected_doc_id,
            }
        return entries

    @staticmethod
    def _formula_signature(formula) -> tuple:
        return (
            getattr(formula, "expression", None),
            tuple(sorted((getattr(formula, "constants", {}) or {}).items())),
        )

    def _formula_candidate_entry(self, doc_id: str | None, formula) -> dict[str, object]:
        document = self.documents.get(doc_id) if doc_id else None
        return {
            "doc_id": doc_id,
            "filename": document.filename if document else doc_id,
            "expression": getattr(formula, "expression", None),
            "constants": dict(getattr(formula, "constants", {}) or {}),
            "variable_hints": dict(getattr(formula, "variable_hints", {}) or {}),
            "confidence": getattr(formula, "confidence", None),
            "source_text": getattr(formula, "source_text", None),
        }


class SessionManager:
    """Creates and disposes isolated retrieval sessions."""

    def __init__(self) -> None:
        self.sessions: dict[str, RetrievalSession] = {}

    def create_session(self) -> RetrievalSession:
        session = RetrievalSession()
        self.sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> RetrievalSession:
        return self.sessions[session_id]

    def close(self, session_id: str) -> None:
        session = self.sessions.pop(session_id, None)
        if session:
            session.clear()
