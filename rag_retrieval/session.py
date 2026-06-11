"""Session lifecycle API for building and querying temporary indexes."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Iterable

from .chunking import StructureAwareChunker
from .config import ChunkingConfig, EmbeddingConfig, ParserConfig, RetrievalConfig
from .data_engine import DataEngine
from .embeddings import Embedder, build_embedder
from .fact_extractor import FactExtractor
from .formula_extractor import FormulaExtractor
from .llm_formula_extractor import LLMFormulaExtractor
from .index import BM25Index, InMemoryVectorIndex
from .ingestion import DocumentIngestor
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
        self.custom_formula = None  # ExtractedFormula | None
        self.formula_extraction_method: str = "default"  # "direct" | "rag_assisted" | "default"
        self.custom_formula_user_inputs: dict[str, dict[str, object]] = {}
        self.custom_formula_status: str = "default"
        self.custom_formula_missing_variables: list[str] = []
        self.custom_formula_validation_by_document: list[dict[str, object]] = []
        self.factor_override_keys: list[str] = []
        self.formula_input_columns: list[dict[str, object]] = []
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

        # Merge all per-document factors into one session-level dict.
        # If two documents define the same key, last-parsed value wins.
        session_factors: dict[str, float] = {}
        for factors in self.document_emission_factors.values():
            session_factors.update(factors)
        self.factor_override_keys = sorted(session_factors)

        # ── Formula extraction: three-level fallback chain ────────────────────
        # Level 1: direct LLM extraction on all non-spreadsheet document text.
        # Level 2: RAG-assisted — search the index for formula-related chunks,
        #          run LLM extraction on that focused text.
        # Level 3: default formula (consumption × emission_factor) with a warning.
        llm_extractor = LLMFormulaExtractor()
        self.custom_formula = llm_extractor.extract_from_documents(list(self.pdf_documents.values()))

        if self.custom_formula is not None:
            self.formula_extraction_method = "direct"
        elif chunks:
            # Level 2: ask the vector index for the most formula-relevant chunks
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
            if rag_texts:
                self.custom_formula = llm_extractor.extract_from_text("\n\n".join(rag_texts))
                if self.custom_formula is not None:
                    self.formula_extraction_method = "rag_assisted"

        if self.custom_formula is None:
            self.formula_extraction_method = "default"
            self.custom_formula_status = "default"

        # Pass 2: create or refresh DataEngines for every document in the session.
        self._rebuild_data_engines()
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk
        self._summarize_formula_validation()
        self._summarize_formula_input_columns()
        self._summarize_report_readiness()

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
        self.custom_formula = None
        self.formula_extraction_method = "default"
        self.custom_formula_user_inputs.clear()
        self.custom_formula_status = "default"
        self.custom_formula_missing_variables.clear()
        self.custom_formula_validation_by_document.clear()
        self.factor_override_keys.clear()
        self.formula_input_columns.clear()
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
        self._summarize_formula_validation()
        self._summarize_formula_input_columns()
        self._summarize_report_readiness()

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
                )
            else:
                self.data_engines[doc_id] = None

    def _session_emission_factors(self) -> dict[str, float]:
        session_factors: dict[str, float] = {}
        for factors in self.document_emission_factors.values():
            session_factors.update(factors)
        self.factor_override_keys = sorted(session_factors)
        return session_factors

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
        self.report_generation_warnings = []

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

        if self.custom_formula and not self.has_structured_data:
            self.report_generation_status = "blocked_missing_structured_data"
            self.report_generation_warnings.append("custom_formula_detected_but_no_structured_data")
        elif self.custom_formula_status in {"incomplete", "partial"} and self.custom_formula_missing_variables:
            self.report_generation_status = "blocked_missing_formula_inputs"
            self.report_generation_warnings.extend(
                [f"custom_formula_missing_variable_definition:{name}" for name in self.custom_formula_missing_variables]
            )


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
