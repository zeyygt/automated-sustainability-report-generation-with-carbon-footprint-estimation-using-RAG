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

        # Pass 2: create DataEngines with session-level factors and custom formula.
        for document in parsed_documents:
            try:
                dataframe = combine_dataframes(table_dfs[document.doc_id], fact_dfs[document.doc_id])
            except ImportError:
                dataframe = None

            if dataframe is not None and not dataframe.empty:
                self.data_engines[document.doc_id] = DataEngine(
                    dataframe,
                    emission_factors=session_factors or None,
                    custom_formula=self.custom_formula,
                )
            else:
                self.data_engines[document.doc_id] = None
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk

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
