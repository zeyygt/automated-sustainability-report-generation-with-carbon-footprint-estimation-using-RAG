"""Session-scoped retrieval pipeline for sustainability RAG systems."""

from .config import ChunkingConfig, EmbeddingConfig, ParserConfig, RetrievalConfig
from .data_engine import DataEngine
from .fact_extractor import FactExtractor
from .models import Chunk, DocumentInput, ParsedDocument, ParsedElement, RetrievalHit
from .pipeline import analyze_data_engines, build_query_response, extract_district, format_retrieval_hits, handle_query
from .report_builder import build_report_input
from .report_models import GeneratedReport, ReportAssets, ReportInput
from .report_pipeline import generate_sustainability_report
from .router import combine_dataframes, parsed_tables_to_dataframe, route_query, spreadsheet_to_dataframe
from .session import RetrievalSession, SessionManager

__all__ = [
    "Chunk",
    "ChunkingConfig",
    "DocumentInput",
    "EmbeddingConfig",
    "DataEngine",
    "FactExtractor",
    "GeneratedReport",
    "ParsedDocument",
    "ParsedElement",
    "ParserConfig",
    "ReportAssets",
    "RetrievalConfig",
    "RetrievalHit",
    "ReportInput",
    "RetrievalSession",
    "SessionManager",
    "analyze_data_engines",
    "build_report_input",
    "build_query_response",
    "extract_district",
    "format_retrieval_hits",
    "generate_sustainability_report",
    "handle_query",
    "combine_dataframes",
    "parsed_tables_to_dataframe",
    "route_query",
    "spreadsheet_to_dataframe",
]
