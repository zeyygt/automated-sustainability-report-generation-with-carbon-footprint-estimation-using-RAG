"""Session-scoped retrieval pipeline for sustainability RAG systems."""

from .config import ChunkingConfig, EmbeddingConfig, ParserConfig, RetrievalConfig
from .data_engine import DataEngine
from .fact_extractor import FactExtractor
from .models import Chunk, DocumentInput, ParsedDocument, ParsedElement, RetrievalHit
from .metric_discovery import DiscoveredMetric
from .metric_registry import MetricDefinition, metric_registry
from .pipeline import analyze_data_engines, build_query_response, extract_district, format_retrieval_hits, handle_query
from .ragas_evaluation import run_ragas_retrieval_evaluation
from .ragas_generation_evaluation import run_ragas_generation_evaluation
from .report_builder import build_report_input
from .insight_engine import build_report_insights
from .recommendation_engine import build_report_recommendations
from .report_models import GeneratedReport, ReportAssets, ReportInput
from .report_pipeline import generate_sustainability_report
from .router import combine_dataframes, parsed_tables_to_dataframe, route_query, spreadsheet_to_dataframe
from .session import RetrievalSession, SessionManager

__all__ = [
    "Chunk",
    "ChunkingConfig",
    "DocumentInput",
    "DiscoveredMetric",
    "EmbeddingConfig",
    "DataEngine",
    "FactExtractor",
    "GeneratedReport",
    "MetricDefinition",
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
    "build_report_recommendations",
    "extract_district",
    "format_retrieval_hits",
    "generate_sustainability_report",
    "handle_query",
    "build_report_insights",
    "metric_registry",
    "combine_dataframes",
    "parsed_tables_to_dataframe",
    "run_ragas_generation_evaluation",
    "run_ragas_retrieval_evaluation",
    "route_query",
    "spreadsheet_to_dataframe",
]
