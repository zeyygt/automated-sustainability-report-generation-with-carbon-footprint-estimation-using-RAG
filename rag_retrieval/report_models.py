"""Data contracts for generated sustainability reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ReportAssets:
    ibb_logo_path: Path | None = None
    cycle_logo_path: Path | None = None


@dataclass(frozen=True, slots=True)
class ReportInput:
    title: str
    language: str
    session_id: str
    generated_at: str
    documents: list[dict[str, Any]]
    query_results: list[dict[str, Any]]
    structured_results: list[dict[str, Any]]
    retrieval_context: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    warnings: list[str]


@dataclass(frozen=True, slots=True)
class GeneratedReport:
    title: str
    language: str
    generated_at: str
    report_input: ReportInput
    ai_content_markdown: str
    charts: list[dict[str, str]] = field(default_factory=list)
    html_path: Path | None = None
    pdf_path: Path | None = None
    warnings: list[str] = field(default_factory=list)
