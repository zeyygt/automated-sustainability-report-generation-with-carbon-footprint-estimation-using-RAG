"""End-to-end report generation pipeline."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable

from .generation import OpenAIReportGenerator
from .plotter import generate_report_charts
from .renderer import discover_report_assets, render_report
from .report_builder import build_report_input
from .report_models import GeneratedReport, ReportAssets
from .session import RetrievalSession


def generate_sustainability_report(
    session: RetrievalSession,
    queries: Iterable[str] | None = None,
    title: str = "Istanbul Metropolitan Municipality Sustainability Report",
    language: str = "English",
    output_dir: str | Path = "outputs/reports",
    chart_dir: str | Path | None = None,
    assets: ReportAssets | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> GeneratedReport:
    output_path = Path(output_dir)
    charts_path = Path(chart_dir) if chart_dir else output_path / "charts"
    assets = assets or discover_report_assets()

    report_input = build_report_input(session, queries=queries, title=title, language=language)
    generator = OpenAIReportGenerator(model=model, reasoning_effort=reasoning_effort)
    generated = generator.generate(report_input)
    charts = generate_report_charts(report_input, charts_path)
    generated = replace(generated, charts=charts)
    return render_report(generated, output_path, assets)
