import tempfile
import unittest
from pathlib import Path

from rag_retrieval.generation import OpenAIReportGenerator
from rag_retrieval.plotter import generate_report_charts
from rag_retrieval.renderer import render_report
from rag_retrieval.report_models import GeneratedReport, ReportAssets, ReportInput


def sample_report_input():
    structured = [
        {
            "doc_id": "doc-1",
            "filename": "report.pdf",
            "source_type": "pdf",
            "parser": "pymupdf-layout",
            "data": {
                "district": "Bakirkoy",
                "total_emission": 1000.0,
                "gas_emission": 1000.0,
                "electricity_emission": 0.0,
                "direct_emissions": 0.0,
                "water_consumption": 4200.0,
                "water_per_capita": 0.012,
                "water_growth": 0.10,
                "metrics": {
                    "water": {
                        "metric_key": "water",
                        "label": "Water",
                        "category": "water",
                        "unit": "m3",
                        "role": "resource_kpi",
                        "report_section": "Water Overview",
                        "value": 4200.0,
                        "per_capita": 0.012,
                        "growth": 0.10,
                    }
                },
                "growth": 0.25,
                "warnings": ["electricity_consumption_not_found"],
            },
        }
    ]
    return ReportInput(
        title="Test Sustainability Report",
        language="English",
        session_id="session-1",
        generated_at="2026-04-20T00:00:00+00:00",
        documents=[
            {
                "doc_id": "doc-1",
                "filename": "report.pdf",
                "parser": "pymupdf-layout",
                "element_count": 3,
                "chunk_count": 2,
                "table_rows": 10,
                "fact_rows": 0,
                "has_data_engine": True,
            }
        ],
        query_results=[],
        structured_results=structured,
        retrieval_context=[],
        sources=[{"doc_id": "doc-1", "filename": "report.pdf", "source_type": "pdf", "usage": ["structured"]}],
        warnings=["report.pdf: electricity_consumption_not_found"],
    )


class ReportGenerationTests(unittest.TestCase):
    def test_openai_generator_uses_fallback_without_api_key(self):
        report_input = sample_report_input()

        report = OpenAIReportGenerator(api_key="").generate(report_input)

        self.assertIn("Executive Summary", report.ai_content_markdown)
        self.assertIn("Water Overview", report.ai_content_markdown)
        self.assertIn("OPENAI_API_KEY not set", report.warnings[-1])

    def test_chart_and_renderer_outputs_files(self):
        report_input = sample_report_input()
        generated = GeneratedReport(
            title=report_input.title,
            language=report_input.language,
            generated_at=report_input.generated_at,
            report_input=report_input,
            ai_content_markdown="# Executive Summary\n- Bakirkoy total emissions were 1,000.",
            warnings=report_input.warnings,
        )

        with tempfile.TemporaryDirectory() as tmp:
            charts = generate_report_charts(report_input, Path(tmp) / "charts")
            rendered = render_report(
                GeneratedReport(
                    title=generated.title,
                    language=generated.language,
                    generated_at=generated.generated_at,
                    report_input=generated.report_input,
                    ai_content_markdown=generated.ai_content_markdown,
                    charts=charts,
                    warnings=generated.warnings,
                ),
                Path(tmp) / "reports",
                ReportAssets(),
            )

            self.assertTrue(charts)
            self.assertTrue(Path(rendered.html_path).exists())
            self.assertTrue(Path(rendered.pdf_path).exists())
            self.assertTrue(any("water_consumption_by_district" in chart["path"] for chart in charts))

    def test_renderer_includes_public_coverage_note_for_custom_formula_fallback(self):
        report_input = sample_report_input()
        report_input = ReportInput(
            title=report_input.title,
            language=report_input.language,
            session_id=report_input.session_id,
            generated_at=report_input.generated_at,
            documents=report_input.documents,
            query_results=report_input.query_results,
            structured_results=report_input.structured_results,
            retrieval_context=report_input.retrieval_context,
            sources=report_input.sources,
            warnings=[
                "test.xlsx: custom_formula_missing_variable_definition:renewable",
            ],
        )
        generated = GeneratedReport(
            title=report_input.title,
            language=report_input.language,
            generated_at=report_input.generated_at,
            report_input=report_input,
            ai_content_markdown="# Executive Summary\nFallback was used.",
            warnings=report_input.warnings,
        )

        with tempfile.TemporaryDirectory() as tmp:
            rendered = render_report(generated, Path(tmp) / "reports", ReportAssets())
            html = Path(rendered.html_path).read_text(encoding="utf-8")

        self.assertIn("Coverage Notes", html)
        self.assertIn("could not be fully applied", html)

    def test_public_metrics_merge_water_and_emissions_from_separate_documents(self):
        report_input = sample_report_input()
        report_input = ReportInput(
            title=report_input.title,
            language=report_input.language,
            session_id=report_input.session_id,
            generated_at=report_input.generated_at,
            documents=report_input.documents,
            query_results=report_input.query_results,
            structured_results=[
                *report_input.structured_results,
                {
                    "doc_id": "doc-2",
                    "filename": "water.xlsx",
                    "source_type": "spreadsheet",
                    "parser": "spreadsheet",
                    "data": {
                        "district": "Bakirkoy",
                        "total_emission": 0.0,
                        "gas_emission": 0.0,
                        "electricity_emission": 0.0,
                        "direct_emissions": 0.0,
                        "water_consumption": 5100.0,
                        "water_per_capita": 0.02,
                        "water_growth": 0.08,
                        "metrics": {
                            "water": {
                                "metric_key": "water",
                                "label": "Water",
                                "category": "water",
                                "unit": "m3",
                                "role": "resource_kpi",
                                "report_section": "Water Overview",
                                "value": 5100.0,
                                "per_capita": 0.02,
                                "growth": 0.08,
                            }
                        },
                        "warnings": [],
                    },
                },
            ],
            retrieval_context=report_input.retrieval_context,
            sources=report_input.sources,
            warnings=report_input.warnings,
        )

        report = OpenAIReportGenerator(api_key="").generate(report_input)
        self.assertIn("Water Overview", report.ai_content_markdown)
        self.assertIn("5,100.00 m3", report.ai_content_markdown)


if __name__ == "__main__":
    unittest.main()
