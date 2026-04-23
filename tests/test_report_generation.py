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


if __name__ == "__main__":
    unittest.main()
