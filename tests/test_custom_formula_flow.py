import tempfile
import unittest
from os import environ
from pathlib import Path
from unittest.mock import patch

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from rag_retrieval.llm_formula_extractor import ExtractedFormula
from rag_retrieval.report_pipeline import generate_sustainability_report
from rag_retrieval.session import RetrievalSession


def _write_pdf(path: Path, lines: list[str]) -> None:
    pdf = canvas.Canvas(str(path), pagesize=A4)
    x = 72
    y = 800
    for line in lines:
        pdf.drawString(x, y, line)
        y -= 18
    pdf.save()


class CustomFormulaFlowTests(unittest.TestCase):
    def test_session_build_index_applies_factor_override_from_documents(self):
        session = RetrievalSession()

        with patch("rag_retrieval.session.LLMFormulaExtractor") as extractor_cls:
            extractor = extractor_cls.return_value
            extractor.extract_from_documents.return_value = None
            extractor.extract_from_text.return_value = None
            session.build_index(
                [
                    Path("sample_docs/emission_factors.pdf"),
                    Path("sample_docs/consumption_data.csv"),
                ]
            )

        result = next(
            engine.analyze_district("Kadikoy")
            for engine in session.data_engines.values()
            if engine is not None and engine.analyze_district("Kadikoy")
        )
        self.assertEqual(session.factor_override_keys, ["electricity", "natural_gas"])
        self.assertEqual(session.custom_formula_status, "default")
        self.assertTrue(session.has_structured_data)
        self.assertEqual(session.report_generation_status, "ready")
        self.assertAlmostEqual(result["emission_factors_used"]["electricity"], 0.75)
        self.assertAlmostEqual(result["emission_factors_used"]["natural_gas"], 2.85)
        self.assertEqual(result["formula_status"], "default")

    def test_session_build_index_marks_custom_formula_incomplete_when_variable_is_missing(self):
        formula = ExtractedFormula(
            expression="electricity * electricity_factor + natural_gas * natural_gas_factor - renewable * renewable_offset",
            constants={"renewable_offset": 0.05},
            variable_hints={"renewable": "renewable energy generated in kWh"},
            confidence="high",
            source_text="Total = electricity * factor + natural_gas * factor - renewable * renewable_offset",
        )
        session = RetrievalSession()

        with patch("rag_retrieval.session.LLMFormulaExtractor") as extractor_cls:
            extractor = extractor_cls.return_value
            extractor.extract_from_documents.return_value = formula
            extractor.extract_from_text.return_value = None
            session.build_index(
                [
                    Path("sample_docs/custom_formula_methodology.pdf"),
                    Path("sample_docs/consumption_data.csv"),
                ]
            )

        result = next(
            engine.analyze_district("Kadikoy")
            for engine in session.data_engines.values()
            if engine is not None and engine.analyze_district("Kadikoy")
        )
        self.assertEqual(session.custom_formula_status, "incomplete")
        self.assertEqual(session.custom_formula_missing_variables, ["renewable"])
        self.assertTrue(session.has_structured_data)
        self.assertEqual(session.report_generation_status, "blocked_missing_formula_inputs")
        self.assertEqual(result["formula_status"], "custom_incomplete")
        self.assertIn("custom_formula_missing_variable_definition:renewable", result["warnings"])

    def test_session_build_index_carries_custom_formula_into_data_engines(self):
        formula = ExtractedFormula(
            expression="electricity * electricity_factor + natural_gas * natural_gas_factor - tree_count * tree_credit",
            constants={"tree_credit": 2.0},
            variable_hints={"tree_count": "number of trees"},
            confidence="high",
            source_text="Total = electricity * factor + natural_gas * factor - tree_count * tree_credit",
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            methodology = tmp_path / "methodology.md"
            methodology.write_text(
                "# Formula\nTotal emissions depend on electricity, natural gas, and tree_count.\n",
                encoding="utf-8",
            )
            dataset = tmp_path / "districts.csv"
            dataset.write_text(
                "district,electricity_kwh,natural_gas_m3,tree_count\nKadikoy,1000,100,10\n",
                encoding="utf-8",
            )

            session = RetrievalSession()
            with patch("rag_retrieval.session.LLMFormulaExtractor") as extractor_cls:
                extractor = extractor_cls.return_value
                extractor.extract_from_documents.return_value = formula
                extractor.extract_from_text.return_value = None
                session.build_index([methodology, dataset])

        result = next(
            engine.analyze_district("Kadikoy")
            for engine in session.data_engines.values()
            if engine is not None and engine.analyze_district("Kadikoy")
        )
        self.assertEqual(session.custom_formula_status, "valid")
        self.assertTrue(session.has_structured_data)
        self.assertEqual(session.report_generation_status, "ready")
        self.assertEqual(result["formula_status"], "custom_applied")
        self.assertEqual(result["formula_bound_variables"]["tree_count"], "column:tree_count")
        self.assertEqual(result["total_emission"], 620.0)

    def test_formula_only_document_blocks_report_generation_until_data_is_uploaded(self):
        formula = ExtractedFormula(
            expression="electricity * electricity_factor + natural_gas * natural_gas_factor",
            constants={},
            variable_hints={},
            confidence="high",
            source_text="Total CO2 = electricity * electricity_factor + natural_gas * natural_gas_factor",
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            methodology = tmp_path / "methodology.md"
            methodology.write_text(
                "# Formula\nTotal CO2 = electricity * electricity_factor + natural_gas * natural_gas_factor\n",
                encoding="utf-8",
            )

            session = RetrievalSession()
            with patch("rag_retrieval.session.LLMFormulaExtractor") as extractor_cls:
                extractor = extractor_cls.return_value
                extractor.extract_from_documents.return_value = formula
                extractor.extract_from_text.return_value = None
                session.build_index([methodology])

            self.assertEqual(session.custom_formula_status, "valid")
            self.assertFalse(session.has_structured_data)
            self.assertEqual(session.structured_district_count, 0)
            self.assertEqual(session.report_generation_status, "blocked_missing_structured_data")
            self.assertEqual(session.report_generation_warnings, ["custom_formula_detected_but_no_structured_data"])

            with self.assertRaisesRegex(ValueError, "no structured district data is available yet"):
                generate_sustainability_report(session, output_dir=tmp_path / "reports")

    def test_formula_only_pdf_uses_deterministic_extractor_without_openai_key(self):
        with patch.dict(environ, {"OPENAI_API_KEY": ""}, clear=False):
            session = RetrievalSession()
            session.build_index([Path("test_custom_formula_methodology.pdf")])

        self.assertIsNotNone(session.custom_formula)
        self.assertEqual(
            session.custom_formula.expression,
            "electricity * electricity_factor + natural_gas * natural_gas_factor - tree_count * tree_credit",
        )
        self.assertEqual(session.custom_formula.constants["tree_credit"], 0.08)
        self.assertEqual(session.report_generation_status, "blocked_missing_structured_data")

    def test_user_inputs_can_resolve_incomplete_formula_and_unlock_report_generation(self):
        formula = ExtractedFormula(
            expression="electricity * electricity_factor + natural_gas * natural_gas_factor - renewable * renewable_offset",
            constants={"renewable_offset": 0.05},
            variable_hints={"renewable": "renewable energy generated in kWh"},
            confidence="high",
            source_text="Total = electricity * electricity_factor + natural_gas * natural_gas_factor - renewable * renewable_offset",
        )
        session = RetrievalSession()

        with patch("rag_retrieval.session.LLMFormulaExtractor") as extractor_cls:
            extractor = extractor_cls.return_value
            extractor.extract_from_documents.return_value = formula
            extractor.extract_from_text.return_value = None
            session.build_index(
                [
                    Path("sample_docs/custom_formula_methodology.pdf"),
                    Path("sample_docs/consumption_data.csv"),
                ]
            )

        self.assertEqual(session.report_generation_status, "blocked_missing_formula_inputs")
        session.update_custom_formula_inputs({"renewable": {"type": "constant", "value": 400}})

        result = next(
            engine.analyze_district("Kadikoy")
            for engine in session.data_engines.values()
            if engine is not None and engine.analyze_district("Kadikoy")
        )
        self.assertEqual(session.custom_formula_status, "valid")
        self.assertEqual(session.report_generation_status, "ready")
        self.assertEqual(result["formula_status"], "custom_applied")
        self.assertEqual(result["formula_bound_variables"]["renewable"], "user_constant")
        self.assertEqual(result["total_emission"], 2550.0)

    def test_uploading_formula_after_data_rebuilds_existing_engines(self):
        formula = ExtractedFormula(
            expression="electricity * electricity_factor + natural_gas * natural_gas_factor - tree_count * tree_credit",
            constants={"tree_credit": 2.0},
            variable_hints={"tree_count": "number of trees"},
            confidence="high",
            source_text="Total = electricity * electricity_factor + natural_gas * natural_gas_factor - tree_count * tree_credit",
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            methodology = tmp_path / "methodology.md"
            methodology.write_text("# Formula\nTree count modifies the total.\n", encoding="utf-8")
            dataset = tmp_path / "districts.csv"
            dataset.write_text(
                "district,electricity_kwh,natural_gas_m3,tree_count\nKadikoy,1000,100,10\n",
                encoding="utf-8",
            )

            session = RetrievalSession()
            with patch("rag_retrieval.session.LLMFormulaExtractor") as extractor_cls:
                extractor = extractor_cls.return_value
                extractor.extract_from_documents.return_value = None
                extractor.extract_from_text.return_value = None
                session.build_index([dataset])

                extractor.extract_from_documents.return_value = formula
                session.build_index([methodology])

        result = next(
            engine.analyze_district("Kadikoy")
            for engine in session.data_engines.values()
            if engine is not None and engine.analyze_district("Kadikoy")
        )
        self.assertEqual(session.custom_formula_status, "valid")
        self.assertEqual(result["formula_status"], "custom_applied")
        self.assertEqual(result["total_emission"], 620.0)

    def test_conflicting_methodology_requires_explicit_resolution(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(environ, {"OPENAI_API_KEY": ""}, clear=False):
            tmp_path = Path(tmp)
            formula_a = tmp_path / "methodology_a.pdf"
            formula_b = tmp_path / "methodology_b.pdf"
            dataset = Path("test_ibb_39_districts.xlsx")

            _write_pdf(
                formula_a,
                [
                    "Total CO2 = electricity * electricity_factor + natural_gas * natural_gas_factor",
                    "electricity factor = 0.45 kgCO2e/kWh",
                    "natural gas factor = 2.04 kgCO2e/m3",
                ],
            )
            _write_pdf(
                formula_b,
                [
                    "Total CO2 = electricity * electricity_factor + natural_gas * natural_gas_factor + direct_emissions",
                    "electricity factor = 0.52 kgCO2e/kWh",
                    "natural gas factor = 2.18 kgCO2e/m3",
                ],
            )

            session = RetrievalSession()
            session.build_index([formula_a, formula_b, dataset])

        self.assertEqual(session.methodology_status, "needs_resolution")
        self.assertEqual(session.report_generation_status, "blocked_methodology_conflict")
        self.assertTrue(session.factor_conflicts)
        self.assertTrue(session.formula_conflicts)
        self.assertIsNone(session.custom_formula)

        formula_doc_id = next(
            candidate["doc_id"]
            for candidate in session.formula_conflicts[0]["candidates"]
            if candidate["filename"] == "methodology_b.pdf"
        )
        session.update_methodology_resolution(
            formula_doc_id=formula_doc_id,
            factor_doc_ids={
                "electricity": next(
                    candidate["doc_id"]
                    for candidate in next(conflict for conflict in session.factor_conflicts if conflict["metric_key"] == "electricity")["candidates"]
                    if candidate["filename"] == "methodology_b.pdf"
                ),
                "natural_gas": next(
                    candidate["doc_id"]
                    for candidate in next(conflict for conflict in session.factor_conflicts if conflict["metric_key"] == "natural_gas")["candidates"]
                    if candidate["filename"] == "methodology_b.pdf"
                ),
            },
        )

        self.assertEqual(session.methodology_status, "clear")
        self.assertEqual(session.report_generation_status, "ready")
        self.assertIsNotNone(session.custom_formula)
        self.assertIn("direct_emissions", session.custom_formula.expression)
        self.assertEqual(session.calculation_audit["formula"]["selected"]["filename"], "methodology_b.pdf")


if __name__ == "__main__":
    unittest.main()
