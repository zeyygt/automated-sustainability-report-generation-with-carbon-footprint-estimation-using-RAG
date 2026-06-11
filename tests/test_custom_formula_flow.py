import tempfile
import unittest
from os import environ
from pathlib import Path
from unittest.mock import patch

from rag_retrieval.llm_formula_extractor import ExtractedFormula
from rag_retrieval.report_pipeline import generate_sustainability_report
from rag_retrieval.session import RetrievalSession


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


if __name__ == "__main__":
    unittest.main()
