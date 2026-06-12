import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag_retrieval.llm_formula_extractor import ExtractedFormula
from rag_retrieval.pipeline import handle_query
from rag_retrieval.report_pipeline import generate_sustainability_report
from rag_retrieval.session import RetrievalSession


class AcceptanceFlowTests(unittest.TestCase):
    def _build_session(self, files, formula=None) -> RetrievalSession:
        session = RetrievalSession()
        with patch("rag_retrieval.session.LLMFormulaExtractor") as extractor_cls:
            extractor = extractor_cls.return_value
            extractor.extract_from_documents.return_value = formula
            extractor.extract_from_text.return_value = None
            session.build_index([Path(path) for path in files])
        return session

    @staticmethod
    def _live_engine(session: RetrievalSession):
        for engine in session.data_engines.values():
            if engine is not None:
                return engine
        raise AssertionError("Expected at least one live data engine")

    @staticmethod
    def _water_dataset_csv(tmp_path: Path) -> Path:
        csv_path = tmp_path / "water_metrics.csv"
        districts = [
            "Adalar", "Arnavutkoy", "Atasehir", "Avcilar", "Bagcilar", "Bahcelievler", "Bakirkoy", "Basaksehir",
            "Bayrampasa", "Besiktas", "Beykoz", "Beylikduzu", "Beyoglu", "Buyukcekmece", "Catalca", "Cekmekoy",
            "Esenler", "Esenyurt", "Eyupsultan", "Fatih", "Gaziosmanpasa", "Gungoren", "Kadikoy", "Kagithane",
            "Kartal", "Kucukcekmece", "Maltepe", "Pendik", "Sancaktepe", "Sariyer", "Silivri", "Sisli", "Sile",
            "Sultanbeyli", "Sultangazi", "Tuzla", "Umraniye", "Uskudar", "Zeytinburnu",
        ]
        rows = ["District,Year,Water Consumption (m3)"]
        for index, district in enumerate(districts, start=1):
            rows.append(f"{district},2023,{1000 + index * 10}")
            rows.append(f"{district},2024,{1100 + index * 10}")
        csv_path.write_text("\n".join(rows), encoding="utf-8")
        return csv_path

    @staticmethod
    def _generic_metric_dataset_csv(tmp_path: Path) -> Path:
        csv_path = tmp_path / "generic_sustainability_metrics.csv"
        rows = [
            "District,Year,Water Consumption (m3),Tree Count,Dam Occupancy (%),Recycling Rate (%)",
            "Kadikoy,2023,1000,42000,68,34",
            "Kadikoy,2024,1100,45000,72,37",
            "Besiktas,2023,700,17000,55,29",
            "Besiktas,2024,750,18000,58,31",
        ]
        csv_path.write_text("\n".join(rows), encoding="utf-8")
        return csv_path

    def test_factor_override_flow_supports_39_district_dataset_and_alias_query(self):
        session = self._build_session(
            [
                "test_factor_override_methodology.pdf",
                "test_ibb_39_districts.xlsx",
            ]
        )
        engine = self._live_engine(session)

        self.assertEqual(session.factor_override_keys, ["electricity", "natural_gas"])
        self.assertEqual(len(engine.districts()), 39)

        result = engine.analyze_district("Kadikoy")
        self.assertAlmostEqual(result["emission_factors_used"]["electricity"], 0.45)
        self.assertAlmostEqual(result["emission_factors_used"]["natural_gas"], 2.04)
        self.assertEqual(result["emission_factors_source"]["electricity"], "document")
        self.assertEqual(result["formula_status"], "default")

        query_result = handle_query("Küçük Çekmece electricity emissions", session)
        self.assertTrue(query_result["structured_results"])
        self.assertEqual(query_result["structured_results"][0]["data"]["district"], "Kucukcekmece")

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            report = generate_sustainability_report(
                session,
                title="Acceptance Flow Report",
                language="English",
                output_dir=tmp,
            )
            html = Path(report.html_path).read_text(encoding="utf-8")

        self.assertIn("District Appendix", html)
        self.assertIn("Adalar", html)
        self.assertIn("Esenyurt", html)

    def test_custom_formula_flow_applies_tree_count_override_on_acceptance_dataset(self):
        formula = ExtractedFormula(
            expression="electricity * electricity_factor + natural_gas * natural_gas_factor - tree_count * tree_credit",
            constants={"tree_credit": 0.08},
            variable_hints={"tree_count": "number of trees"},
            confidence="high",
            source_text="Total CO2 = electricity * electricity_factor + natural_gas * natural_gas_factor - tree_count * tree_credit",
        )
        session = self._build_session(
            [
                "test_custom_formula_methodology.pdf",
                "test_ibb_39_districts.xlsx",
            ],
            formula=formula,
        )
        engine = self._live_engine(session)
        result = engine.analyze_district("Kadikoy")

        self.assertEqual(session.custom_formula_status, "valid")
        self.assertEqual(result["formula_status"], "custom_applied")
        self.assertEqual(result["formula_bound_variables"]["tree_count"], "column:tree_count")
        self.assertAlmostEqual(result["total_emission"], 65252385.76)

    def test_incomplete_custom_formula_flow_requires_user_input_then_generates_report(self):
        formula = ExtractedFormula(
            expression="electricity * electricity_factor + natural_gas * natural_gas_factor - renewable * renewable_offset",
            constants={"renewable_offset": 0.05},
            variable_hints={"renewable": "renewable energy generated in kWh"},
            confidence="high",
            source_text="Total CO2 = electricity * electricity_factor + natural_gas * natural_gas_factor - renewable * renewable_offset",
        )
        session = self._build_session(
            [
                "test_incomplete_formula_methodology.pdf",
                "test_ibb_39_districts.xlsx",
            ],
            formula=formula,
        )

        self.assertEqual(session.custom_formula_status, "incomplete")
        self.assertEqual(session.custom_formula_missing_variables, ["renewable"])
        self.assertEqual(session.report_generation_status, "blocked_missing_formula_inputs")

        query_result = handle_query("Sile total emissions", session)
        self.assertTrue(query_result["structured_results"])
        data = query_result["structured_results"][0]["data"]
        self.assertEqual(data["district"], "Sile")
        self.assertEqual(data["formula_status"], "custom_incomplete")
        self.assertIn("custom_formula_missing_variable_definition:renewable", data["warnings"])

        session.update_custom_formula_inputs({"renewable": {"type": "constant", "value": 100000}})
        self.assertEqual(session.custom_formula_status, "valid")
        self.assertEqual(session.report_generation_status, "ready")

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            report = generate_sustainability_report(
                session,
                title="Acceptance Flow Report",
                language="English",
                output_dir=tmp,
            )
            html = Path(report.html_path).read_text(encoding="utf-8")

        self.assertEqual(len(report.report_input.structured_results), 39)
        self.assertNotIn("could not be fully applied", html)

    def test_water_only_upload_generates_district_results_and_water_section(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            tmp_path = Path(tmp)
            dataset = self._water_dataset_csv(tmp_path)
            session = self._build_session([dataset])

            self.assertEqual(session.report_generation_status, "ready")
            engine = self._live_engine(session)
            result = engine.analyze_district("Kadikoy")
            self.assertGreater(result["water_consumption"], 0.0)
            self.assertIn("water", result["metrics"])

            query_result = handle_query("Kadikoy water consumption", session)
            self.assertTrue(query_result["structured_results"])
            self.assertGreater(query_result["structured_results"][0]["data"]["water_consumption"], 0.0)

            report = generate_sustainability_report(
                session,
                title="Water Acceptance Report",
                language="English",
                output_dir=tmp_path / "reports",
            )
            html = Path(report.html_path).read_text(encoding="utf-8")

        self.assertIn("Water Overview", report.ai_content_markdown)
        self.assertIn("Municipality-Wide Assessment", report.ai_content_markdown)
        self.assertIn("Water Overview", html)

    def test_generic_metric_dataset_surfaces_tree_count_and_dam_occupancy_in_report(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            tmp_path = Path(tmp)
            dataset = self._generic_metric_dataset_csv(tmp_path)
            session = self._build_session([dataset])

            detected = {item["metric_key"]: item for item in session.detected_metrics}
            self.assertIn("tree_count", detected)
            self.assertIn("dam_occupancy", detected)
            self.assertTrue(detected["tree_count"]["sustainability_related"])
            self.assertEqual(detected["dam_occupancy"]["report_section"], "Water Overview")

            engine = self._live_engine(session)
            result = engine.analyze_district("Kadikoy")
            self.assertEqual(result["metrics"]["tree_count"]["value"], 87000.0)
            self.assertEqual(result["metrics"]["dam_occupancy"]["value"], 140.0)

            report = generate_sustainability_report(
                session,
                title="Generic Metric Report",
                language="English",
                output_dir=tmp_path / "reports",
            )

        self.assertIn("Municipality-Wide Assessment", report.ai_content_markdown)
        self.assertIn("District Context and Sustainability Signals", report.ai_content_markdown)
        self.assertIn("Priority Districts", report.ai_content_markdown)
        self.assertIn("Tree Count", report.ai_content_markdown)
        self.assertIn("Dam Occupancy", report.ai_content_markdown)


if __name__ == "__main__":
    unittest.main()
