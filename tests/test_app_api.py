import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from app.main import SESSION_ID_HEADER, app


ROOT = Path(__file__).resolve().parents[1]


def _write_pdf(path: Path, lines: list[str]) -> None:
    pdf = canvas.Canvas(str(path), pagesize=A4)
    x = 72
    y = 800
    for line in lines:
        pdf.drawString(x, y, line)
        y -= 18
    pdf.save()


class AppApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        self.session_ids: list[str] = []

    def tearDown(self) -> None:
        for session_id in self.session_ids:
            self.client.delete("/session", headers={SESSION_ID_HEADER: session_id})

    def _create_session(self) -> str:
        response = self.client.post("/session")
        self.assertEqual(response.status_code, 200)
        session_id = response.json()["session_id"]
        self.session_ids.append(session_id)
        return session_id

    def _upload_paths(self, session_id: str, paths: list[Path]) -> None:
        files = []
        for path in paths:
            media_type = "application/pdf" if path.suffix.lower() == ".pdf" else "application/octet-stream"
            files.append(("files", (path.name, path.read_bytes(), media_type)))
        response = self.client.post("/upload", headers={SESSION_ID_HEADER: session_id}, files=files)
        self.assertEqual(response.status_code, 200, response.text)

    @staticmethod
    def _write_csv(path: Path, lines: list[str]) -> None:
        path.write_text("\n".join(lines), encoding="utf-8")

    def test_sessions_are_isolated(self):
        session_a = self._create_session()
        session_b = self._create_session()

        self._upload_paths(session_a, [ROOT / "test_custom_formula_methodology.pdf"])

        status_a = self.client.get("/status", headers={SESSION_ID_HEADER: session_a})
        status_b = self.client.get("/status", headers={SESSION_ID_HEADER: session_b})
        self.assertEqual(status_a.status_code, 200)
        self.assertEqual(status_b.status_code, 200)
        self.assertEqual(len(status_a.json()["files"]), 1)
        self.assertEqual(status_b.json()["files"], [])

        factors_a = self.client.get("/emission-factors", headers={SESSION_ID_HEADER: session_a}).json()
        factors_b = self.client.get("/emission-factors", headers={SESSION_ID_HEADER: session_b}).json()
        self.assertEqual(factors_a["report_generation_status"], "blocked_missing_structured_data")
        self.assertEqual(factors_b["report_generation_status"], "ready")
        self.assertIsNotNone(factors_a["custom_formula"])
        self.assertIsNone(factors_b["custom_formula"])

    def test_formula_only_upload_blocks_report_generation(self):
        session_id = self._create_session()

        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            self._upload_paths(session_id, [ROOT / "test_custom_formula_methodology.pdf"])

        diagnostics = self.client.get("/emission-factors", headers={SESSION_ID_HEADER: session_id})
        self.assertEqual(diagnostics.status_code, 200)
        payload = diagnostics.json()
        self.assertEqual(payload["report_generation_status"], "blocked_missing_structured_data")
        self.assertEqual(payload["structured_district_count"], 0)
        self.assertEqual(payload["custom_formula_status"], "valid")

        report = self.client.post(
            "/generate-report",
            headers={SESSION_ID_HEADER: session_id},
            json={"title": "Blocked", "language": "English"},
        )
        self.assertEqual(report.status_code, 400)
        self.assertIn("no structured district data", report.json()["detail"].lower())

    def test_methodology_conflict_requires_resolution_then_unblocks(self):
        session_id = self._create_session()

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            tmp_path = Path(tmp)
            formula_a = tmp_path / "methodology_a.pdf"
            formula_b = tmp_path / "methodology_b.pdf"
            dataset = ROOT / "test_ibb_39_districts.xlsx"

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

            self._upload_paths(session_id, [formula_a, formula_b, dataset])

            diagnostics = self.client.get("/emission-factors", headers={SESSION_ID_HEADER: session_id})
            self.assertEqual(diagnostics.status_code, 200)
            payload = diagnostics.json()
            self.assertEqual(payload["report_generation_status"], "blocked_methodology_conflict")
            self.assertEqual(payload["methodology_status"], "needs_resolution")
            self.assertTrue(payload["factor_conflicts"])
            self.assertTrue(payload["formula_conflicts"])

            formula_doc_id = next(
                candidate["doc_id"]
                for candidate in payload["formula_conflicts"][0]["candidates"]
                if "direct_emissions" in (candidate["expression"] or "")
            )
            electricity_conflict = next(conflict for conflict in payload["factor_conflicts"] if conflict["metric_key"] == "electricity")
            electricity_doc_id = next(
                candidate["doc_id"]
                for candidate in electricity_conflict["candidates"]
                if candidate["filename"] == "methodology_b.pdf"
            )
            gas_conflict = next(conflict for conflict in payload["factor_conflicts"] if conflict["metric_key"] == "natural_gas")
            gas_doc_id = next(
                candidate["doc_id"]
                for candidate in gas_conflict["candidates"]
                if candidate["filename"] == "methodology_b.pdf"
            )

            resolved = self.client.post(
                "/methodology-resolution",
                headers={SESSION_ID_HEADER: session_id},
                json={
                    "formula_doc_id": formula_doc_id,
                    "factor_doc_ids": {
                        "electricity": electricity_doc_id,
                        "natural_gas": gas_doc_id,
                    },
                },
            )
            self.assertEqual(resolved.status_code, 200, resolved.text)
            resolved_payload = resolved.json()
            self.assertEqual(resolved_payload["report_generation_status"], "ready")
            self.assertEqual(resolved_payload["methodology_status"], "clear")
            self.assertEqual(
                resolved_payload["calculation_audit"]["formula"]["selected"]["filename"],
                "methodology_b.pdf",
            )
            self.assertEqual(
                resolved_payload["calculation_audit"]["factors"]["selected"]["electricity"]["filename"],
                "methodology_b.pdf",
            )

            report = self.client.post(
                "/generate-report",
                headers={SESSION_ID_HEADER: session_id},
                json={"title": "Resolved", "language": "English"},
            )
            self.assertEqual(report.status_code, 200, report.text)
            report_payload = report.json()
            self.assertEqual(len(report_payload["metrics"]), 39)
            self.assertIn("insights", report_payload)
            self.assertIn("municipality", report_payload["insights"])
            self.assertIn("recommendations", report_payload)
            self.assertIn("priority_district_commentary", report_payload["recommendations"])

    def test_detected_metrics_are_exposed_and_can_be_reclassified(self):
        session_id = self._create_session()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset = tmp_path / "generic_metrics.csv"
            self._write_csv(
                dataset,
                [
                    "District,Year,Water Consumption (m3),Tree Count,Dam Occupancy (%)",
                    "Kadikoy,2023,1000,42000,68",
                    "Kadikoy,2024,1100,45000,72",
                ],
            )
            self._upload_paths(session_id, [dataset])

        diagnostics = self.client.get("/emission-factors", headers={SESSION_ID_HEADER: session_id})
        self.assertEqual(diagnostics.status_code, 200)
        payload = diagnostics.json()
        detected = {item["metric_key"]: item for item in payload["detected_metrics"]}
        self.assertIn("tree_count", detected)
        self.assertIn("dam_occupancy", detected)
        self.assertTrue(detected["tree_count"]["sustainability_related"])

        updated = self.client.post(
            "/detected-metrics",
            headers={SESSION_ID_HEADER: session_id},
            json={
                "metrics": {
                    "tree_count": {
                        "sustainability_related": False,
                        "category": "other",
                        "role": "context_indicator",
                        "report_section": "District Context and Sustainability Signals",
                    }
                }
            },
        )
        self.assertEqual(updated.status_code, 200, updated.text)
        updated_payload = updated.json()
        tree_metric = next(item for item in updated_payload["detected_metrics"] if item["metric_key"] == "tree_count")
        self.assertFalse(tree_metric["sustainability_related"])
        self.assertEqual(tree_metric["classification_source"], "user")

    def test_new_upload_invalidates_previous_report_artifacts(self):
        session_id = self._create_session()

        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            self._upload_paths(session_id, [ROOT / "test_ibb_39_districts.xlsx"])

            generated = self.client.post(
                "/generate-report",
                headers={SESSION_ID_HEADER: session_id},
                json={"title": "First", "language": "English"},
            )
            self.assertEqual(generated.status_code, 200, generated.text)

            report_pdf = self.client.get("/report/pdf", headers={SESSION_ID_HEADER: session_id})
            self.assertEqual(report_pdf.status_code, 200)

            self._upload_paths(session_id, [ROOT / "test_factor_override_methodology.pdf"])

            stale_report_pdf = self.client.get("/report/pdf", headers={SESSION_ID_HEADER: session_id})
            self.assertEqual(stale_report_pdf.status_code, 404)

    def test_llm_metric_override_cannot_move_tree_count_out_of_context_section(self):
        session_id = self._create_session()

        with patch("rag_retrieval.session.suggest_metric_overrides") as suggest_overrides, patch.dict(
            os.environ, {"OPENAI_API_KEY": "", "ENABLE_LLM_METRIC_INTERPRETATION": "true"}, clear=False
        ):
            suggest_overrides.return_value = {
                "tree_count": {
                    "metric_key": "tree_count",
                    "display_name": "Tree Count",
                    "category": "environmental",
                    "role": "offset_or_sink",
                    "report_section": "Resource Overview",
                    "sustainability_related": True,
                    "classification_source": "llm",
                }
            }
            self._upload_paths(session_id, [ROOT / "test_ibb_39_districts.xlsx"])

            diagnostics = self.client.get("/emission-factors", headers={SESSION_ID_HEADER: session_id})
            self.assertEqual(diagnostics.status_code, 200)
            detected_tree = next(
                item for item in diagnostics.json()["detected_metrics"] if item["metric_key"] == "tree_count"
            )
            self.assertEqual(detected_tree["report_section"], "District Context and Sustainability Signals")
            self.assertEqual(detected_tree["role"], "context_indicator")
            self.assertEqual(detected_tree["classification_source"], "heuristic")

            report = self.client.post(
                "/generate-report",
                headers={SESSION_ID_HEADER: session_id},
                json={"title": "Tree Context", "language": "English"},
            )
            self.assertEqual(report.status_code, 200, report.text)
            markdown = report.json()["markdown"]
            self.assertIn("# District Context and Sustainability Signals", markdown)
            self.assertIn("Tree Count", markdown)


if __name__ == "__main__":
    unittest.main()
