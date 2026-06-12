import os
import socket
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from urllib.request import urlopen

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover - optional local dependency
    sync_playwright = None


ROOT = Path(__file__).resolve().parents[1]


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server(url: str, timeout: float = 30.0) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urlopen(url) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.25)
    raise RuntimeError(f"Server did not start: {url}")


def _write_pdf(path: Path, lines: list[str]) -> None:
    pdf = canvas.Canvas(str(path), pagesize=A4)
    x = 72
    y = 800
    for line in lines:
        pdf.drawString(x, y, line)
        y -= 18
    pdf.save()


@unittest.skipIf(sync_playwright is None, "playwright is not installed")
class UiE2ETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.port = _find_free_port()
        env = dict(os.environ)
        env["OPENAI_API_KEY"] = ""
        cls.server = subprocess.Popen(
            [
                str(ROOT / ".venv" / "bin" / "python"),
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(cls.port),
            ],
            cwd=ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _wait_for_server(f"http://127.0.0.1:{cls.port}/")
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.browser.close()
        cls.playwright.stop()
        cls.server.terminate()
        try:
            cls.server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            cls.server.kill()

    def setUp(self) -> None:
        self.context = self.browser.new_context()
        self.page = self.context.new_page()
        self.page.goto(f"http://127.0.0.1:{self.port}/", wait_until="networkidle")

    def tearDown(self) -> None:
        self.context.close()

    def _upload(self, *paths: Path) -> None:
        self.page.set_input_files("#sidebar-file-input", [str(path) for path in paths])

    def test_factor_override_upload_and_report_generation(self):
        self._upload(ROOT / "test_factor_override_methodology.pdf", ROOT / "test_ibb_39_districts.xlsx")

        self.page.get_by_text("I've indexed", exact=False).wait_for(timeout=30000)
        self.assertEqual(self.page.locator("#generate-text").text_content().strip(), "Generate Report")

        self.page.locator("#generate-btn").click()
        self.page.get_by_text("39 districts analysed", exact=False).wait_for(timeout=120000)

        self.page.locator("#tab-insights").click()
        self.page.locator("#tab-content-insights").get_by_text("Strategic Recommendations", exact=False).wait_for(timeout=10000)
        self.page.locator("#tab-content-insights").get_by_text("District Commentary", exact=False).wait_for(timeout=10000)

        self.page.locator("#tab-audit").click()
        self.page.get_by_text("0.45", exact=False).wait_for(timeout=10000)
        self.page.get_by_text("2.04", exact=False).wait_for(timeout=10000)

    def test_formula_only_upload_blocks_until_data_exists(self):
        self._upload(ROOT / "test_custom_formula_methodology.pdf")

        self.page.get_by_text("Supporting data required", exact=False).wait_for(timeout=30000)
        self.assertEqual(self.page.locator("#generate-text").text_content().strip(), "Upload Supporting Data")

        self.page.locator("#generate-btn").click()
        self.page.locator("#session-alert-body").get_by_text("there is no district-level data", exact=False).wait_for(timeout=10000)

    def test_incomplete_formula_can_be_resolved_from_ui(self):
        self._upload(ROOT / "test_incomplete_formula_methodology.pdf", ROOT / "test_ibb_39_districts.xlsx")

        self.page.get_by_text("Formula inputs required", exact=False).wait_for(timeout=30000)
        self.assertEqual(self.page.locator("#generate-text").text_content().strip(), "Resolve Formula Inputs")

        self.page.locator("#generate-btn").click()
        self.page.locator("#formula-resolver-modal").wait_for(state="visible", timeout=10000)
        self.page.locator("#formula-value-renewable").fill("100000")
        self.page.locator("#formula-resolver-save").click()
        self.page.get_by_text("The missing formula inputs were saved.", exact=False).wait_for(timeout=10000)

        self.assertEqual(self.page.locator("#generate-text").text_content().strip(), "Generate Report")
        self.page.locator("#generate-btn").click()
        self.page.get_by_text("39 districts analysed", exact=False).wait_for(timeout=120000)

    def test_methodology_conflict_is_visible_and_resolvable(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            formula_a = tmp_path / "methodology_a.pdf"
            formula_b = tmp_path / "methodology_b.pdf"
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

            self._upload(formula_a, formula_b, ROOT / "test_ibb_39_districts.xlsx")

            self.page.get_by_text("Methodology conflict detected", exact=False).wait_for(timeout=30000)
            self.assertEqual(self.page.locator("#generate-text").text_content().strip(), "Resolve Methodology")

            self.page.locator("#generate-btn").click()
            self.page.locator("#methodology-resolver-modal").wait_for(state="visible", timeout=10000)
            self.page.locator("#methodology-formula-0").select_option(label="methodology_b.pdf")
            self.page.locator("#methodology-factor-electricity").select_option(label="methodology_b.pdf (0.52)")
            self.page.locator("#methodology-factor-natural_gas").select_option(label="methodology_b.pdf (2.18)")
            self.page.locator("#methodology-resolver-save").click()
            self.page.get_by_text("I saved the methodology resolution", exact=False).wait_for(timeout=10000)

            self.assertEqual(self.page.locator("#generate-text").text_content().strip(), "Generate Report")

    def test_detected_metrics_modal_allows_metric_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "generic_metrics.csv"
            dataset.write_text(
                "\n".join(
                    [
                        "District,Year,Water Consumption (m3),Tree Count,Dam Occupancy (%)",
                        "Kadikoy,2023,1000,42000,68",
                        "Kadikoy,2024,1100,45000,72",
                    ]
                ),
                encoding="utf-8",
            )

            self._upload(dataset)
            self.page.get_by_text("I've indexed", exact=False).wait_for(timeout=30000)
            self.page.locator("#review-metrics-btn").click()
            self.page.locator("#detected-metrics-modal").wait_for(state="visible", timeout=10000)
            self.page.get_by_text("Tree Count", exact=False).wait_for(timeout=10000)
            self.page.locator("#metric-sustainability-tree_count").select_option("false")
            self.page.locator("#detected-metrics-save").click()
            self.page.get_by_text("I saved the detected metric review", exact=False).wait_for(timeout=10000)


if __name__ == "__main__":
    unittest.main()
