import unittest

from rag_retrieval.insight_engine import build_report_insights


def _metric(
    district,
    total_emission,
    *,
    electricity=0.0,
    gas=0.0,
    per_capita=None,
    growth=None,
    context=None,
):
    metric_summaries = {}
    for key, (label, value) in (context or {}).items():
        metric_summaries[key] = {
            "label": label,
            "value": value,
            "report_section": "District Context and Sustainability Signals",
            "role": "context_indicator",
        }
    return {
        "district": district,
        "total_emission": float(total_emission),
        "electricity_emission": float(electricity),
        "gas_emission": float(gas),
        "direct_emissions": 0.0,
        "emission_unit": "kgCO2e",
        "per_capita": per_capita,
        "growth": growth,
        "water_consumption": 0.0,
        "warnings": [],
        "metric_summaries": metric_summaries,
    }


class InsightAnalyticsTests(unittest.TestCase):
    def test_concentration_flags_a_dispersed_distribution(self):
        metrics = [_metric(f"D{i}", 100.0, electricity=80.0, gas=20.0) for i in range(15)]

        analytics = build_report_insights(metrics)["analytics"]

        self.assertEqual(analytics["concentration"]["classification"], "dispersed")
        self.assertAlmostEqual(analytics["concentration"]["top3_share"], 0.2, places=3)

    def test_concentration_flags_a_concentrated_distribution(self):
        metrics = [
            _metric("Big", 900.0, electricity=900.0),
            _metric("Small1", 50.0, electricity=50.0),
            _metric("Small2", 50.0, electricity=50.0),
        ]

        analytics = build_report_insights(metrics)["analytics"]

        self.assertEqual(analytics["concentration"]["classification"], "concentrated")

    def test_energy_lever_identifies_dominant_source(self):
        metrics = [_metric(f"D{i}", 100.0, electricity=75.0, gas=25.0) for i in range(5)]

        lever = build_report_insights(metrics)["analytics"]["energy_lever"]

        self.assertEqual(lever["dominant"], "electricity")
        self.assertAlmostEqual(lever["electricity_share"], 0.75, places=3)

    def test_intensity_spread_raises_plausibility_flag(self):
        metrics = [
            _metric("Tiny", 1000.0, electricity=1000.0, per_capita=5000.0),
            _metric("Mid", 800.0, electricity=800.0, per_capita=400.0),
            _metric("Big", 600.0, electricity=600.0, per_capita=20.0),
        ]

        insights = build_report_insights(metrics)
        intensity = insights["analytics"]["intensity"]
        flag_codes = {flag["code"] for flag in insights["coverage"]["plausibility_flags"]}

        self.assertTrue(intensity["available"])
        self.assertEqual(intensity["top"]["district"], "Tiny")
        self.assertGreater(intensity["spread_ratio"], 50)
        self.assertIn("implausible_intensity_spread", flag_codes)

    def test_context_metric_perfectly_correlated_is_flagged(self):
        metrics = [
            _metric(f"D{i}", value, electricity=value, context={"tree_count": ("Tree Count", value)})
            for i, value in enumerate([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
        ]

        insights = build_report_insights(metrics)
        correlations = insights["analytics"]["correlations"]
        flag_codes = {flag["code"] for flag in insights["coverage"]["plausibility_flags"]}

        self.assertTrue(correlations)
        self.assertTrue(correlations[0]["strong"])
        self.assertAlmostEqual(correlations[0]["coefficient"], 1.0, places=3)
        self.assertIn("spurious_context_correlation", flag_codes)

    def test_implausible_growth_is_flagged(self):
        metrics = [_metric(f"D{i}", 100.0, electricity=100.0, growth=1.4) for i in range(6)]

        insights = build_report_insights(metrics)
        flag_codes = {flag["code"] for flag in insights["coverage"]["plausibility_flags"]}

        self.assertIn("implausible_growth", flag_codes)

    def test_analytical_findings_are_emitted(self):
        metrics = [_metric(f"D{i}", 100.0, electricity=75.0, gas=25.0, per_capita=50.0) for i in range(5)]

        findings = build_report_insights(metrics)["analytical_findings"]
        ids = {finding["id"] for finding in findings}

        self.assertIn("concentration", ids)
        self.assertIn("energy_lever", ids)


if __name__ == "__main__":
    unittest.main()
