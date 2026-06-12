import unittest

from rag_retrieval.insight_engine import build_report_insights
from rag_retrieval.recommendation_engine import build_report_recommendations


class RecommendationEngineTests(unittest.TestCase):
    def test_build_report_recommendations_creates_district_commentary_and_actions(self):
        metrics = [
            {
                "district": "Bakirkoy",
                "total_emission": 1000.0,
                "gas_emission": 1000.0,
                "electricity_emission": 0.0,
                "direct_emissions": 0.0,
                "water_consumption": 4200.0,
                "water_growth": 0.10,
                "growth": 0.25,
                "warnings": ["electricity_consumption_not_found"],
                "metric_summaries": {
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
                    },
                    "tree_count": {
                        "metric_key": "tree_count",
                        "label": "Tree Count",
                        "category": "ecology",
                        "unit": "",
                        "role": "context_indicator",
                        "report_section": "District Context and Sustainability Signals",
                        "value": 12000.0,
                        "per_capita": None,
                        "growth": 0.05,
                        "sustainability_related": True,
                    },
                },
            }
        ]
        insights = build_report_insights(metrics, warnings=["report.pdf: electricity_consumption_not_found"])

        recommendations = build_report_recommendations(
            metrics,
            insights,
            warnings=["report.pdf: electricity_consumption_not_found"],
        )

        self.assertIn("priority_district_commentary", recommendations)
        self.assertTrue(recommendations["priority_district_commentary"])
        first = recommendations["priority_district_commentary"][0]
        self.assertEqual(first["district"], "Bakirkoy")
        self.assertEqual(first["archetype_key"], "multi_pressure_hotspot")
        self.assertTrue(first["recommended_actions_en"])
        self.assertIn("#1 of 1", first["headline_en"])
        self.assertIn("water demand", first["summary_en"])
        self.assertIn("Tree Count", first["summary_en"])
        self.assertIn("strategic_recommendations", recommendations)
        self.assertTrue(recommendations["strategic_recommendations"])
        notes = recommendations["data_quality_notes"]
        self.assertTrue(any("electricity" in (item.get("note_en") or "").lower() for item in notes))

    def test_priority_district_commentary_prefers_diverse_patterns(self):
        raw_rows = [
            ("Core", 1000.0, 0.18, 120.0, 1200.0),
            ("Core2", 950.0, 0.16, 30.0, 1100.0),
            ("Core3", 900.0, 0.15, 20.0, 900.0),
            ("Ecology", 850.0, 0.14, 15.0, 8000.0),
            ("Mid1", 650.0, 0.13, 10.0, 700.0),
            ("Mid2", 600.0, 0.12, 8.0, 650.0),
            ("Mid3", 550.0, 0.11, 7.0, 600.0),
            ("Mid4", 500.0, 0.10, 6.0, 550.0),
            ("Growth", 220.0, 0.40, 5.0, 500.0),
            ("Baseline", 120.0, 0.02, 0.0, 450.0),
        ]
        metrics = []
        for district, total_emission, growth, direct_emissions, tree_count in raw_rows:
            metric_summaries = {}
            if tree_count > 0.0:
                metric_summaries["tree_count"] = {
                    "metric_key": "tree_count",
                    "label": "Tree Count",
                    "category": "ecology",
                    "unit": "",
                    "role": "context_indicator",
                    "report_section": "District Context and Sustainability Signals",
                    "value": tree_count,
                }
            metrics.append(
                {
                    "district": district,
                    "total_emission": total_emission,
                    "gas_emission": total_emission * 0.32,
                    "electricity_emission": total_emission * 0.68,
                    "direct_emissions": direct_emissions,
                    "water_consumption": 0.0,
                    "growth": growth,
                    "warnings": [],
                    "metric_summaries": metric_summaries,
                }
            )
        insights = build_report_insights(metrics, warnings=[])

        recommendations = build_report_recommendations(metrics, insights, warnings=[])

        commentary = recommendations["priority_district_commentary"]
        angles = {item["commentary_angle"] for item in commentary}
        districts = {item["district"] for item in commentary}

        self.assertIn("Core", districts)
        self.assertIn("Growth", districts)
        self.assertGreaterEqual(len(angles), 3)

    def test_commentary_adds_comparative_grounding_clause(self):
        metrics = [
            _commentary_metric("High", 2000.0, per_capita=400.0),
            _commentary_metric("Low", 1000.0, per_capita=100.0),
        ]
        insights = build_report_insights(metrics)

        recommendations = build_report_recommendations(metrics, insights)
        summaries = " ".join(item["summary_en"] for item in recommendations["priority_district_commentary"])

        self.assertIn("municipal median", summaries)
        self.assertIn("per-capita intensity", summaries)

    def test_tree_count_size_proxy_replaces_naive_ecology_claim(self):
        # Tree count moves in lockstep with emissions, so the analytics layer
        # flags it; commentary must call it out as size, not an advantage.
        metrics = [
            _commentary_metric(f"D{i}", float(value), per_capita=50.0, tree_count=float(value))
            for i, value in enumerate([600, 500, 400, 300, 200, 100])
        ]
        insights = build_report_insights(metrics)

        recommendations = build_report_recommendations(metrics, insights)
        summaries = " ".join(item["summary_en"] for item in recommendations["priority_district_commentary"])

        self.assertIn("reflects district size", summaries)
        self.assertNotIn("Tree Count sits above the municipal median", summaries)

    def test_waste_and_recycling_are_framed_generically(self):
        # No hard-coding: a waste metric reads as pressure, a recycling metric
        # as an asset, with no tree_count anywhere in the dataset.
        metrics = []
        for i, district in enumerate(["Kadikoy", "Besiktas", "Sisli", "Uskudar", "Fatih", "Maltepe"]):
            metrics.append(
                _commentary_metric(
                    district,
                    float(600 - i * 40),
                    per_capita=50.0,
                    extras={
                        "waste": ("Waste", "waste", "resource_kpi", float(300 + (i % 3) * 150)),
                        "recycling_rate": ("Recycling Rate", "waste", "resource_kpi", float(80 - (i % 4) * 15)),
                    },
                )
            )
        insights = build_report_insights(metrics)

        recommendations = build_report_recommendations(metrics, insights)
        summaries = " ".join(item["summary_en"] for item in recommendations["priority_district_commentary"])

        self.assertIn("offsetting sustainability asset", summaries)  # recycling
        self.assertIn("resource pressure", summaries)  # waste
        self.assertNotIn("tree", summaries.lower())

    def test_strategic_recommendations_are_operational(self):
        metrics = [
            _commentary_metric("Adalar", 1000.0, per_capita=400.0),
            _commentary_metric("Sile", 900.0, per_capita=300.0),
            _commentary_metric("Catalca", 800.0, per_capita=200.0),
            _commentary_metric("Esenyurt", 200.0, per_capita=50.0),
        ]
        insights = build_report_insights(metrics)

        recommendations = build_report_recommendations(metrics, insights)
        core = next(
            item
            for item in recommendations["strategic_recommendations"]
            if "emissions core" in item["title_en"].lower()
        )

        self.assertTrue(core["instruments_en"])  # concrete measures, not a one-liner
        self.assertTrue(core["sequence_en"])  # has a timeframe/sequencing
        self.assertTrue(core["target_en"])  # has a measurable target
        # Lever-aware: electricity dominates this synthetic data.
        self.assertIn("electricity", core["rationale_en"].lower())


def _commentary_metric(district, total_emission, *, per_capita=None, tree_count=None, extras=None):
    metric_summaries = {}
    if tree_count is not None:
        metric_summaries["tree_count"] = {
            "metric_key": "tree_count",
            "label": "Tree Count",
            "category": "ecology",
            "unit": "",
            "role": "context_indicator",
            "report_section": "District Context and Sustainability Signals",
            "value": float(tree_count),
            "per_capita": None,
            "growth": None,
            "sustainability_related": True,
        }
    for metric_key, (label, category, role, value) in (extras or {}).items():
        section = "Resource Overview" if category in {"waste", "water"} else "District Context and Sustainability Signals"
        metric_summaries[metric_key] = {
            "metric_key": metric_key,
            "label": label,
            "category": category,
            "unit": "",
            "role": role,
            "report_section": section,
            "value": float(value),
            "per_capita": None,
            "growth": None,
            "sustainability_related": True,
        }
    return {
        "district": district,
        "total_emission": float(total_emission),
        "gas_emission": float(total_emission) * 0.25,
        "electricity_emission": float(total_emission) * 0.75,
        "direct_emissions": 0.0,
        "emission_unit": "kgCO2e",
        "per_capita": per_capita,
        "water_consumption": 0.0,
        "growth": None,
        "warnings": [],
        "metric_summaries": metric_summaries,
    }


if __name__ == "__main__":
    unittest.main()
