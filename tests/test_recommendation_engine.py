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


if __name__ == "__main__":
    unittest.main()
