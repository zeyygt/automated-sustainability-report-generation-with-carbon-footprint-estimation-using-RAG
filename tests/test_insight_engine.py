import unittest

from rag_retrieval.insight_engine import build_report_insights
from rag_retrieval.report_metrics import public_metrics


class InsightEngineTests(unittest.TestCase):
    def test_build_report_insights_surfaces_priority_districts_and_context(self):
        structured_results = [
            {
                "doc_id": "doc-1",
                "filename": "metrics.csv",
                "source_type": "spreadsheet",
                "parser": "spreadsheet",
                "data": {
                    "district": "Kadikoy",
                    "total_emission": 1200.0,
                    "gas_emission": 700.0,
                    "electricity_emission": 500.0,
                    "direct_emissions": 0.0,
                    "water_consumption": 6400.0,
                    "water_growth": 0.12,
                    "growth": 0.18,
                    "metrics": {
                        "water": {
                            "metric_key": "water",
                            "label": "Water",
                            "category": "water",
                            "unit": "m3",
                            "role": "resource_kpi",
                            "report_section": "Water Overview",
                            "value": 6400.0,
                            "growth": 0.12,
                        },
                        "tree_count": {
                            "metric_key": "tree_count",
                            "label": "Tree Count",
                            "category": "ecology",
                            "unit": "",
                            "role": "context_indicator",
                            "report_section": "District Context and Sustainability Signals",
                            "value": 45000.0,
                            "growth": 0.05,
                            "sustainability_related": True,
                        },
                    },
                    "warnings": [],
                },
            },
            {
                "doc_id": "doc-2",
                "filename": "metrics.csv",
                "source_type": "spreadsheet",
                "parser": "spreadsheet",
                "data": {
                    "district": "Besiktas",
                    "total_emission": 800.0,
                    "gas_emission": 450.0,
                    "electricity_emission": 350.0,
                    "direct_emissions": 0.0,
                    "water_consumption": 4200.0,
                    "water_growth": -0.04,
                    "growth": -0.08,
                    "metrics": {
                        "water": {
                            "metric_key": "water",
                            "label": "Water",
                            "category": "water",
                            "unit": "m3",
                            "role": "resource_kpi",
                            "report_section": "Water Overview",
                            "value": 4200.0,
                            "growth": -0.04,
                        }
                    },
                    "warnings": [],
                },
            },
        ]
        metrics = public_metrics(structured_results)
        detected_metrics = [
            {
                "metric_key": "water",
                "display_name": "Water",
                "sustainability_related": True,
            },
            {
                "metric_key": "tree_count",
                "display_name": "Tree Count",
                "sustainability_related": True,
            },
            {
                "metric_key": "electricity",
                "display_name": "Electricity",
                "sustainability_related": True,
            },
        ]

        insights = build_report_insights(
            metrics,
            warnings=["metrics.csv: electricity_consumption_not_found"],
            detected_metrics=detected_metrics,
        )

        self.assertEqual(insights["municipality"]["district_count"], 2)
        self.assertEqual(insights["municipality"]["highest_emission_district"]["district"], "Kadikoy")
        self.assertEqual(insights["coverage"]["level"], "moderate")
        self.assertTrue(insights["priority_districts"])
        self.assertTrue(insights["context_highlights"])
        self.assertEqual(insights["context_highlights"][0]["metric_key"], "tree_count")
        self.assertTrue(any("Priority districts" in line for line in insights["headlines"]))


if __name__ == "__main__":
    unittest.main()
