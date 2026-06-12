import unittest

from rag_retrieval.metric_semantics import metric_semantic_profile


class MetricSemanticsTests(unittest.TestCase):
    def _profile(self, metric_key, category, role):
        return metric_semantic_profile(
            category=category,
            role=role,
            metric_key=metric_key,
            label=metric_key.replace("_", " ").title(),
        )

    def test_emission_input_is_a_driver(self):
        profile = self._profile("electricity", "energy", "emission_input")
        self.assertEqual(profile["direction"], "higher_worse")
        self.assertEqual(profile["relation"], "driver")
        self.assertFalse(profile["asset"])

    def test_waste_is_a_pressure(self):
        profile = self._profile("waste", "waste", "resource_kpi")
        self.assertEqual(profile["direction"], "higher_worse")
        self.assertEqual(profile["relation"], "pressure")
        self.assertFalse(profile["asset"])

    def test_recycling_is_an_asset_despite_waste_category(self):
        profile = self._profile("recycling_rate", "waste", "resource_kpi")
        self.assertEqual(profile["direction"], "higher_better")
        self.assertTrue(profile["asset"])

    def test_vegetation_is_a_sink_asset(self):
        profile = self._profile("tree_count", "ecology", "context_indicator")
        self.assertEqual(profile["relation"], "sink_offset")
        self.assertTrue(profile["asset"])

    def test_renewable_energy_is_an_asset(self):
        profile = self._profile("renewable_share", "energy", "context_indicator")
        self.assertTrue(profile["asset"])

    def test_unknown_metric_defaults_to_neutral_context(self):
        profile = self._profile("dam_occupancy", "resilience", "context_indicator")
        self.assertEqual(profile["direction"], "neutral")
        self.assertEqual(profile["relation"], "context")
        self.assertTrue(profile["significance_en"])
        self.assertTrue(profile["significance_tr"])


if __name__ == "__main__":
    unittest.main()
