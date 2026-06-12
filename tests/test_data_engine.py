import unittest

from rag_retrieval.data_engine import DataEngine
from rag_retrieval.llm_formula_extractor import ExtractedFormula

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None


@unittest.skipUnless(pd is not None, "pandas is not installed")
class DataEngineTests(unittest.TestCase):
    def test_analyze_district_detects_turkish_columns(self):
        dataframe = pd.DataFrame(
            [
                {"Yıl": 2022, "Ay No": 1, "İlçe": "ADALAR", "Dogalgaz Tüketim Miktarı (m3)": "100"},
                {"Yıl": 2023, "Ay No": 1, "İlçe": "ADALAR", "Dogalgaz Tüketim Miktarı (m3)": "150"},
                {"Yıl": 2023, "Ay No": 1, "İlçe": "BAKIRKÖY", "Dogalgaz Tüketim Miktarı (m3)": "250"},
            ]
        )

        engine = DataEngine(dataframe)
        result = engine.analyze_district("adalar")

        self.assertEqual(engine.district_column, "ilce")
        self.assertEqual(engine.value_column, "dogalgaz_tuketim_miktari_m3")
        self.assertEqual(engine.time_column, "yil")
        self.assertEqual(result["district"], "ADALAR")
        self.assertEqual(result["electricity_emission"], 0.0)
        self.assertEqual(result["gas_emission"], 525.0)
        self.assertEqual(result["direct_emissions"], 0.0)
        self.assertEqual(result["total_emission"], 525.0)
        self.assertEqual(result["growth"], 0.5)
        self.assertAlmostEqual(result["per_capita"], 525.0 / 17489)
        self.assertAlmostEqual(result["per_household"], 525.0 / (17489 / 3.15))
        self.assertIn("electricity_consumption_not_found", result["warnings"])

    def test_analyze_district_handles_missing_data(self):
        engine = DataEngine(pd.DataFrame([{"District": "A", "Value": None}]))

        self.assertEqual(engine.analyze_district("B"), {})
        result = engine.analyze_district("A")
        self.assertEqual(result["district"], "A")
        self.assertEqual(result["electricity_emission"], 0.0)
        self.assertEqual(result["gas_emission"], 0.0)
        self.assertEqual(result["direct_emissions"], 0.0)
        self.assertEqual(result["total_emission"], 0.0)
        self.assertIsNone(result["per_capita"])
        self.assertIsNone(result["per_household"])
        self.assertIsNone(result["growth"])
        self.assertIn("no_consumption_or_emissions_detected", result["warnings"])
        self.assertIn("population_reference_not_found", result["warnings"])

    def test_analyze_district_uses_safe_growth_for_zero_first_value(self):
        dataframe = pd.DataFrame(
            [
                {"Year": 2022, "District": "A", "Consumption": 0},
                {"Year": 2023, "District": "A", "Consumption": 10},
            ]
        )

        result = DataEngine(dataframe).analyze_district("A")

        self.assertIsNone(result["growth"])

    def test_wide_pdf_table_is_normalized_to_long_format(self):
        dataframe = pd.DataFrame(
            [
                {
                    "District Name": "Bakirkoy",
                    "2022 Consumption (m3)": "100",
                    "2023 Consumption (m3)": "150",
                    "2024 Consumption (m3)": "200",
                }
            ]
        )

        engine = DataEngine(dataframe)
        result = engine.analyze_district("Bakirkoy")

        self.assertEqual(engine.district_column, "district_name")
        self.assertEqual(engine.time_column, "year")
        self.assertEqual(engine.gas_column, "natural_gas_consumption_m3")
        self.assertEqual(result["gas_emission"], 945.0)
        self.assertEqual(result["growth"], 1.0)

    def test_wide_pdf_table_with_metric_column_is_computable(self):
        dataframe = pd.DataFrame(
            [
                {"District": "Kadikoy", "Metric": "Electricity Consumption", "Unit": "kWh", "2023": "1000", "2024": "1500"},
                {"District": "Kadikoy", "Metric": "Natural Gas Consumption", "Unit": "m3", "2023": "100", "2024": "200"},
            ]
        )

        engine = DataEngine(dataframe)
        result = engine.analyze_district("Kadikoy")

        self.assertEqual(engine.time_column, "year")
        self.assertEqual(engine.metric_column, "metric")
        self.assertEqual(result["electricity_emission"], 1075.0)
        self.assertEqual(result["gas_emission"], 630.0)
        self.assertEqual(result["total_emission"], 1705.0)
        self.assertAlmostEqual(result["growth"], 600 / 1100)

    def test_direct_emission_fact_is_used_when_consumption_is_absent(self):
        dataframe = pd.DataFrame(
            [
                {"District": "Kadikoy", "Year": 2024, "Metric": "emissions", "Value": "250", "Unit": "tCO2e"},
            ]
        )

        result = DataEngine(dataframe).analyze_district("Kadikoy")

        self.assertEqual(result["electricity_emission"], 0.0)
        self.assertEqual(result["gas_emission"], 0.0)
        self.assertEqual(result["direct_emissions"], 250.0)
        self.assertEqual(result["total_emission"], 250.0)

    def test_water_metric_is_detected_from_wide_column_and_reported_per_capita(self):
        dataframe = pd.DataFrame(
            [
                {"District": "Kadikoy", "Year": 2022, "Water Consumption (m3)": 1200},
                {"District": "Kadikoy", "Year": 2024, "Water Consumption (m3)": 1500},
            ]
        )

        engine = DataEngine(dataframe)
        result = engine.analyze_district("Kadikoy")

        self.assertEqual(engine.water_column, "water_consumption_m3")
        self.assertEqual(result["water_consumption"], 2700.0)
        self.assertAlmostEqual(result["water_per_capita"], 2700.0 / 458573)
        self.assertAlmostEqual(result["water_growth"], 0.25)
        self.assertEqual(result["metrics"]["water"]["report_section"], "Water Overview")
        self.assertNotIn("no_consumption_or_emissions_detected", result["warnings"])

    def test_water_metric_is_detected_from_metric_value_table(self):
        dataframe = pd.DataFrame(
            [
                {"District": "Kadikoy", "Year": 2023, "Metric": "Water Consumption", "Unit": "m3", "Value": 1400},
                {"District": "Kadikoy", "Year": 2024, "Metric": "Water Consumption", "Unit": "m3", "Value": 1600},
            ]
        )

        result = DataEngine(dataframe).analyze_district("Kadikoy")

        self.assertEqual(result["water_consumption"], 3000.0)
        self.assertAlmostEqual(result["water_growth"], (1600 - 1400) / 1400)
        self.assertEqual(result["metrics"]["water"]["value"], 3000.0)

    def test_growth_uses_available_metric_for_mixed_fact_frames(self):
        dataframe = pd.DataFrame(
            [
                {"District": "Kadikoy", "Year": 2022, "Electricity Consumption kWh": "3800000"},
                {"District": "Kadikoy", "Year": 2023, "Electricity Consumption kWh": "4100000"},
                {"District": "Kadikoy", "Year": 2024, "Electricity Consumption kWh": "4500000"},
                {"District": "Besiktas", "Year": 2024, "Natural Gas Consumption m3": "1000"},
            ]
        )

        result = DataEngine(dataframe).analyze_district("Kadikoy")

        self.assertAlmostEqual(result["growth"], (4_500_000 - 3_800_000) / 3_800_000)

    def test_compare_districts_returns_top_and_lowest(self):
        dataframe = pd.DataFrame(
            [
                {"District": "A", "Consumption": 10},
                {"District": "A", "Consumption": 20},
                {"District": "B", "Consumption": 5},
            ]
        )

        result = DataEngine(dataframe).compare_districts()

        self.assertEqual(result["top_district"], "A")
        self.assertEqual(result["top_value"], 30.0)
        self.assertEqual(result["lowest_district"], "B")
        self.assertEqual(result["lowest_value"], 5.0)

    def test_custom_formula_applies_generic_column_binding(self):
        dataframe = pd.DataFrame(
            [
                {"District": "Kadikoy", "Electricity Consumption (kWh)": 1000, "Natural Gas Consumption (m3)": 100, "Tree Count": 10},
            ]
        )
        formula = ExtractedFormula(
            expression="electricity * electricity_factor + natural_gas * natural_gas_factor - tree_count * tree_credit",
            constants={"tree_credit": 2.0},
            variable_hints={"tree_count": "number of trees"},
            confidence="high",
            source_text="Total = electricity * factor + natural_gas * factor - tree_count * tree_credit",
        )

        result = DataEngine(dataframe, custom_formula=formula).analyze_district("Kadikoy")

        self.assertEqual(result["formula_status"], "custom_applied")
        self.assertEqual(result["formula_missing_variables"], [])
        self.assertEqual(result["formula_bound_variables"]["tree_count"], "column:tree_count")
        self.assertEqual(result["total_emission"], 620.0)

    def test_custom_formula_uses_metric_binding_when_column_is_not_present(self):
        dataframe = pd.DataFrame(
            [
                {"District": "Kadikoy", "Metric": "Electricity Consumption", "Unit": "kWh", "Value": 1000},
                {"District": "Kadikoy", "Metric": "Natural Gas Consumption", "Unit": "m3", "Value": 100},
                {"District": "Kadikoy", "Metric": "Renewable Energy Generated", "Unit": "kWh", "Value": 400},
            ]
        )
        formula = ExtractedFormula(
            expression="electricity * electricity_factor + natural_gas * natural_gas_factor - renewable * renewable_offset",
            constants={"renewable_offset": 0.05},
            variable_hints={"renewable": "renewable energy generated in kWh"},
            confidence="high",
            source_text="Total = electricity * factor + natural_gas * factor - renewable * renewable_offset",
        )

        result = DataEngine(dataframe, custom_formula=formula).analyze_district("Kadikoy")

        self.assertEqual(result["formula_status"], "custom_applied")
        self.assertEqual(result["formula_bound_variables"]["renewable"], "metric:renewable,energy,generated,kwh")
        self.assertEqual(result["total_emission"], 620.0)

    def test_custom_formula_incomplete_definition_falls_back_to_default(self):
        dataframe = pd.DataFrame(
            [
                {"District": "Kadikoy", "Electricity Consumption (kWh)": 1000, "Natural Gas Consumption (m3)": 100},
            ]
        )
        formula = ExtractedFormula(
            expression="electricity * electricity_factor + natural_gas * natural_gas_factor - renewable * renewable_offset",
            constants={"renewable_offset": 0.05},
            variable_hints={"renewable": "renewable energy generated in kWh"},
            confidence="high",
            source_text="Total = electricity * factor + natural_gas * factor - renewable * renewable_offset",
        )

        result = DataEngine(dataframe, custom_formula=formula).analyze_district("Kadikoy")

        self.assertEqual(result["formula_status"], "custom_incomplete")
        self.assertEqual(result["formula_missing_variables"], ["renewable"])
        self.assertIn("custom_formula_missing_variable_definition:renewable", result["warnings"])
        self.assertEqual(result["total_emission"], 640.0)

    def test_custom_formula_user_constant_resolves_missing_variable(self):
        dataframe = pd.DataFrame(
            [
                {"District": "Kadikoy", "Electricity Consumption (kWh)": 1000, "Natural Gas Consumption (m3)": 100},
            ]
        )
        formula = ExtractedFormula(
            expression="electricity * electricity_factor + natural_gas * natural_gas_factor - renewable * renewable_offset",
            constants={"renewable_offset": 0.05},
            variable_hints={"renewable": "renewable energy generated in kWh"},
            confidence="high",
            source_text="Total = electricity * factor + natural_gas * factor - renewable * renewable_offset",
        )

        result = DataEngine(
            dataframe,
            custom_formula=formula,
            custom_formula_inputs={"renewable": {"type": "constant", "value": 400}},
        ).analyze_district("Kadikoy")

        self.assertEqual(result["formula_status"], "custom_applied")
        self.assertEqual(result["formula_bound_variables"]["renewable"], "user_constant")
        self.assertEqual(result["total_emission"], 620.0)

    def test_custom_formula_user_column_mapping_overrides_automatic_miss(self):
        dataframe = pd.DataFrame(
            [
                {"District": "Kadikoy", "Electricity Consumption (kWh)": 1000, "Natural Gas Consumption (m3)": 100, "Tree Count": 10},
            ]
        )
        formula = ExtractedFormula(
            expression="electricity * electricity_factor + natural_gas * natural_gas_factor - carbon_sink * sink_credit",
            constants={"sink_credit": 2.0},
            variable_hints={"carbon_sink": "custom sink total"},
            confidence="high",
            source_text="Total = electricity * factor + natural_gas * factor - carbon_sink * sink_credit",
        )

        unresolved = DataEngine(dataframe, custom_formula=formula).analyze_district("Kadikoy")
        resolved = DataEngine(
            dataframe,
            custom_formula=formula,
            custom_formula_inputs={"carbon_sink": {"type": "column", "column": "tree_count"}},
        ).analyze_district("Kadikoy")

        self.assertEqual(unresolved["formula_status"], "custom_incomplete")
        self.assertEqual(resolved["formula_status"], "custom_applied")
        self.assertEqual(resolved["formula_bound_variables"]["carbon_sink"], "column:tree_count")
        self.assertEqual(resolved["total_emission"], 620.0)


if __name__ == "__main__":
    unittest.main()
