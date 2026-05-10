import unittest

from rag_retrieval.formula_extractor import FormulaExtractor
from rag_retrieval.models import ElementType, ParsedDocument, ParsedElement

try:
    import pandas as pd
except ImportError:
    pd = None


def _doc(text: str, doc_id: str = "doc-1") -> ParsedDocument:
    return ParsedDocument(
        doc_id=doc_id,
        filename="report.pdf",
        elements=[
            ParsedElement(
                element_id="p1",
                doc_id=doc_id,
                page=1,
                element_type=ElementType.PARAGRAPH,
                text=text,
            )
        ],
        metadata={"parser": "pymupdf-layout"},
    )


class FormulaExtractorTextTests(unittest.TestCase):
    def setUp(self):
        self.extractor = FormulaExtractor()

    def test_extracts_electricity_factor_from_sentence(self):
        doc = _doc("The electricity emission factor used in this report is 0.52 kgCO2/kWh.")
        result = self.extractor.extract_from_document(doc)
        self.assertAlmostEqual(result["electricity"], 0.52)

    def test_extracts_natural_gas_factor_from_sentence(self):
        doc = _doc("Natural gas emission factor applied: 2.34 kgCO2/m3.")
        result = self.extractor.extract_from_document(doc)
        self.assertAlmostEqual(result["natural_gas"], 2.34)

    def test_extracts_both_factors_from_same_paragraph(self):
        doc = _doc(
            "The electricity emission factor is 0.43 kgCO2/kWh "
            "and the natural gas emission factor is 2.1 kgCO2/m3."
        )
        result = self.extractor.extract_from_document(doc)
        self.assertAlmostEqual(result["electricity"], 0.43)
        self.assertAlmostEqual(result["natural_gas"], 2.1)

    def test_extracts_reverse_order_factor_declaration(self):
        # "emission factor for electricity: 0.47"
        doc = _doc("The emission factor for electricity consumption is 0.47.")
        result = self.extractor.extract_from_document(doc)
        self.assertAlmostEqual(result["electricity"], 0.47)

    def test_ignores_implausible_values(self):
        # 500 is outside the plausible range (0.001–50)
        doc = _doc("The electricity emission factor is 500 kgCO2/kWh.")
        result = self.extractor.extract_from_document(doc)
        self.assertNotIn("electricity", result)

    def test_returns_empty_dict_when_no_factor_found(self):
        doc = _doc("Bakirkoy consumed 1,200 kWh of electricity in 2023.")
        result = self.extractor.extract_from_document(doc)
        self.assertEqual(result, {})

    def test_extracts_comma_decimal_format(self):
        doc = _doc("Electricity emission factor: 0,43 kgCO2/kWh.")
        result = self.extractor.extract_from_document(doc)
        self.assertAlmostEqual(result["electricity"], 0.43)

    def test_turkish_variant(self):
        doc = _doc("Elektrik emisyon faktörü 0.52 kgCO2/kWh olarak alınmıştır.")
        result = self.extractor.extract_from_document(doc)
        self.assertAlmostEqual(result["electricity"], 0.52)


@unittest.skipUnless(pd is not None, "pandas is not installed")
class FormulaExtractorSpreadsheetTests(unittest.TestCase):
    def setUp(self):
        self.extractor = FormulaExtractor()

    def test_extracts_factors_from_row_with_factor_keyword(self):
        df = pd.DataFrame([
            {"type": "electricity emission factor", "value": 0.47},
            {"type": "natural gas emission factor", "value": 2.05},
        ])
        result = self.extractor.extract_from_dataframe(df)
        self.assertAlmostEqual(result["electricity"], 0.47)
        self.assertAlmostEqual(result["natural_gas"], 2.05)

    def test_extracts_from_column_header_with_row_index_as_energy_type(self):
        df = pd.DataFrame(
            {"emission factor": [0.52, 2.1]},
            index=["electricity", "natural gas"],
        )
        result = self.extractor.extract_from_dataframe(df)
        self.assertAlmostEqual(result["electricity"], 0.52)
        self.assertAlmostEqual(result["natural_gas"], 2.1)

    def test_returns_empty_for_dataframe_with_no_factor_rows(self):
        df = pd.DataFrame([
            {"district": "Kadikoy", "electricity_kwh": 5000},
            {"district": "Bakirkoy", "electricity_kwh": 3000},
        ])
        result = self.extractor.extract_from_dataframe(df)
        self.assertEqual(result, {})

    def test_returns_empty_for_none_dataframe(self):
        result = self.extractor.extract_from_dataframe(None)
        self.assertEqual(result, {})

    def test_spreadsheet_overrides_text_in_extract(self):
        doc = _doc("The electricity emission factor is 0.43 kgCO2/kWh.")
        df = pd.DataFrame([{"type": "electricity emission factor", "value": 0.60}])
        result = self.extractor.extract(doc, dataframe=df)
        # spreadsheet value wins
        self.assertAlmostEqual(result["electricity"], 0.60)


@unittest.skipUnless(pd is not None, "pandas is not installed")
class DataEngineEmissionFactorOverrideTests(unittest.TestCase):
    def test_document_factor_overrides_reference_for_electricity(self):
        from rag_retrieval.data_engine import DataEngine

        df = pd.DataFrame([
            {"district": "Kadikoy", "electricity_kwh": 1000, "gas_m3": 0},
        ])
        engine = DataEngine(df, emission_factors={"electricity": 0.99})

        self.assertAlmostEqual(engine.emission_factors["electricity"], 0.99)
        # natural_gas falls back to reference_Data.json default (2.1)
        self.assertAlmostEqual(engine.emission_factors["natural_gas"], 2.1)

    def test_emission_factors_source_tracks_origin(self):
        from rag_retrieval.data_engine import DataEngine

        df = pd.DataFrame([{"district": "Kadikoy", "electricity_kwh": 100}])
        engine = DataEngine(df, emission_factors={"electricity": 0.55})

        self.assertEqual(engine.emission_factors_source["electricity"], "document")
        self.assertEqual(engine.emission_factors_source["natural_gas"], "reference")

    def test_analyze_district_result_includes_factors_used(self):
        from rag_retrieval.data_engine import DataEngine

        df = pd.DataFrame([
            {"district": "Kadikoy", "electricity_kwh": 1000},
        ])
        engine = DataEngine(df, emission_factors={"electricity": 0.80})
        result = engine.analyze_district("Kadikoy")

        self.assertIn("emission_factors_used", result)
        self.assertIn("emission_factors_source", result)
        self.assertAlmostEqual(result["emission_factors_used"]["electricity"], 0.80)
        self.assertEqual(result["emission_factors_source"]["electricity"], "document")

    def test_no_override_uses_reference_defaults(self):
        from rag_retrieval.data_engine import DataEngine

        df = pd.DataFrame([{"district": "Kadikoy", "electricity_kwh": 1000}])
        engine = DataEngine(df)

        self.assertAlmostEqual(engine.emission_factors["electricity"], 0.43)
        self.assertAlmostEqual(engine.emission_factors["natural_gas"], 2.1)
        self.assertEqual(engine.emission_factors_source["electricity"], "reference")


if __name__ == "__main__":
    unittest.main()
