import unittest

from rag_retrieval.fact_extractor import FactExtractor
from rag_retrieval.models import ElementType, ParsedDocument, ParsedElement


class FactExtractorTests(unittest.TestCase):
    def test_extracts_gas_fact_from_pdf_paragraph(self):
        document = ParsedDocument(
            doc_id="doc-1",
            filename="report.pdf",
            elements=[
                ParsedElement(
                    element_id="p1",
                    doc_id="doc-1",
                    page=2,
                    element_type=ElementType.PARAGRAPH,
                    text="In 2024, natural gas consumption in Bakirkoy reached 1,343,469.02 m3.",
                )
            ],
            metadata={"parser": "pymupdf-layout"},
        )

        facts = FactExtractor().extract(document)

        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["district"], "Bakirkoy")
        self.assertEqual(facts[0]["year"], 2024)
        self.assertEqual(facts[0]["metric"], "natural_gas_consumption")
        self.assertEqual(facts[0]["natural_gas_consumption_m3"], 1343469.02)
        self.assertEqual(facts[0]["source_document"], "report.pdf")
        self.assertEqual(facts[0]["page"], 2)

    def test_extracts_electricity_fact_from_pdf_paragraph(self):
        document = ParsedDocument(
            doc_id="doc-1",
            filename="report.pdf",
            elements=[
                ParsedElement(
                    element_id="p1",
                    doc_id="doc-1",
                    page=1,
                    element_type=ElementType.PARAGRAPH,
                    text="Kadikoy electricity consumption was 4.5 million kWh in 2023.",
                )
            ],
            metadata={"parser": "pymupdf-layout"},
        )

        facts = FactExtractor().extract(document)

        self.assertEqual(facts[0]["district"], "Kadikoy")
        self.assertEqual(facts[0]["year"], 2023)
        self.assertEqual(facts[0]["metric"], "electricity_consumption")
        self.assertEqual(facts[0]["electricity_consumption_kwh"], 4_500_000.0)

    def test_extracts_turkish_number_format_from_pdf_paragraph(self):
        document = ParsedDocument(
            doc_id="doc-1",
            filename="report.pdf",
            elements=[
                ParsedElement(
                    element_id="p1",
                    doc_id="doc-1",
                    page=3,
                    element_type=ElementType.PARAGRAPH,
                    text="Bakirkoy ilçesinde 2024 yılında doğalgaz tüketimi 1.343.469,02 m3 olarak gerçekleşmiştir.",
                )
            ],
            metadata={"parser": "pymupdf-layout"},
        )

        facts = FactExtractor().extract(document)

        self.assertEqual(facts[0]["district"], "Bakirkoy")
        self.assertEqual(facts[0]["natural_gas_consumption_m3"], 1343469.02)

    def test_extracts_direct_emission_fact_from_pdf_paragraph(self):
        document = ParsedDocument(
            doc_id="doc-1",
            filename="report.pdf",
            elements=[
                ParsedElement(
                    element_id="p1",
                    doc_id="doc-1",
                    page=4,
                    element_type=ElementType.PARAGRAPH,
                    text="Kadikoy carbon footprint was 250 tCO2e in 2024.",
                )
            ],
            metadata={"parser": "pymupdf-layout"},
        )

        facts = FactExtractor().extract(document)

        self.assertEqual(facts[0]["metric"], "emissions")
        self.assertEqual(facts[0]["emissions"], 250.0)


if __name__ == "__main__":
    unittest.main()
