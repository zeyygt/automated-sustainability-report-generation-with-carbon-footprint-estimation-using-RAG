import unittest

from rag_retrieval.chunking import StructureAwareChunker
from rag_retrieval.models import ElementType, ParsedDocument, ParsedElement


class ChunkingTests(unittest.TestCase):
    def test_table_chunk_preserves_headers_and_key_values(self):
        document = ParsedDocument(
            doc_id="doc-1",
            filename="report.pdf",
            elements=[
                ParsedElement(
                    element_id="e1",
                    doc_id="doc-1",
                    page=3,
                    element_type=ElementType.HEADING,
                    text="GHG Emissions",
                    section_path=("GHG Emissions",),
                ),
                ParsedElement(
                    element_id="e2",
                    doc_id="doc-1",
                    page=3,
                    element_type=ElementType.TABLE,
                    text="Metric | 2023 | 2024 | Unit",
                    section_path=("GHG Emissions",),
                    metadata={
                        "headers": ("Metric", "2023", "2024", "Unit"),
                        "rows": (("Scope 1 emissions", "12,400", "11,900", "tCO2e"),),
                    },
                ),
            ],
        )

        chunks = StructureAwareChunker().chunk_document(document)

        self.assertEqual(len(chunks), 1)
        self.assertIn("Headers: Metric, 2023, 2024, Unit.", chunks[0].text)
        self.assertIn("Metric Scope 1 emissions; 2023: 12,400; 2024: 11,900; Unit: tCO2e.", chunks[0].text)
        self.assertEqual(chunks[0].section_path, ("GHG Emissions",))


if __name__ == "__main__":
    unittest.main()

