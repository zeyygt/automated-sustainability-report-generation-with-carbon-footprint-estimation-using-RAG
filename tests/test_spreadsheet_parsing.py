import tempfile
import unittest
from pathlib import Path

from rag_retrieval.chunking import StructureAwareChunker
from rag_retrieval.ingestion import DocumentIngestor
from rag_retrieval.models import ElementType
from rag_retrieval.parsing import SpreadsheetParser

try:
    from openpyxl import Workbook
except ImportError:  # pragma: no cover
    Workbook = None


@unittest.skipUnless(Workbook is not None, "openpyxl is not installed")
class SpreadsheetParsingTests(unittest.TestCase):
    def test_xlsx_table_becomes_table_element_and_chunk(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "energy.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Energy"
            sheet.append(["Year", "District", "Consumption (m3)"])
            sheet.append([2024, "Bakirkoy", 1343469.02])
            sheet.append([2024, "Besiktas", 977821.33])
            workbook.save(path)

            document = DocumentIngestor().ingest_paths([path])[0]
            parsed = SpreadsheetParser().parse(document)
            chunks = StructureAwareChunker().chunk_document(parsed)

        table = next(element for element in parsed.elements if element.element_type == ElementType.TABLE)
        self.assertEqual(table.metadata["sheet_name"], "Energy")
        self.assertEqual(table.metadata["cell_range"], "A1:C3")
        self.assertEqual(table.metadata["headers"], ("Year", "District", "Consumption (m3)"))
        self.assertEqual(len(table.metadata["rows"]), 2)
        self.assertEqual(len(chunks), 1)
        self.assertIn("Sheet: Energy, range A1:C3", chunks[0].text)
        self.assertIn("District: Bakirkoy", chunks[0].text)


if __name__ == "__main__":
    unittest.main()

