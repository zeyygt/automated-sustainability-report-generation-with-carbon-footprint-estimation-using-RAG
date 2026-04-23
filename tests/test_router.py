import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rag_retrieval.models import DocumentInput, ElementType, ParsedDocument, ParsedElement
from rag_retrieval.router import parsed_tables_to_dataframe, route_query, spreadsheet_to_dataframe
from rag_retrieval.session import RetrievalSession


class RouterTests(unittest.TestCase):
    def test_route_query_prefers_excel_for_analytical_or_numeric(self):
        analytical = SimpleNamespace(intents=("analytical",), source_hints=("pdf",))
        numeric = SimpleNamespace(intents=("numeric",), source_hints=())

        self.assertEqual(route_query(analytical), "excel")
        self.assertEqual(route_query(numeric), "excel")

    def test_route_query_uses_source_hints_then_hybrid(self):
        pdf = SimpleNamespace(intents=(), source_hints=("pdf", "spreadsheet"))
        spreadsheet = SimpleNamespace(intents=(), source_hints=("spreadsheet",))
        hybrid = SimpleNamespace(intents=(), source_hints=())

        self.assertEqual(route_query(pdf), "pdf")
        self.assertEqual(route_query(spreadsheet), "excel")
        self.assertEqual(route_query(hybrid), "hybrid")


class SpreadsheetDataFrameTests(unittest.TestCase):
    def test_spreadsheet_to_dataframe_combines_table_elements(self):
        try:
            import pandas  # noqa: F401
        except ImportError:
            self.skipTest("pandas is not installed")

        document = ParsedDocument(
            doc_id="doc-1",
            filename="metrics.xlsx",
            elements=[
                ParsedElement(
                    element_id="t1",
                    doc_id="doc-1",
                    page=1,
                    element_type=ElementType.TABLE,
                    text="",
                    metadata={"headers": ("Year", "Value"), "rows": (("2023", "10"),)},
                ),
                ParsedElement(
                    element_id="t2",
                    doc_id="doc-1",
                    page=1,
                    element_type=ElementType.TABLE,
                    text="",
                    metadata={"headers": ("Year", "Value"), "rows": (("2024", "12"),)},
                ),
            ],
            metadata={"parser": "spreadsheet"},
        )

        frame = spreadsheet_to_dataframe(document)

        self.assertEqual(list(frame.columns), ["Year", "Value"])
        self.assertEqual(frame.shape, (2, 2))
        self.assertEqual(frame.iloc[1]["Value"], "12")

    def test_spreadsheet_to_dataframe_handles_missing_tables(self):
        try:
            import pandas  # noqa: F401
        except ImportError:
            self.skipTest("pandas is not installed")

        document = ParsedDocument("doc-1", "empty.xlsx", [], {"parser": "spreadsheet"})

        frame = spreadsheet_to_dataframe(document)

        self.assertTrue(frame.empty)


class SessionSpreadsheetSplitTests(unittest.TestCase):
    def test_build_index_excludes_spreadsheets_from_vector_pipeline(self):
        class FakeParser:
            def parse(self, document: DocumentInput) -> ParsedDocument:
                if document.path.suffix == ".xlsx":
                    return ParsedDocument(
                        document.doc_id,
                        document.filename,
                        [
                            ParsedElement(
                                element_id="table-1",
                                doc_id=document.doc_id,
                                page=1,
                                element_type=ElementType.TABLE,
                                text="",
                                metadata={"headers": ("Year",), "rows": (("2024",),)},
                            )
                        ],
                        {"parser": "spreadsheet"},
                    )
                return ParsedDocument(
                    document.doc_id,
                    document.filename,
                    [
                        ParsedElement(
                            element_id="paragraph-1",
                            doc_id=document.doc_id,
                            page=1,
                            element_type=ElementType.PARAGRAPH,
                            text="Annual natural gas consumption table.",
                        )
                    ],
                    {"parser": "pymupdf-layout"},
                )

        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "report.pdf"
            xlsx_path = Path(tmp) / "metrics.xlsx"
            pdf_path.write_text("placeholder", encoding="utf-8")
            xlsx_path.write_text("placeholder", encoding="utf-8")

            session = RetrievalSession(parser=FakeParser())
            fake_dataframe = SimpleNamespace(empty=False)
            fake_engine = object()
            with patch("rag_retrieval.session.parsed_tables_to_dataframe", return_value=fake_dataframe):
                with patch("rag_retrieval.session.combine_dataframes", return_value=fake_dataframe):
                    with patch("rag_retrieval.session.FactExtractor") as extractor_cls:
                        extractor_cls.return_value.to_dataframe.return_value = None
                        with patch("rag_retrieval.session.DataEngine", return_value=fake_engine):
                            stats = session.build_index([pdf_path, xlsx_path])

        self.assertEqual(stats.document_count, 2)
        self.assertEqual(stats.chunk_count, 1)
        self.assertEqual(stats.embedding_count, 1)
        self.assertEqual(len(session.pdf_documents), 1)
        self.assertEqual(len(session.excel_documents), 1)
        self.assertEqual(len(session.spreadsheet_dataframes), 1)
        self.assertEqual(len(session.table_dataframes), 2)
        self.assertEqual(len(session.fact_dataframes), 2)
        self.assertEqual(list(session.data_engines.values()), [fake_engine, fake_engine])
        self.assertEqual(len(session.chunks), 1)

    def test_parsed_tables_to_dataframe_accepts_pdf_tables(self):
        try:
            import pandas  # noqa: F401
        except ImportError:
            self.skipTest("pandas is not installed")

        document = ParsedDocument(
            doc_id="doc-1",
            filename="report.pdf",
            elements=[
                ParsedElement(
                    element_id="t1",
                    doc_id="doc-1",
                    page=1,
                    element_type=ElementType.TABLE,
                    text="",
                    metadata={"headers": ("District", "2024 Consumption"), "rows": (("Bakirkoy", "100"),)},
                )
            ],
            metadata={"parser": "pymupdf-layout"},
        )

        frame = parsed_tables_to_dataframe(document)

        self.assertEqual(list(frame.columns), ["District", "2024 Consumption"])
        self.assertEqual(frame.iloc[0]["District"], "Bakirkoy")


if __name__ == "__main__":
    unittest.main()
