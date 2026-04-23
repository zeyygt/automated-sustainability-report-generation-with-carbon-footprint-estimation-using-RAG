import unittest
from types import SimpleNamespace

from rag_retrieval.models import Query
from rag_retrieval.pipeline import analyze_data_engines, extract_district, handle_query


def make_query(
    text="",
    terms=(),
    intents=(),
    source_hints=(),
):
    return Query(
        raw_text=text,
        normalized_text=text,
        expanded_text=text,
        terms=terms,
        expanded_terms=terms,
        phrases=(),
        numbers=(),
        years=(),
        scope_terms=(),
        intents=intents,
        source_hints=source_hints,
    )


class FakeQueryProcessor:
    def __init__(self, query):
        self.query = query

    def process(self, query_text):
        return self.query


class FakeEngine:
    def __init__(self, total=42.0, empty=False):
        self.total = total
        self.empty = empty
        self.calls = []

    def analyze_district(self, district):
        self.calls.append(district)
        if self.empty:
            return {}
        return {"district": district, "total": self.total}


class FakeSession:
    def __init__(self, query, engines=None, documents=None):
        self.retrieval = SimpleNamespace(query_processor=FakeQueryProcessor(query))
        self.data_engines = engines or {}
        self.documents = documents or {}
        self.search_calls = []

    def search(self, query_text):
        self.search_calls.append(query_text)
        return ["hit"]


class PipelineTests(unittest.TestCase):
    def test_extract_district_matches_known_terms(self):
        query = make_query(terms=("monthly", "kadikoy", "consumption"))

        self.assertEqual(extract_district(query), "Kadikoy")

    def test_handle_query_excel_route_uses_data_engine(self):
        query = make_query(terms=("adalar",), intents=("numeric",))
        engine = FakeEngine()
        session = FakeSession(
            query,
            {"doc-1": engine},
            {"doc-1": SimpleNamespace(filename="metrics.xlsx", metadata={"parser": "spreadsheet"})},
        )

        result = handle_query("ADALAR 2023 consumption", session)

        expected_structured = [
            {
                "doc_id": "doc-1",
                "filename": "metrics.xlsx",
                "source_type": "spreadsheet",
                "parser": "spreadsheet",
                "data": {"district": "Adalar", "total": 42.0},
            }
        ]
        self.assertEqual(result["query"], "ADALAR 2023 consumption")
        self.assertEqual(result["route"], "excel")
        self.assertEqual(result["retrieval_context"], [])
        self.assertEqual(result["structured_results"], expected_structured)
        self.assertEqual(result["sources"][0]["filename"], "metrics.xlsx")
        self.assertEqual(result["sources"][0]["usage"], ["structured"])
        self.assertEqual(result["type"], "excel")
        self.assertEqual(result["data"], expected_structured)
        self.assertEqual(engine.calls, ["Adalar"])
        self.assertEqual(session.search_calls, [])

    def test_handle_query_excel_route_without_engine_returns_empty_data(self):
        query = make_query(terms=("adalar",), intents=("numeric",))
        session = FakeSession(query)

        result = handle_query("ADALAR 2023 consumption", session)

        self.assertEqual(result["route"], "excel")
        self.assertEqual(result["structured_results"], [])
        self.assertEqual(result["warnings"], ["no_structured_results"])

    def test_handle_query_pdf_route_uses_retrieval(self):
        query = make_query(terms=("report",), source_hints=("pdf",))
        session = FakeSession(query, {"doc-1": FakeEngine()})

        result = handle_query("show report context", session)

        self.assertEqual(result["route"], "pdf")
        self.assertEqual(result["retrieval_context"], [{"rank": 1, "text": "hit"}])
        self.assertEqual(result["context"], [{"rank": 1, "text": "hit"}])
        self.assertEqual(result["structured_results"], [])
        self.assertEqual(session.search_calls, ["show report context"])

    def test_handle_query_hybrid_route_uses_both_paths(self):
        query = make_query(terms=("besiktas",))
        engine = FakeEngine()
        session = FakeSession(
            query,
            {"doc-1": engine},
            {"doc-1": SimpleNamespace(filename="report.pdf", metadata={"parser": "pymupdf-layout"})},
        )

        result = handle_query("Besiktas sustainability context", session)

        expected_structured = [
            {
                "doc_id": "doc-1",
                "filename": "report.pdf",
                "source_type": "pdf",
                "parser": "pymupdf-layout",
                "data": {"district": "Besiktas", "total": 42.0},
            }
        ]
        self.assertEqual(result["route"], "hybrid")
        self.assertEqual(result["retrieval_context"], [{"rank": 1, "text": "hit"}])
        self.assertEqual(result["structured_results"], expected_structured)
        self.assertEqual(result["sources"][0]["filename"], "report.pdf")
        self.assertEqual(result["sources"][0]["usage"], ["structured"])
        self.assertEqual(session.search_calls, ["Besiktas sustainability context"])
        self.assertEqual(engine.calls, ["Besiktas"])

    def test_analyze_data_engines_returns_source_based_results_for_all_engines(self):
        pdf_engine = FakeEngine(total=10.0)
        spreadsheet_engine = FakeEngine(total=20.0)
        empty_engine = FakeEngine(empty=True)
        session = SimpleNamespace(
            data_engines={"pdf-doc": pdf_engine, "sheet-doc": spreadsheet_engine, "empty-doc": empty_engine},
            documents={
                "pdf-doc": SimpleNamespace(filename="report.pdf", metadata={"parser": "pymupdf-layout"}),
                "sheet-doc": SimpleNamespace(filename="metrics.xlsx", metadata={"parser": "spreadsheet"}),
                "empty-doc": SimpleNamespace(filename="empty.pdf", metadata={"parser": "pymupdf-layout"}),
            },
        )

        results = analyze_data_engines(session, "Kadikoy")

        self.assertEqual(
            results,
            [
                {
                    "doc_id": "pdf-doc",
                    "filename": "report.pdf",
                    "source_type": "pdf",
                    "parser": "pymupdf-layout",
                    "data": {"district": "Kadikoy", "total": 10.0},
                },
                {
                    "doc_id": "sheet-doc",
                    "filename": "metrics.xlsx",
                    "source_type": "spreadsheet",
                    "parser": "spreadsheet",
                    "data": {"district": "Kadikoy", "total": 20.0},
                },
            ],
        )
        self.assertEqual(pdf_engine.calls, ["Kadikoy"])
        self.assertEqual(spreadsheet_engine.calls, ["Kadikoy"])
        self.assertEqual(empty_engine.calls, ["Kadikoy"])

    def test_analyze_data_engines_respects_source_hints(self):
        pdf_engine = FakeEngine(total=10.0)
        spreadsheet_engine = FakeEngine(total=20.0)
        session = SimpleNamespace(
            data_engines={"pdf-doc": pdf_engine, "sheet-doc": spreadsheet_engine},
            documents={
                "pdf-doc": SimpleNamespace(filename="report.pdf", metadata={"parser": "pymupdf-layout"}),
                "sheet-doc": SimpleNamespace(filename="metrics.xlsx", metadata={"parser": "spreadsheet"}),
            },
        )

        results = analyze_data_engines(session, "Kadikoy", source_hints=("spreadsheet",))

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["doc_id"], "sheet-doc")
        self.assertEqual(pdf_engine.calls, [])
        self.assertEqual(spreadsheet_engine.calls, ["Kadikoy"])


if __name__ == "__main__":
    unittest.main()
