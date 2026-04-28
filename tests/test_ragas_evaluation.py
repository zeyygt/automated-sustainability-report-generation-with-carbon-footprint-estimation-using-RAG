import importlib.util
import unittest
from pathlib import Path

from rag_retrieval import RetrievalSession
from rag_retrieval.ragas_evaluation import (
    DEFAULT_REFERENCE_SPECS,
    resolve_reference_chunks,
    run_ragas_retrieval_evaluation,
)


RAGAS_AVAILABLE = importlib.util.find_spec("ragas") is not None and importlib.util.find_spec("datasets") is not None


class RagasEvaluationTests(unittest.TestCase):
    def test_reference_chunk_resolution(self):
        session = RetrievalSession()
        session.build_index([Path("test1.pdf"), Path("rawtext.pdf")])

        resolved = resolve_reference_chunks(session, DEFAULT_REFERENCE_SPECS)

        self.assertEqual(set(resolved), {"test1_intro", "test1_table", "rawtext_notes"})

    @unittest.skipUnless(RAGAS_AVAILABLE, "ragas optional dependency not installed")
    def test_ragas_retrieval_evaluation_smoke(self):
        report = run_ragas_retrieval_evaluation([Path("test1.pdf"), Path("rawtext.pdf")], top_k=3)

        self.assertIn("ragas_metrics", report)
        self.assertIn("classical_metrics", report)
        self.assertTrue(report["per_query"])
        self.assertIn("id_based_context_recall", report["ragas_metrics"])


if __name__ == "__main__":
    unittest.main()
