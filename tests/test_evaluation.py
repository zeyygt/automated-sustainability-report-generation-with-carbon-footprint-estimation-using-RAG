import unittest

from rag_retrieval.evaluation import (
    RetrievalExample,
    evaluate_rankings,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class EvaluationTests(unittest.TestCase):
    def test_metrics(self):
        ranked = ["c3", "c2", "c1"]
        relevant = {"c1", "c2"}

        self.assertEqual(recall_at_k(ranked, relevant, 2), 0.5)
        self.assertEqual(precision_at_k(ranked, relevant, 2), 0.5)
        self.assertEqual(reciprocal_rank(ranked, relevant), 0.5)

    def test_evaluate_rankings(self):
        examples = [RetrievalExample("q1", frozenset({"c1"}))]
        metrics = evaluate_rankings({"q1": ["c2", "c1"]}, examples, k_values=(1, 2))

        self.assertEqual(metrics["recall@1"], 0.0)
        self.assertEqual(metrics["recall@2"], 1.0)
        self.assertEqual(metrics["mrr"], 0.5)


if __name__ == "__main__":
    unittest.main()

