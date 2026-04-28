import json
import tempfile
import unittest
from pathlib import Path

from rag_retrieval.ragas_generation_evaluation import (
    render_generation_answers_markdown,
    write_generation_answer_artifacts,
)


class RagasGenerationEvaluationTests(unittest.TestCase):
    def test_generation_answer_artifacts_are_written(self):
        output = {
            "generation_model": "gpt-4o-mini",
            "metrics": {
                "faithfulness": 0.75,
                "answer_relevancy": 0.98,
                "response_groundedness": 1.0,
            },
            "per_query": [
                {
                    "user_input": "What was Bakirkoy natural gas consumption in 2024?",
                    "response": "Bakirkoy's natural gas consumption in 2024 was 1,343,469.02 m³.",
                    "retrieved_contexts": [
                        "Bakirkoy 2024 natural gas consumption: 1,343,469.02 m³",
                    ],
                    "faithfulness": 1.0,
                    "answer_relevancy": 0.99,
                    "response_groundedness": 1.0,
                }
            ],
        }

        markdown = render_generation_answers_markdown(output)
        self.assertIn("Generated Answers Evaluation", markdown)
        self.assertIn("Bakirkoy's natural gas consumption", markdown)

        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = write_generation_answer_artifacts(output, Path(tmp_dir))

            json_path = Path(paths["json"])
            markdown_path = Path(paths["markdown"])
            self.assertTrue(json_path.exists())
            self.assertTrue(markdown_path.exists())

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload[0]["query"], output["per_query"][0]["user_input"])


if __name__ == "__main__":
    unittest.main()
