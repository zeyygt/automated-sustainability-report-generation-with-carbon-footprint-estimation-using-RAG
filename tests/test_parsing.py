import tempfile
import unittest
from pathlib import Path

from rag_retrieval.models import DocumentInput, ElementType
from rag_retrieval.parsing import FallbackTextParser, _looks_like_heading


class ParsingTests(unittest.TestCase):
    def test_markdown_heading_keeps_following_paragraph(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.md"
            path.write_text("# GHG Emissions\nScope 1 emissions decreased in 2024.\n", encoding="utf-8")
            document = DocumentInput("doc-1", path, path.name, "text/markdown")

            parsed = FallbackTextParser().parse(document)

        self.assertEqual([element.element_type for element in parsed.elements], [ElementType.HEADING, ElementType.PARAGRAPH])
        self.assertEqual(parsed.elements[1].section_path, ("GHG Emissions",))
        self.assertIn("Scope 1 emissions", parsed.elements[1].text)

    def test_table_intro_sentence_is_not_heading(self):
        text = "The following table presents the annual natural gas consumption figures for 15 districts:"

        self.assertFalse(_looks_like_heading(text, font_size=14.0, body_size=10.0, boldish=True))


if __name__ == "__main__":
    unittest.main()
