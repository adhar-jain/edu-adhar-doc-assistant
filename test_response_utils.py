import unittest
from types import SimpleNamespace

from agents.rag_agent import reason, rewrite_query
from llm.response_utils import extract_text_content


class ExtractTextContentTests(unittest.TestCase):
    def test_returns_plain_text_unchanged(self):
        self.assertEqual(extract_text_content("hello"), "hello")

    def test_joins_gemini_text_blocks_and_ignores_metadata(self):
        content = [
            {"type": "text", "text": "first "},
            {"type": "text", "text": "second", "extras": {"signature": "ignored"}},
        ]

        self.assertEqual(extract_text_content(content), "first second")

    def test_raises_for_unsupported_content(self):
        with self.assertRaises(TypeError):
            extract_text_content(None)

    def test_rewrite_query_returns_text_for_structured_model_content(self):
        llm = SimpleNamespace(
            invoke=lambda _: SimpleNamespace(
                content=[{"type": "text", "text": "rewritten query"}]
            )
        )

        result = rewrite_query({"question": "original question"}, llm)

        self.assertEqual(result["rewritten_question"], "rewritten query")

    def test_reason_returns_text_for_structured_model_content(self):
        llm = SimpleNamespace(
            invoke=lambda _: SimpleNamespace(
                content=[{"type": "text", "text": "grounded answer"}]
            )
        )

        result = reason(
            {"question": "question", "context": "document context"},
            llm,
        )

        self.assertEqual(result["answer"], "grounded answer")


if __name__ == "__main__":
    unittest.main()
