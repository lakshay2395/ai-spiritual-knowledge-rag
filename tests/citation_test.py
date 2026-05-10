import os
import sys
import unittest

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.orchestrator import RAGOrchestrator, Source


class TestCitationEngine(unittest.TestCase):
    def setUp(self):
        # Mock GEMINI_API_KEY to avoid validation errors
        os.environ["GEMINI_API_KEY"] = "mock-key"
        # We don't need a real LLM for unit testing these methods
        self.orchestrator = RAGOrchestrator(model_name="mock-model")
        self.mock_sources = [
            Source(
                text="Text 1",
                citation="John 3:16",
                metadata={"book": "John", "chapter": 3, "verse": 16},
            ),
            Source(
                text="Text 2",
                citation="Matthew 5:1",
                metadata={"book": "Matthew", "chapter": 5, "verse": 1},
            ),
            Source(
                text="Text 3",
                citation="Bhagavad Gita 2.13",
                metadata={"chapter": 2, "verse": 13},
            ),
        ]

    def test_validate_citations_success(self):
        answer = "As seen in [John 3:16], God loved the world. Also [Bhagavad Gita 2.13] mentions the soul."
        valid = self.orchestrator._validate_citations(answer, self.mock_sources)
        self.assertIn("John 3:16", valid)
        self.assertIn("Bhagavad Gita 2.13", valid)
        self.assertEqual(len(valid), 2)

    def test_validate_citations_hallucination(self):
        answer = "The text says something [Luke 1:1] which is not in context."
        valid = self.orchestrator._validate_citations(answer, self.mock_sources)
        self.assertEqual(len(valid), 0)

    def test_format_response(self):
        response_data = {
            "answer": "Test answer [John 3:16]",
            "sources": [
                {"text": "For God so loved...", "citation": "John 3:16", "metadata": {}}
            ],
        }
        formatted = self.orchestrator.format_response(response_data)
        self.assertIn("ANSWER:", formatted)
        self.assertIn("Test answer [John 3:16]", formatted)
        self.assertIn("SOURCES:", formatted)
        self.assertIn("[1] John 3:16", formatted)


if __name__ == "__main__":
    unittest.main()
