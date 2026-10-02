import unittest
from types import SimpleNamespace

from local_rag.api import RagService


class FakeProvider:
    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def answer(self, question, context):
        return f"answer: {question}"


class FakeStore:
    def status(self):
        return 2, "green"

    def search(self, question, vector, limit):
        return [{
            "source": "notes.md", "locator": "page 1", "chunk": 0,
            "text": "The answer is here.", "_fused_score": 0.9,
        }][:limit]


class ApiTest(unittest.TestCase):
    def setUp(self):
        settings = SimpleNamespace(collection="test", provider="test")
        self.service = RagService(settings, FakeStore(), FakeProvider())

    def test_status_and_ask(self):
        self.assertEqual(self.service.status()["points"], 2)
        result = self.service.ask("What?")
        self.assertEqual(result["answer"], "answer: What?")
        self.assertEqual(result["sources"][0]["source"], "notes.md")

    def test_rejects_empty_question(self):
        with self.assertRaises(ValueError):
            self.service.search(" ")


if __name__ == "__main__":
    unittest.main()
