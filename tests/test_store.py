import tempfile
import unittest
from pathlib import Path

from qdrant_client import QdrantClient

from local_rag.store import Store


class FakeProvider:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [float("cats" in text.lower()), float("dogs" in text.lower()), 0.1]
            for text in texts
        ]


class FailingProvider:
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("model unavailable")


class StoreTest(unittest.TestCase):
    def test_ingest_and_search(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "animals.md").write_text(
                "Cats are quiet pets. Dogs enjoy long walks.", encoding="utf-8"
            )
            store = Store("", "test", QdrantClient(":memory:"))
            result = store.ingest(root, FakeProvider(), chunk_size=24, overlap=4)

            self.assertEqual(result.files, 1)
            self.assertGreaterEqual(result.chunks, 2)
            matches = store.search([1.0, 0.0, 0.1], limit=1)
            self.assertIn("Cats", matches[0]["text"])

            (root / "animals.md").write_text("Dogs enjoy long walks.", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                store.ingest(root, FailingProvider(), chunk_size=24, overlap=4)
            self.assertIn("Cats", store.search([1.0, 0.0, 0.1], limit=1)[0]["text"])

            store.ingest(root, FakeProvider(), chunk_size=24, overlap=4)
            self.assertNotIn(
                "Cats",
                " ".join(item["text"] for item in store.search([1.0, 0.0, 0.1], 5)),
            )


if __name__ == "__main__":
    unittest.main()
