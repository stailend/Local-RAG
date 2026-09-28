import tempfile
import unittest
from pathlib import Path

from qdrant_client import QdrantClient

from local_rag.store import Store


class SparseVector:
    def __init__(self, indices, values):
        self.indices = __import__("numpy").array(indices)
        self.values = __import__("numpy").array(values)


class FakeSparse:
    def embed(self, texts):
        return [SparseVector([1 if "cats" in text.lower() else 2], [1.0]) for text in texts]

    def query_embed(self, text):
        return iter([SparseVector([1 if "cats" in text.lower() else 2], [1.0])])


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
            knowledge = root / "knowledge"
            knowledge.mkdir()
            (knowledge / "animals.md").write_text(
                "Cats are quiet pets. Dogs enjoy long walks.", encoding="utf-8"
            )
            store = Store(
                "",
                "test",
                QdrantClient(path=str(root / "qdrant")),
                FakeSparse(),
            )
            result = store.ingest(
                knowledge, FakeProvider(), chunk_size=24, overlap=4
            )

            self.assertEqual(result.files, 1)
            self.assertGreaterEqual(result.chunks, 2)
            matches = store.search("cats", [1.0, 0.0, 0.1], limit=1)
            self.assertIn("Cats", matches[0]["text"])
            explained = store.search(
                "cats", [1.0, 0.0, 0.1], limit=2, explain=True
            )
            self.assertEqual(explained[0]["_dense_rank"], 1)
            self.assertEqual(explained[0]["_sparse_rank"], 1)

            (knowledge / "animals.md").write_text(
                "Dogs enjoy long walks.", encoding="utf-8"
            )
            with self.assertRaises(RuntimeError):
                store.ingest(knowledge, FailingProvider(), chunk_size=24, overlap=4)
            self.assertIn(
                "Cats", store.search("cats", [1.0, 0.0, 0.1], limit=1)[0]["text"]
            )

            store.ingest(knowledge, FakeProvider(), chunk_size=24, overlap=4)
            self.assertNotIn(
                "Cats",
                " ".join(
                    item["text"]
                    for item in store.search("cats", [1.0, 0.0, 0.1], 5)
                ),
            )

    def test_sync_adds_updates_skips_and_deletes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            knowledge = root / "knowledge"
            knowledge.mkdir()
            first = knowledge / "first.md"
            removed = knowledge / "removed.md"
            first.write_text("Cats are quiet pets.", encoding="utf-8")
            removed.write_text("Dogs enjoy walks.", encoding="utf-8")
            store = Store(
                "",
                "sync",
                QdrantClient(path=str(root / "qdrant")),
                FakeSparse(),
            )

            initial = store.sync(knowledge, FakeProvider(), 1000, 100)
            unchanged = store.sync(knowledge, FakeProvider(), 1000, 100)
            first.write_text("Cats sleep most of the day.", encoding="utf-8")
            removed.unlink()
            (knowledge / "added.md").write_text("Birds can fly.", encoding="utf-8")
            changed = store.sync(knowledge, FakeProvider(), 1000, 100)

            self.assertEqual((initial.added, initial.updated, initial.deleted), (2, 0, 0))
            self.assertEqual(unchanged.unchanged, 2)
            self.assertEqual(
                (changed.added, changed.updated, changed.deleted, changed.unchanged),
                (1, 1, 1, 0),
            )
            self.assertEqual(store.status()[0], 2)


if __name__ == "__main__":
    unittest.main()
