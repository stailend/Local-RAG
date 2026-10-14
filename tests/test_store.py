import tempfile
import unittest
from pathlib import Path

from qdrant_client import QdrantClient

from local_rag.store import Store
from local_rag.library import cluster_sources, clusters, summarize_files


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

    def summarize(self, text: str) -> str:
        return text[:150]


class FailingProvider:
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("model unavailable")


class StoreTest(unittest.TestCase):
    def test_file_summaries_groups_and_sync_lifecycle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = root / "docs"
            docs.mkdir()
            cats = docs / "cats.md"
            kittens = docs / "kittens.md"
            dogs = docs / "dogs.md"
            cats.write_text("Cats are quiet pets.", encoding="utf-8")
            kittens.write_text("Cats like warm places.", encoding="utf-8")
            dogs.write_text("Dogs enjoy walks.", encoding="utf-8")
            provider = FakeProvider()
            store = Store("", "library", QdrantClient(path=str(root / "qdrant")), FakeSparse())
            store.sync(docs, provider, 1000, 100)
            self.assertEqual(summarize_files(store, provider), {"created": 3, "unchanged": 0})
            self.assertEqual(summarize_files(store, provider), {"created": 0, "unchanged": 3})
            store.ingest(cats, provider, 1000, 100)
            self.assertEqual(len(store.file_summaries()), 3)
            groups = clusters(store)
            self.assertEqual(sorted(len(group["files"]) for group in groups), [1, 2])
            cat_group = next(group for group in groups if len(group["files"]) == 2)
            sources = cluster_sources(store, cat_group["id"])
            self.assertEqual(set(sources), {str(cats.resolve()), str(kittens.resolve())})
            matches = store.search("cats", [1, 0, .1], 5, sources=sources)
            self.assertEqual({item["source"] for item in matches}, set(sources))

            cats.write_text("Dogs and walks.", encoding="utf-8")
            kittens.unlink()
            store.sync(docs, provider, 1000, 100)
            self.assertEqual([row["source"] for row in store.file_summaries()], [str(dogs.resolve())])
            self.assertEqual(summarize_files(store, provider), {"created": 1, "unchanged": 1})
            self.assertEqual(len(store.file_summaries()), 2)
            store.delete()
            self.assertEqual(store.file_summaries(), [])

    def test_filters_dense_hybrid_and_confidence_by_file_and_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = root / "docs"
            nested = docs / "nested"
            nested.mkdir(parents=True)
            cats = docs / "cats.md"
            dogs = nested / "dogs.md"
            cats.write_text("Cats sleep in warm places.", encoding="utf-8")
            dogs.write_text("Dogs enjoy long walks.", encoding="utf-8")
            store = Store("", "filters", QdrantClient(path=str(root / "qdrant")), FakeSparse())
            store.ingest(docs, FakeProvider(), chunk_size=1000, overlap=100)

            self.assertEqual(store.sources(), sorted([str(cats.resolve()), str(dogs.resolve())]))
            for mode in ("dense", "hybrid"):
                matches = store.search("cats", [1.0, 0.0, 0.1], 5, mode=mode, source=str(dogs))
                self.assertEqual([item["source"] for item in matches], [str(dogs.resolve())])
                matches = store.search("cats", [1.0, 0.0, 0.1], 5, mode=mode, folder=str(docs))
                self.assertEqual({item["source"] for item in matches}, {str(cats.resolve()), str(dogs.resolve())})
            self.assertEqual(store.top_dense_score([1.0, 0.0, 0.1], source=str(root / "missing.md")), 0.0)

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
            progress = []
            result = store.ingest(
                knowledge, FakeProvider(), chunk_size=24, overlap=4,
                progress=lambda stage, done, total: progress.append((stage, done, total)),
            )

            self.assertEqual(result.files, 1)
            self.assertGreaterEqual(result.chunks, 2)
            self.assertEqual(progress[-1], ("indexing", result.chunks, result.chunks))
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

            changes = []
            initial = store.sync(
                knowledge, FakeProvider(), 1000, 100,
                on_change=lambda source, before, after: changes.append((source, before, after)),
            )
            self.assertEqual(len(changes), 2)
            self.assertTrue(all(before is None and after is not None for _, before, after in changes))
            unchanged = store.sync(knowledge, FakeProvider(), 1000, 100)
            first.write_text("Cats sleep most of the day.", encoding="utf-8")
            removed.unlink()
            (knowledge / "added.md").write_text("Birds can fly.", encoding="utf-8")
            changes.clear()
            changed = store.sync(
                knowledge, FakeProvider(), 1000, 100,
                on_change=lambda source, before, after: changes.append((source, before, after)),
            )

            self.assertEqual((initial.added, initial.updated, initial.deleted), (2, 0, 0))
            self.assertEqual(unchanged.unchanged, 2)
            self.assertEqual(
                (changed.added, changed.updated, changed.deleted, changed.unchanged),
                (1, 1, 1, 0),
            )
            self.assertEqual(store.status()[0], 2)
            self.assertEqual(len(changes), 3)
            updated = next(change for change in changes if change[0] == str(first.resolve()))
            self.assertIn("Cats are quiet", updated[1]["chunks"][0]["text"])
            self.assertIn("Cats sleep", updated[2]["chunks"][0]["text"])
            removed = next(change for change in changes if change[0] == str(removed.resolve()))
            self.assertIsNone(removed[2])


if __name__ == "__main__":
    unittest.main()
