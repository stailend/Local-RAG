import unittest

from local_rag.library import group_files, summarize_files


class FakeStore:
    def __init__(self):
        self.saved = None

    def source_chunks(self, source=None):
        document = {"source": "/docs/long.md", "source_id": "id", "source_version": "v1",
                    "chunks": [{"locator": f"page {i}", "text": "Cats sleep. " * 100}
                               for i in range(10)]}
        return {document["source"]: document}

    def file_summaries(self):
        return []

    def save_file_summary(self, document, summary, vector):
        self.saved = (document, summary, vector)


class FakeProvider:
    def __init__(self):
        self.calls = []

    def summarize(self, text):
        self.calls.append(text)
        return "Cats sleep."

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class LibraryTest(unittest.TestCase):
    def test_long_summary_is_staged_and_embedded(self):
        store, provider = FakeStore(), FakeProvider()
        self.assertEqual(summarize_files(store, provider)["created"], 1)
        self.assertGreater(len(provider.calls), 2)
        self.assertLess(max(map(len, provider.calls)), 6000)
        self.assertEqual(store.saved[1:], ("Cats sleep.", [1.0, 0.0]))

    def test_grouping_rejects_invalid_threshold(self):
        with self.assertRaises(ValueError):
            group_files([], 1.1)


if __name__ == "__main__":
    unittest.main()
