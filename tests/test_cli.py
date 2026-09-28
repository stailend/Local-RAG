import unittest

from local_rag.cli import _merge_matches, _rerank


class FakeRanker:
    def rerank(self, request):
        return [
            {"meta": request.passages[1]["meta"], "score": 0.9},
            {"meta": request.passages[0]["meta"], "score": 0.1},
        ]


class MergeMatchesTest(unittest.TestCase):
    def test_merges_overlapping_adjacent_chunks_in_document_order(self):
        matches = [
            {"source": "cv.pdf", "locator": "page 1", "chunk": 1, "text": "uptime 99.7%. Next job"},
            {"source": "cv.pdf", "locator": "page 1", "chunk": 0, "text": "RentenNavi uptime 99.7%."},
            {"source": "other.pdf", "locator": "page 1", "chunk": 0, "text": "Other"},
        ]

        result = _merge_matches(matches)

        self.assertEqual(result[0]["text"], "RentenNavi uptime 99.7%. Next job")
        self.assertEqual(result[1]["text"], "Other")

    def test_reranker_reorders_candidates(self):
        matches = [{"text": "wrong"}, {"text": "right"}]

        result = _rerank("question", matches, 1, "unused", FakeRanker())

        self.assertEqual(result[0]["text"], "right")
        self.assertEqual(result[0]["_rerank_score"], 0.9)


if __name__ == "__main__":
    unittest.main()
