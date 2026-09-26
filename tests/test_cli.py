import unittest

from local_rag.cli import _merge_matches


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


if __name__ == "__main__":
    unittest.main()
