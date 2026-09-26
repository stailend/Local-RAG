import tempfile
import unittest
from pathlib import Path

from local_rag.documents import chunks, discover, extract


class DocumentsTest(unittest.TestCase):
    def test_chunks_overlap_and_reconstruct(self):
        text = " ".join(f"word-{number}" for number in range(100))
        result = chunks(text, size=120, overlap=20)
        self.assertGreater(len(result), 1)
        self.assertTrue(all(len(chunk) <= 120 for chunk in result))

    def test_discovers_and_extracts_supported_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "note.md").write_text("private knowledge", encoding="utf-8")
            (root / "ignored.bin").write_bytes(b"nope")
            files = discover(root)
            self.assertEqual(files, [root / "note.md"])
            self.assertEqual(extract(files[0])[0].text, "private knowledge")

    def test_rejects_invalid_chunk_settings(self):
        with self.assertRaises(ValueError):
            chunks("text", size=10, overlap=10)


if __name__ == "__main__":
    unittest.main()

