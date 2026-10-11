import unittest
import tempfile
import time
import json
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

from local_rag.api import RagService, _Handler


class FakeProvider:
    calls = 0
    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def answer(self, question, context):
        self.calls += 1
        return f"answer: {question}"


class FakeStore:
    score = 0.9

    def status(self):
        return 2, "green"

    def search(self, question, vector, limit, source=None, folder=None):
        return [{
            "source": "notes.md", "locator": "page 1", "chunk": 0,
            "text": "The answer is here.", "_fused_score": 0.9,
        }][:limit]

    def top_dense_score(self, vector, source=None, folder=None):
        return self.score

    def sources(self):
        return ["/docs/notes.md"]

    def ingest(self, path, provider, chunk_size, overlap, progress=None):
        progress("embedding", 1, 2)
        progress("indexing", 2, 2)
        return SimpleNamespace(chunks=2, skipped=0)


class ApiTest(unittest.TestCase):
    def setUp(self):
        settings = SimpleNamespace(collection="test", provider="test")
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.service = RagService(
            settings, FakeStore(), FakeProvider(), Path(self.directory.name)
        )
        self.addCleanup(self.service._executor.shutdown, wait=True)

    def test_status_and_ask(self):
        self.assertEqual(self.service.status()["points"], 2)
        self.assertEqual(self.service.sources()["folders"][0], "/docs")
        result = self.service.ask("What?")
        self.assertEqual(result["answer"], "answer: What?")
        self.assertEqual(result["sources"][0]["source"], "notes.md")

    def test_rejects_empty_question(self):
        with self.assertRaises(ValueError):
            self.service.search(" ")

    def test_low_score_does_not_call_llm(self):
        self.service.store.score = 0.1
        result = self.service.ask("Unrelated?", min_score=0.35)
        self.assertEqual(result["sources"], [])
        self.assertEqual(self.service.provider.calls, 0)

    def test_upload_indexes_and_reports_progress(self):
        queued = self.service.upload("notes.md", b"The answer is here")
        for _ in range(100):
            job = self.service.job(queued["id"])
            if job["stage"] in {"done", "error"}:
                break
            time.sleep(0.01)
        self.assertEqual(job["stage"], "done")
        self.assertEqual((job["done"], job["total"]), (2, 2))
        self.assertEqual(len(list(Path(self.directory.name).glob("*.md"))), 1)

    def test_upload_rejects_unsafe_name(self):
        with self.assertRaises(ValueError):
            self.service.upload("../notes.md", b"text")

    def test_http_upload_and_job_status(self):
        handler = type("TestHandler", (_Handler,), {"service": self.service})
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        connection = HTTPConnection("127.0.0.1", server.server_port)
        self.addCleanup(connection.close)
        connection.request(
            "POST", "/api/uploads?filename=notes.md", b"The answer is here",
            {"Content-Type": "application/octet-stream"},
        )
        response = connection.getresponse()
        self.assertEqual(response.status, 202)
        job_id = json.loads(response.read())["id"]
        for _ in range(100):
            connection.request("GET", f"/api/jobs/{job_id}")
            job = json.loads(connection.getresponse().read())
            if job["stage"] in {"done", "error"}:
                break
            time.sleep(0.01)
        self.assertEqual(job["stage"], "done")


if __name__ == "__main__":
    unittest.main()
