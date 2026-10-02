from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .cli import _merge_matches, _rerank
from .config import Settings
from .providers import provider_from_settings
from .store import Store


class RagService:
    def __init__(self, settings: Settings, store: Store | None = None, provider=None):
        self.settings = settings
        self.store = store or Store(settings.qdrant_url, settings.collection)
        self.provider = provider or provider_from_settings(settings)

    def status(self) -> dict[str, Any]:
        points, state = self.store.status()
        return {
            "collection": self.settings.collection,
            "points": points,
            "status": state,
            "provider": self.settings.provider,
        }

    def search(
        self,
        question: str,
        top_k: int = 5,
        candidates: int = 20,
        rerank: bool = True,
        rerank_model: str = "ms-marco-TinyBERT-L-2-v2",
    ) -> list[dict]:
        question = question.strip()
        if not question:
            raise ValueError("question is required")
        if top_k < 1 or candidates < top_k:
            raise ValueError("candidates must be greater than or equal to top_k >= 1")
        vector = self.provider.embed([question])[0]
        matches = self.store.search(question, vector, candidates)
        if rerank:
            matches = _rerank(question, matches, top_k, rerank_model)
        else:
            matches = matches[:top_k]
        return [_public_match(match) for match in matches]

    def ask(self, question: str, **options) -> dict[str, Any]:
        matches = _merge_matches(self.search(question, **options))
        if not matches:
            return {"answer": "No indexed context found.", "sources": []}
        context = "\n\n".join(
            f"[{number}] {item['source']} ({item['locator']})\n{item['text']}"
            for number, item in enumerate(matches, 1)
        )
        return {
            "answer": self.provider.answer(question, context),
            "sources": [
                {key: item[key] for key in ("source", "locator", "chunk")}
                for item in matches
            ],
        }


def _public_match(match: dict) -> dict:
    return {
        key: match.get(key)
        for key in ("source", "locator", "chunk", "text", "_fused_score", "_rerank_score")
        if match.get(key) is not None
    }


class _Handler(BaseHTTPRequestHandler):
    service: RagService
    web_root = Path(__file__).with_name("web")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._file("index.html", "text/html; charset=utf-8")
        elif path == "/api/status":
            self._json(self.service.status())
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in {"/api/search", "/api/ask"}:
            self._json({"error": "not found"}, 404)
            return
        try:
            payload = self._body()
            options = {
                "top_k": int(payload.get("top_k", 5)),
                "candidates": int(payload.get("candidates", 20)),
                "rerank": bool(payload.get("rerank", True)),
                "rerank_model": str(payload.get("rerank_model", "ms-marco-TinyBERT-L-2-v2")),
            }
            if path == "/api/search":
                result = {"matches": self.service.search(str(payload.get("question", "")), **options)}
            else:
                result = self.service.ask(str(payload.get("question", "")), **options)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, 400)
        except Exception as exc:  # keep provider/database details out of tracebacks
            self._json({"error": str(exc)}, 502)
        else:
            self._json(result)

    def log_message(self, *_args) -> None:
        return

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length > 1_000_000:
            raise ValueError("request body is too large")
        value = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def _file(self, name: str, content_type: str) -> None:
        try:
            data = (self.web_root / name).read_bytes()
        except OSError:
            self._json({"error": "web asset not found"}, 500)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, value: dict, status: int = 200) -> None:
        data = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def serve(settings: Settings, host: str = "127.0.0.1", port: int = 8000) -> None:
    service = RagService(settings)
    handler = type("RagHandler", (_Handler,), {"service": service})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Local-RAG UI: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
