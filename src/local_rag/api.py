from __future__ import annotations

import json
import hashlib
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .cli import _merge_matches, _rerank
from .config import Settings
from .documents import SUPPORTED_EXTENSIONS
from .library import cluster_sources, clusters, summarize_files
from .providers import provider_from_settings
from .store import Store

MAX_UPLOAD = 25 * 1024 * 1024
DEFAULT_MIN_SCORE = 0.50


class RagService:
    def __init__(self, settings: Settings, store: Store | None = None, provider=None,
                 upload_root: Path | None = None):
        self.settings = settings
        self.store = store or Store(settings.qdrant_url, settings.collection)
        self.provider = provider or provider_from_settings(settings)
        self.upload_root = upload_root or Path(".localrag/uploads") / hashlib.sha256(
            settings.collection.encode()
        ).hexdigest()[:16]
        self.jobs: dict[str, dict] = {}
        self._jobs_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)

    def status(self) -> dict[str, Any]:
        points, state = self.store.status()
        return {
            "collection": self.settings.collection,
            "points": points,
            "status": state,
            "provider": self.settings.provider,
        }

    def sources(self) -> dict[str, list[str]]:
        sources = self.store.sources()
        return {
            "sources": sources,
            "folders": sorted({str(parent) for source in sources
                               for parent in Path(source).parents if parent != parent.parent}),
        }

    def clusters(self) -> dict[str, list[dict]]:
        return {"clusters": clusters(self.store, self.provider)}

    def source_preview(self, source: str, locator: str, chunk: int,
                       last_chunk: int | None = None) -> dict:
        if not source or not locator or chunk < 0 or (last_chunk is not None and last_chunk < chunk):
            raise ValueError("invalid source location")
        document = self.store.source_chunks(source).get(source)
        if not document:
            raise FileNotFoundError("indexed source not found")
        chunks = [
            {"chunk": item["chunk"], "text": item["text"],
             "selected": chunk <= item["chunk"] <= (last_chunk if last_chunk is not None else chunk)}
            for item in document["chunks"] if item["locator"] == locator
        ]
        if not any(item["selected"] for item in chunks):
            raise FileNotFoundError("indexed passage not found")
        return {"source": source, "locator": locator, "chunks": chunks}

    def search(
        self,
        question: str,
        top_k: int = 5,
        candidates: int = 20,
        rerank: bool = True,
        rerank_model: str = "ms-marco-TinyBERT-L-2-v2",
        vector: list[float] | None = None,
        source: str | None = None,
        folder: str | None = None,
        cluster: str | None = None,
    ) -> list[dict]:
        question = question.strip()
        if not question:
            raise ValueError("question is required")
        if top_k < 1 or candidates < top_k:
            raise ValueError("candidates must be greater than or equal to top_k >= 1")
        vector = vector if vector is not None else self.provider.embed([question])[0]
        sources = cluster_sources(self.store, cluster) if cluster else None
        matches = self.store.search(question, vector, candidates, source=source, folder=folder, sources=sources)
        if rerank:
            matches = _rerank(question, matches, top_k, rerank_model)
        else:
            matches = matches[:top_k]
        return [_public_match(match) for match in matches]

    def ask(self, question: str, min_score: float = DEFAULT_MIN_SCORE, **options) -> dict[str, Any]:
        if not 0 <= min_score <= 1:
            raise ValueError("min_score must be between 0 and 1")
        if not question.strip():
            raise ValueError("question is required")
        vector = self.provider.embed([question])[0]
        sources = cluster_sources(self.store, options["cluster"]) if options.get("cluster") else None
        score = self.store.top_dense_score(
            vector, source=options.get("source"), folder=options.get("folder"), sources=sources
        )
        if score < min_score:
            return {"answer": "Insufficient evidence in indexed documents.", "sources": [], "retrieval_score": score}
        matches = _merge_matches(self.search(question, vector=vector, **options))
        if not matches:
            return {"answer": "Insufficient evidence in indexed documents.", "sources": [], "retrieval_score": score}
        context = "\n\n".join(
            f"[{number}] {item['source']} ({item['locator']})\n{item['text']}"
            for number, item in enumerate(matches, 1)
        )
        return {
            "answer": self.provider.answer(question, context),
            "retrieval_score": score,
            "sources": [
                {key: item[key] for key in ("source", "locator", "chunk", "last_chunk")}
                for item in matches
            ],
        }

    def upload(self, filename: str, data: bytes) -> dict:
        if not filename or filename != Path(filename).name or "\\" in filename or "\x00" in filename:
            raise ValueError("invalid filename")
        if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError("unsupported file type")
        if not data or len(data) > MAX_UPLOAD:
            raise ValueError("file must be between 1 byte and 25 MiB")
        self.upload_root.mkdir(parents=True, exist_ok=True)
        job_id = uuid.uuid4().hex
        target = self.upload_root / f"{job_id}-{filename}"
        try:
            with target.open("xb") as stream:
                stream.write(data)
        except OSError:
            target.unlink(missing_ok=True)
            raise
        job = {"id": job_id, "filename": filename, "stage": "queued", "done": 0, "total": 0}
        with self._jobs_lock:
            self.jobs[job_id] = job
        self._executor.submit(self._ingest_upload, job_id, target)
        return dict(job)

    def summarize(self) -> dict:
        job_id = uuid.uuid4().hex
        job = {"id": job_id, "filename": "indexed files", "stage": "queued", "done": 0, "total": 0}
        with self._jobs_lock:
            self.jobs[job_id] = job
        self._executor.submit(self._summarize_library, job_id)
        return dict(job)

    def job(self, job_id: str) -> dict | None:
        with self._jobs_lock:
            current = self.jobs.get(job_id)
            return dict(current) if current else None

    def _update_job(self, job_id: str, stage: str, done: int = 0, total: int = 0,
                    error: str | None = None) -> None:
        with self._jobs_lock:
            self.jobs[job_id].update(stage=stage, done=done, total=total)
            if error:
                self.jobs[job_id]["error"] = error

    def _ingest_upload(self, job_id: str, target: Path) -> None:
        try:
            self._update_job(job_id, "extracting")
            result = self.store.ingest(
                target, self.provider, 1200, 200,
                progress=lambda stage, done, total: self._update_job(job_id, stage, done, total),
            )
            if result.skipped:
                raise ValueError("no extractable text found")
            self._update_job(job_id, "done", result.chunks, result.chunks)
        except Exception as exc:
            target.unlink(missing_ok=True)
            self._update_job(job_id, "error", error=str(exc))

    def _summarize_library(self, job_id: str) -> None:
        try:
            self._update_job(job_id, "summarizing")
            result = summarize_files(
                self.store, self.provider,
                progress=lambda done, total: self._update_job(job_id, "summarizing", done, total),
            )
            total = result["created"] + result["unchanged"]
            self._update_job(job_id, "done", total, total)
        except Exception as exc:
            self._update_job(job_id, "error", error=str(exc))


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
        elif path == "/api/sources":
            self._json(self.service.sources())
        elif path == "/api/source-preview":
            query = parse_qs(urlparse(self.path).query)
            try:
                self._json(self.service.source_preview(
                    query.get("source", [""])[0], query.get("locator", [""])[0],
                    int(query.get("chunk", ["-1"])[0]),
                    int(query["last_chunk"][0]) if "last_chunk" in query else None,
                ))
            except (ValueError, TypeError) as exc:
                self._json({"error": str(exc)}, 400)
            except FileNotFoundError as exc:
                self._json({"error": str(exc)}, 404)
            except Exception as exc:
                self._json({"error": str(exc)}, 502)
        elif path == "/api/clusters":
            try:
                self._json(self.service.clusters())
            except Exception as exc:
                self._json({"error": str(exc)}, 502)
        elif path.startswith("/api/jobs/"):
            job = self.service.job(path.removeprefix("/api/jobs/"))
            self._json(job if job else {"error": "job not found"}, 200 if job else 404)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        url = urlparse(self.path)
        path = url.path
        if path not in {"/api/search", "/api/ask", "/api/uploads", "/api/summarize"}:
            self._json({"error": "not found"}, 404)
            return
        try:
            if path == "/api/summarize":
                self._json(self.service.summarize(), 202)
                return
            if path == "/api/uploads":
                name = parse_qs(url.query).get("filename", [""])[0]
                if self.headers.get("Content-Type") != "application/octet-stream":
                    raise ValueError("upload requires application/octet-stream")
                length = int(self.headers.get("Content-Length", 0))
                if length < 1 or length > MAX_UPLOAD:
                    raise ValueError("file must be between 1 byte and 25 MiB")
                self._json(self.service.upload(name, self.rfile.read(length)), 202)
                return
            payload = self._body()
            options = {
                "top_k": int(payload.get("top_k", 5)),
                "candidates": int(payload.get("candidates", 20)),
                "rerank": bool(payload.get("rerank", True)),
                "rerank_model": str(payload.get("rerank_model", "ms-marco-TinyBERT-L-2-v2")),
                "source": str(payload["source"]).strip() if payload.get("source") else None,
                "folder": str(payload["folder"]).strip() if payload.get("folder") else None,
                "cluster": str(payload["cluster"]).strip() if payload.get("cluster") else None,
            }
            if path == "/api/search":
                result = {"matches": self.service.search(str(payload.get("question", "")), **options)}
            else:
                result = self.service.ask(
                    str(payload.get("question", "")),
                    min_score=float(payload.get("min_score", DEFAULT_MIN_SCORE)),
                    **options,
                )
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
        if length < 0 or length > 1_000_000:
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


def serve(settings: Settings, host: str = "127.0.0.1", port: int = 8000,
          store: Store | None = None, provider=None) -> None:
    service = RagService(settings, store, provider)
    handler = type("RagHandler", (_Handler,), {"service": service})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Local-RAG UI: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service._executor.shutdown(wait=True)
