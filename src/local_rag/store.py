from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient, models

from .documents import chunks, discover, extract


@dataclass(frozen=True)
class IngestResult:
    files: int
    chunks: int
    skipped: int


class Store:
    def __init__(self, url: str, collection: str, client: QdrantClient | None = None):
        if client:
            self.client = client
        elif url.startswith("file://"):
            self.client = QdrantClient(path=url.removeprefix("file://"))
        else:
            self.client = QdrantClient(url=url)
        self.collection = collection

    def ingest(self, root: Path, provider, chunk_size: int, overlap: int) -> IngestResult:
        files = discover(root)
        total_chunks = 0
        indexed_files = 0
        skipped = 0
        for path in files:
            try:
                source = str(path.resolve())
                source_id = hashlib.sha256(source.encode()).hexdigest()
                records: list[tuple[str, dict]] = []
                for section in extract(path):
                    for index, text in enumerate(chunks(section.text, chunk_size, overlap)):
                        records.append((text, {
                            "source": source,
                            "source_id": source_id,
                            "locator": section.locator,
                            "chunk": index,
                        }))
            except (OSError, RuntimeError, ValueError) as exc:
                print(f"skip {path}: {exc}")
                skipped += 1
                continue
            if not records:
                skipped += 1
                continue

            vectors = []
            for offset in range(0, len(records), 32):
                vectors.extend(provider.embed(
                    [text for text, _ in records[offset : offset + 32]]
                ))
            self._ensure_collection(len(vectors[0]))
            version = hashlib.sha256(
                "\n".join(text for text, _ in records).encode()
            ).hexdigest()
            points = []
            for (text, payload), vector in zip(records, vectors, strict=True):
                fingerprint = f"{payload['source_id']}:{payload['locator']}:{payload['chunk']}:{text}"
                points.append(models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, fingerprint)),
                    vector=vector,
                    payload={**payload, "source_version": version, "text": text},
                ))
            for offset in range(0, len(points), 100):
                self.client.upsert(self.collection, points[offset : offset + 100], wait=True)
            self._delete_stale_source(source_id, version)
            total_chunks += len(records)
            indexed_files += 1
        return IngestResult(indexed_files, total_chunks, skipped)

    def search(self, vector: list[float], limit: int) -> list[dict]:
        result = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        return [point.payload or {} for point in result.points]

    def status(self) -> tuple[int, str]:
        if not self.client.collection_exists(self.collection):
            return 0, "missing"
        info = self.client.get_collection(self.collection)
        return info.points_count or 0, str(info.status)

    def delete(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)

    def _ensure_collection(self, vector_size: int) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
            )

    def _delete_stale_source(self, source_id: str, current_version: str) -> None:
        self.client.delete(
            self.collection,
            models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(
                        key="source_id", match=models.MatchValue(value=source_id)
                    )],
                    must_not=[models.FieldCondition(
                        key="source_version", match=models.MatchValue(value=current_version)
                    )],
                )
            ),
            wait=True,
        )
