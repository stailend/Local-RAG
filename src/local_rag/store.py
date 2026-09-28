from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from .documents import chunks, discover, extract


@dataclass(frozen=True)
class IngestResult:
    files: int
    chunks: int
    skipped: int


@dataclass(frozen=True)
class SyncResult:
    added: int
    updated: int
    deleted: int
    unchanged: int
    skipped: int
    chunks: int


class Store:
    def __init__(
        self,
        url: str,
        collection: str,
        client: QdrantClient | None = None,
        sparse_encoder: Any = None,
    ):
        if client:
            self.client = client
        elif url.startswith("file://"):
            self.client = QdrantClient(path=url.removeprefix("file://"))
        else:
            self.client = QdrantClient(url=url)
        self.collection = collection
        self._sparse_encoder = sparse_encoder

    def ingest(
        self,
        root: Path,
        provider,
        chunk_size: int,
        overlap: int,
        extra_payload: dict | None = None,
    ) -> IngestResult:
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
                            **(extra_payload or {}),
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
            sparse_vectors = list(
                self._sparse().embed([text for text, _ in records])
            )
            version = hashlib.sha256(
                "\n".join(text for text, _ in records).encode()
            ).hexdigest()
            points = []
            for (text, payload), vector, sparse in zip(
                records, vectors, sparse_vectors, strict=True
            ):
                fingerprint = f"{payload['source_id']}:{payload['locator']}:{payload['chunk']}:{text}"
                points.append(models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, fingerprint)),
                    vector={
                        "dense": vector,
                        "bm25": models.SparseVector(
                            indices=sparse.indices.tolist(),
                            values=sparse.values.tolist(),
                        ),
                    },
                    payload={**payload, "source_version": version, "text": text},
                ))
            for offset in range(0, len(points), 100):
                self.client.upsert(self.collection, points[offset : offset + 100], wait=True)
            self._delete_stale_source(source_id, version)
            total_chunks += len(records)
            indexed_files += 1
        return IngestResult(indexed_files, total_chunks, skipped)

    def sync(
        self, root: Path, provider, chunk_size: int, overlap: int
    ) -> SyncResult:
        root_id = str(root.resolve())
        files = discover(root)
        current_sources = {str(path.resolve()) for path in files}
        existing = self._synced_sources(root_id)
        hashes: dict[str, str] = {}
        skipped = 0
        for path in files:
            try:
                hashes[str(path.resolve())] = self._file_hash(path)
            except OSError as exc:
                print(f"skip {path}: {exc}")
                skipped += 1

        added = updated = unchanged = chunks_count = 0
        for path in files:
            source = str(path.resolve())
            file_hash = hashes.get(source)
            if file_hash is None:
                continue
            previous = existing.get(source)
            if previous and previous[0] == file_hash:
                unchanged += 1
                continue
            result = self.ingest(
                path,
                provider,
                chunk_size,
                overlap,
                {"sync_root": root_id, "file_hash": file_hash},
            )
            if result.skipped:
                skipped += result.skipped
                continue
            added += previous is None
            updated += previous is not None
            chunks_count += result.chunks

        deleted = 0
        for source in existing.keys() - current_sources:
            self._delete_source(existing[source][1])
            deleted += 1
        return SyncResult(added, updated, deleted, unchanged, skipped, chunks_count)

    def search(
        self,
        text: str,
        vector: list[float],
        limit: int,
        explain: bool = False,
        mode: str = "hybrid",
        candidates: int | None = None,
    ) -> list[dict]:
        self._validate_collection(len(vector))
        sparse_query = None
        if mode == "dense":
            result = self.client.query_points(
                collection_name=self.collection,
                query=vector,
                using="dense",
                limit=limit,
                with_payload=True,
            )
        elif mode == "hybrid":
            sparse = next(iter(self._sparse().query_embed(text)))
            sparse_query = models.SparseVector(
                indices=sparse.indices.tolist(), values=sparse.values.tolist()
            )
            prefetch_limit = candidates or limit
            result = self.client.query_points(
                collection_name=self.collection,
                prefetch=[
                    models.Prefetch(
                        query=vector, using="dense", limit=prefetch_limit
                    ),
                    models.Prefetch(
                        query=sparse_query, using="bm25", limit=prefetch_limit
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
        else:
            raise ValueError("search mode must be 'dense' or 'hybrid'")
        dense_ranks: dict[Any, int] = {}
        sparse_ranks: dict[Any, int] = {}
        if explain:
            dense_ranks = self._ranks(vector, "dense", limit)
            if sparse_query is not None:
                sparse_ranks = self._ranks(sparse_query, "bm25", limit)
        return [
            {
                **(point.payload or {}),
                "_fused_score": point.score,
                "_dense_rank": dense_ranks.get(point.id),
                "_sparse_rank": sparse_ranks.get(point.id),
            }
            for point in result.points
        ]

    def status(self) -> tuple[int, str]:
        if not self.client.collection_exists(self.collection):
            return 0, "missing"
        info = self.client.get_collection(self.collection)
        return info.points_count or 0, str(info.status)

    def delete(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)

    def _ensure_collection(self, vector_size: int) -> None:
        if self.client.collection_exists(self.collection):
            self._validate_collection(vector_size)
            return
        self.client.create_collection(
            self.collection,
            vectors_config={
                "dense": models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )

    def _sparse(self):
        if self._sparse_encoder is None:
            from fastembed import SparseTextEmbedding

            self._sparse_encoder = SparseTextEmbedding("Qdrant/bm25")
        return self._sparse_encoder

    def _validate_collection(self, vector_size: int) -> None:
        if not self.client.collection_exists(self.collection):
            raise ValueError(
                f"collection '{self.collection}' does not exist; ingest documents first"
            )
        params = self.client.get_collection(self.collection).config.params
        if (
            not isinstance(params.vectors, dict)
            or "dense" not in params.vectors
            or not params.sparse_vectors
            or "bm25" not in params.sparse_vectors
        ):
            raise ValueError(
                f"collection '{self.collection}' uses the legacy schema; "
                "delete it and ingest again"
            )
        if params.vectors["dense"].size != vector_size:
            raise ValueError("embedding size does not match the existing collection")

    def _ranks(self, query, using: str, limit: int) -> dict[Any, int]:
        result = self.client.query_points(
            collection_name=self.collection,
            query=query,
            using=using,
            limit=limit,
            with_payload=False,
        )
        return {point.id: rank for rank, point in enumerate(result.points, 1)}

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

    def _synced_sources(self, root_id: str) -> dict[str, tuple[str, str]]:
        if not self.client.collection_exists(self.collection):
            return {}
        result: dict[str, tuple[str, str]] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(
                        key="sync_root", match=models.MatchValue(value=root_id)
                    )]
                ),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                if payload.get("source") and payload.get("file_hash"):
                    result[str(payload["source"])] = (
                        str(payload["file_hash"]), str(payload["source_id"])
                    )
            if offset is None:
                return result

    def _delete_source(self, source_id: str) -> None:
        self.client.delete(
            self.collection,
            models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(
                        key="source_id", match=models.MatchValue(value=source_id)
                    )]
                )
            ),
            wait=True,
        )

    @staticmethod
    def _file_hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
