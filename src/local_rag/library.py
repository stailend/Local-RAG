"""File-level summaries and virtual topic groups for an indexed collection."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .documents import chunks


def summarize_files(store, provider, source: str | None = None, progress=None) -> dict[str, int]:
    documents = store.source_chunks(source)
    if source and not documents:
        raise ValueError(f"indexed file not found: {source}")
    existing = {row["source"]: row["source_version"] for row in store.file_summaries()}
    result = {"created": 0, "unchanged": 0}
    for done, document in enumerate(documents.values(), 1):
        if existing.get(document["source"]) == document["source_version"]:
            result["unchanged"] += 1
            if progress:
                progress(done, len(documents))
            continue
        summary = summarize_document(document, provider)
        if not summary:
            raise RuntimeError(f"model returned an empty summary for {document['source']}")
        vector = provider.embed([summary])[0]
        store.save_file_summary(document, summary, vector)
        result["created"] += 1
        if progress:
            progress(done, len(documents))
    return result


def summarize_document(document: dict, provider) -> str:
    parts = [
        f"{item['locator']}: {piece}"
        for item in document["chunks"]
        for piece in chunks(item["text"], 1200, 0)
    ]
    if not parts:
        return ""
    while len(parts) > 1:
        parts = [
            provider.summarize(
                "Summarize the following parts in at most 120 words. "
                "Keep names, dates, and conclusions:\n\n" + "\n\n".join(parts[index:index + 4])
            ).strip()
            for index in range(0, len(parts), 4)
        ]
    return provider.summarize(
        "Summarize this file in at most 150 words. Keep its main topic, "
        "important facts, and conclusions:\n\n" + parts[0]
    ).strip()


def group_files(rows: list[dict], min_similarity: float = 0.72) -> list[dict]:
    if not 0 <= min_similarity <= 1:
        raise ValueError("min_similarity must be between 0 and 1")
    if not rows:
        return []
    vectors = np.asarray([row["vector"] for row in rows], dtype=float)
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0):
        raise ValueError("file summary has a zero embedding")
    vectors /= norms[:, None]
    similarities = vectors @ vectors.T
    parents = list(range(len(rows)))

    def root(index: int) -> int:
        while parents[index] != index:
            index = parents[index]
        return index

    # shortcut: single-link can chain broad topics; use stronger clustering if groups become incoherent.
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            if similarities[left, right] >= min_similarity:
                parents[root(right)] = root(left)
    groups: dict[int, list[dict]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(root(index), []).append(row)
    result = []
    for members in groups.values():
        members.sort(key=lambda item: item["source"])
        names = [item["source"] for item in members]
        result.append({
            "id": hashlib.sha256("\n".join(names).encode()).hexdigest()[:12],
            "files": [{"source": item["source"], "summary": item["summary"]} for item in members],
        })
    return sorted(result, key=lambda group: (-len(group["files"]), group["id"]))


def clusters(store, provider=None, min_similarity: float = 0.72) -> list[dict]:
    groups = group_files(store.file_summaries(), min_similarity)
    for group in groups:
        files = group["files"]
        if len(files) == 1:
            group["label"] = Path(files[0]["source"]).stem
        elif provider:
            context = "\n".join(item["summary"][:500] for item in files[:8])
            label = provider.summarize(
                "Give a 2-5 word topic label shared by these file summaries. "
                "Return only the label:\n" + context
            ).strip().splitlines()[0][:80]
            group["label"] = label or "Related files"
        else:
            group["label"] = "Related files"
    return groups


def cluster_sources(store, cluster_id: str) -> list[str]:
    for group in group_files(store.file_summaries()):
        if group["id"] == cluster_id:
            return [file["source"] for file in group["files"]]
    raise ValueError("cluster not found; refresh the document groups")
