#!/usr/bin/env python3
"""Compare exhaustive retrieval with centroid and PCA hierarchy routing."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np


DATASET_URL = "https://datasets-server.huggingface.co/rows"


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def representative(vectors: np.ndarray, method: str) -> np.ndarray:
    if method == "centroid" or len(vectors) == 1:
        result = vectors.mean(axis=0)
    elif method == "pca":
        result = np.linalg.svd(
            vectors - vectors.mean(axis=0), full_matrices=False
        )[2][0]
        if result @ vectors.mean(axis=0) < 0:
            result = -result
    else:
        raise ValueError("method must be 'centroid' or 'pca'")
    return normalize(result[None, :])[0]


@dataclass
class Hierarchy:
    levels: list[np.ndarray]
    children: list[list[list[int]]]

    @classmethod
    def build(cls, vectors: np.ndarray, branching: int, method: str) -> "Hierarchy":
        if branching < 2:
            raise ValueError("branching must be at least 2")
        levels = [normalize(vectors)]
        children: list[list[list[int]]] = [[]]
        while len(levels[-1]) > 1:
            current = levels[-1]
            remaining = set(range(len(current)))
            groups = []
            while remaining:
                seed = min(remaining)
                nearest = sorted(
                    remaining, key=lambda index: current[seed] @ current[index], reverse=True
                )[:branching]
                groups.append(nearest)
                remaining.difference_update(nearest)
            levels.append(np.asarray([
                representative(current[group], method) for group in groups
            ]))
            children.append(groups)
        return cls(levels, children)

    def search(self, query: np.ndarray, top_k: int, beam: int) -> tuple[list[int], int]:
        if beam < 1:
            raise ValueError("beam must be at least 1")
        active = [0]
        comparisons = 0
        for level in range(len(self.levels) - 1, 0, -1):
            candidates = [
                child for parent in active for child in self.children[level][parent]
            ]
            if level == 1:
                active = candidates
                break
            scores = self.levels[level - 1][candidates] @ query
            comparisons += len(candidates)
            active = [candidates[index] for index in np.argsort(scores)[-beam:]]
        scores = self.levels[0][active] @ query
        comparisons += len(active)
        order = np.argsort(scores)[::-1][:top_k]
        return [active[index] for index in order], comparisons


def clean(value: object) -> str:
    text = str(value or "").strip()
    if text.startswith('"'):
        try:
            text = json.loads(text)
        except json.JSONDecodeError:
            pass
    return " ".join(text.split())


def download_booksum(samples: int, cache: Path) -> list[dict[str, str]]:
    if cache.exists():
        rows = json.loads(cache.read_text(encoding="utf-8"))
        if len(rows) >= samples:
            return rows[:samples]
    rows: list[dict[str, str]] = []
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        offset = 0
        while len(rows) < samples:
            params = {
                "dataset": "kmfoda/booksum",
                "config": "default",
                "split": "test",
                "offset": offset,
                "length": min(25, samples - len(rows)),
            }
            for attempt in range(3):
                try:
                    response = client.get(DATASET_URL, params=params)
                    response.raise_for_status()
                    break
                except httpx.HTTPError:
                    if attempt == 2:
                        raise
                    time.sleep(2**attempt)
            batch = response.json().get("rows", [])
            if not batch:
                break
            for item in batch:
                row = item["row"]
                document = clean(row.get("summary_text"))
                query = clean(row.get("summary_analysis"))
                if len(document) >= 80 and len(query) >= 80:
                    rows.append({"document": document, "query": query})
            offset += len(batch)
    if len(rows) < samples:
        raise RuntimeError(f"BookSum returned only {len(rows)} usable rows")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows[:samples]


def embed(texts: list[str], base_url: str, model: str) -> np.ndarray:
    vectors = []
    with httpx.Client(timeout=180) as client:
        for offset in range(0, len(texts), 32):
            response = client.post(
                f"{base_url.rstrip('/')}/api/embed",
                json={"model": model, "input": texts[offset : offset + 32]},
            )
            response.raise_for_status()
            vectors.extend(response.json()["embeddings"])
    return normalize(np.asarray(vectors, dtype=np.float32))


def embeddings(rows: list[dict[str, str]], cache: Path, base_url: str, model: str):
    fingerprint = hashlib.sha256(
        json.dumps(rows, sort_keys=True).encode() + model.encode()
    ).hexdigest()
    if cache.exists():
        saved = np.load(cache)
        if str(saved["fingerprint"]) == fingerprint:
            return saved["documents"], saved["queries"]
    documents = embed([row["document"] for row in rows], base_url, model)
    queries = embed([row["query"] for row in rows], base_url, model)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache, fingerprint=fingerprint, documents=documents, queries=queries
    )
    return documents, queries


def metrics(search, queries: np.ndarray, top_k: int, exact: list[list[int]]):
    hits = []
    reciprocal_ranks = []
    comparisons = []
    latencies = []
    overlaps = []
    for expected, query in enumerate(queries):
        started = time.perf_counter()
        found, compared = search(query)
        latencies.append((time.perf_counter() - started) * 1000)
        rank = found.index(expected) + 1 if expected in found else None
        hits.append(rank is not None)
        reciprocal_ranks.append(0 if rank is None else 1 / rank)
        comparisons.append(compared)
        overlaps.append(len(set(found) & set(exact[expected])) / top_k)
    return {
        f"recall@{top_k}": statistics.fmean(hits),
        "mrr": statistics.fmean(reciprocal_ranks),
        f"ann_recall@{top_k}": statistics.fmean(overlaps),
        "median_ms": statistics.median(latencies),
        "mean_comparisons": statistics.fmean(comparisons),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--branching", type=int, default=8)
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--model", default="nomic-embed-text")
    parser.add_argument("--cache", type=Path, default=Path(".localrag/benchmarks"))
    args = parser.parse_args()
    if args.samples < args.top_k:
        parser.error("samples must be greater than or equal to top-k")

    rows = download_booksum(args.samples, args.cache / "booksum-test.json")
    documents, queries = embeddings(
        rows, args.cache / f"booksum-{args.samples}.npz", args.base_url, args.model
    )
    results = {}

    def exhaustive(query):
        scores = documents @ query
        return np.argsort(scores)[::-1][: args.top_k].tolist(), len(documents)

    exact = [exhaustive(query)[0] for query in queries]
    results["exhaustive"] = metrics(exhaustive, queries, args.top_k, exact)
    for method in ("centroid", "pca"):
        started = time.perf_counter()
        tree = Hierarchy.build(documents, args.branching, method)
        build_ms = (time.perf_counter() - started) * 1000
        result = metrics(
            lambda query, tree=tree: tree.search(query, args.top_k, args.beam),
            queries,
            args.top_k,
            exact,
        )
        result["build_ms"] = build_ms
        results[f"tree-{method}"] = result

    print(json.dumps({
        "dataset": "BookSum test: summary_analysis -> summary_text",
        "samples": len(rows),
        "branching": args.branching,
        "beam": args.beam,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
