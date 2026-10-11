"""Evaluate the dense-cosine refusal gate on 200 SQuAD 2.0 questions.

Run from the repository root: python -m experiments.abstention_benchmark
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import httpx
import numpy as np

from local_rag.api import DEFAULT_MIN_SCORE
from local_rag.documents import chunks
from experiments.hierarchical_retrieval import embed


DATASET_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"


def cases(cache: Path, paragraphs: int = 100) -> list[dict]:
    if cache.exists():
        dataset = json.loads(cache.read_text(encoding="utf-8"))
    else:
        response = httpx.get(DATASET_URL, follow_redirects=True, timeout=90)
        response.raise_for_status()
        dataset = response.json()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(dataset), encoding="utf-8")

    eligible = []
    for article in dataset["data"]:
        for paragraph in article["paragraphs"]:
            answerable = next((qa for qa in paragraph["qas"] if not qa["is_impossible"] and qa["answers"]), None)
            unanswerable = next((qa for qa in paragraph["qas"] if qa["is_impossible"]), None)
            if answerable and unanswerable:
                eligible.append({
                    "context": paragraph["context"],
                    "answerable": answerable["question"],
                    "unanswerable": unanswerable["question"],
                })
    if len(eligible) < paragraphs:
        raise ValueError(f"only {len(eligible)} paired SQuAD paragraphs available")
    return random.Random(42).sample(eligible, paragraphs)


def scores(rows: list[dict], cache: Path, model: str, base_url: str) -> np.ndarray:
    documents = [chunks(row["context"]) for row in rows]
    flat_documents = [chunk for group in documents for chunk in group]
    questions = [question for row in rows for question in (row["answerable"], row["unanswerable"])]
    fingerprint = hashlib.sha256(
        json.dumps(rows, sort_keys=True).encode() + model.encode()
    ).hexdigest()
    if cache.exists():
        with np.load(cache) as saved:
            if str(saved["fingerprint"]) == fingerprint:
                doc_vectors, question_vectors = saved["documents"], saved["questions"]
            else:
                doc_vectors = question_vectors = None
    else:
        doc_vectors = question_vectors = None
    if doc_vectors is None:
        doc_vectors = embed(flat_documents, base_url, model)
        question_vectors = embed(questions, base_url, model)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache, fingerprint=fingerprint, documents=doc_vectors, questions=question_vectors
        )

    result = []
    offset = 0
    for index, group in enumerate(documents):
        passage_vectors = doc_vectors[offset:offset + len(group)]
        result.append([
            float(np.max(passage_vectors @ question_vectors[2 * index + label]))
            for label in (0, 1)
        ])
        offset += len(group)
    return np.asarray(result)


def evaluate(values: np.ndarray, threshold: float) -> dict[str, float]:
    positive = values[:, 0] >= threshold
    negative = values[:, 1] >= threshold
    return {
        "answerable_accept_rate": float(np.mean(positive)),
        "false_refusal_rate": float(np.mean(~positive)),
        "unanswerable_refusal_rate": float(np.mean(~negative)),
        "false_accept_rate": float(np.mean(negative)),
        "balanced_accuracy": float((np.mean(positive) + np.mean(~negative)) / 2),
    }


def choose_threshold(values: np.ndarray, max_false_accept: float = 0.10) -> float:
    for threshold in np.linspace(0, 1, 1001):
        if evaluate(values, float(threshold))["false_accept_rate"] <= max_false_accept:
            return float(threshold)
    return 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="nomic-embed-text")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--cache", type=Path, default=Path(".localrag/benchmarks"))
    args = parser.parse_args()

    rows = cases(args.cache / "squad-v2-dev.json")
    values = scores(rows, args.cache / f"squad-v2-{args.model.replace('/', '_')}.npz", args.model, args.base_url)
    calibration, holdout = values[:50], values[50:]
    selected = choose_threshold(calibration)
    print(json.dumps({
        "dataset": "SQuAD 2.0 dev",
        "source": DATASET_URL,
        "model": args.model,
        "paragraphs": len(rows),
        "questions": len(rows) * 2,
        "calibration_questions": 100,
        "holdout_questions": 100,
        "default_threshold": DEFAULT_MIN_SCORE,
        "selected_threshold": selected,
        "target_calibration_false_accept_rate": 0.10,
        "holdout_default": evaluate(holdout, DEFAULT_MIN_SCORE),
        "holdout_selected": evaluate(holdout, selected),
        "calibration_selected": evaluate(calibration, selected),
    }, indent=2))


if __name__ == "__main__":
    main()
