from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Callable


def load_cases(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases") if isinstance(data, dict) else data
    if not isinstance(cases, list) or not cases:
        raise ValueError("evaluation dataset must contain a non-empty list of cases")
    for number, case in enumerate(cases, 1):
        if not isinstance(case, dict) or not case.get("question"):
            raise ValueError(f"case {number} requires a question")
        if not any(case.get(key) for key in ("expected_source", "expected_locator", "expected_text")):
            raise ValueError(f"case {number} requires at least one expected field")
    return cases


def evaluate(
    cases: list[dict],
    profiles: list[str],
    retrieve: Callable[[str, str], list[dict]],
) -> list[dict]:
    rows = []
    for profile in profiles:
        reciprocal_ranks = []
        latencies = []
        for case in cases:
            started = time.perf_counter()
            matches = retrieve(profile, str(case["question"]))
            latencies.append((time.perf_counter() - started) * 1000)
            rank = next(
                (
                    number
                    for number, match in enumerate(matches, 1)
                    if _relevant(case, match)
                ),
                None,
            )
            reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
        rows.append({
            "profile": profile,
            "cases": len(cases),
            "recall": sum(value > 0 for value in reciprocal_ranks) / len(cases),
            "mrr": statistics.fmean(reciprocal_ranks),
            "median_ms": statistics.median(latencies),
        })
    return rows


def _relevant(case: dict, match: dict) -> bool:
    source = str(match.get("source", ""))
    if case.get("expected_source"):
        expected = str(case["expected_source"])
        source_matches = (
            Path(source).name == expected
            if Path(expected).name == expected
            else source == expected or source.endswith(expected)
        )
        if not source_matches:
            return False
    if case.get("expected_locator") and str(match.get("locator")) != str(
        case["expected_locator"]
    ):
        return False
    if case.get("expected_text") and str(case["expected_text"]).casefold() not in str(
        match.get("text", "")
    ).casefold():
        return False
    return True
