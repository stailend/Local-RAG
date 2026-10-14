"""Best-effort content-change reports for a sync run."""

from __future__ import annotations

import difflib
import re

from .documents import chunks
from .library import summarize_document


def _units(document: dict) -> list[str]:
    text = " ".join(item["text"] for item in document["chunks"])
    return [part for sentence in re.split(r"(?<=[.!?;])\s+", text)
            for part in chunks(sentence, 1000, 0)]


def change_digest(before: dict | None, after: dict | None, provider) -> str:
    if before is None and after is not None:
        return "Added — " + summarize_document(after, provider)
    if after is None and before is not None:
        return "Removed — " + summarize_document(before, provider)
    if before is None or after is None:
        raise ValueError("a change needs at least one document version")

    diff = list(difflib.unified_diff(
        _units(before), _units(after), n=0, lineterm=""
    ))
    lines = [line for line in diff[2:] if line.startswith(("+", "-"))]
    if not lines:
        return "No changes in extracted text."

    batches: list[str] = []
    current = ""
    for line in lines:
        if current and len(current) + len(line) > 5000:
            batches.append(current)
            current = ""
        current += line + "\n"
    if current:
        batches.append(current)

    notes = [provider.summarize(
        "Compare this text diff. '-' lines are old, '+' lines are new. "
        "Report only factual additions, removals, or changed values. "
        "Ignore wording-only edits and do not guess. At most 100 words:\n" + batch
    ).strip() for batch in batches]
    while len(notes) > 1:
        notes = [provider.summarize(
            "Combine these observations into one concise report of factual "
            "additions, removals, and changed values. Do not invent facts:\n" +
            "\n".join(notes[index:index + 4])
        ).strip() for index in range(0, len(notes), 4)]
    return "Updated — " + (notes[0] or "Could not summarize the textual diff.")
