#!/usr/bin/env python3
"""Spelling and typo patches for SocratDataset-EN.

The 15 meaning/content fixes from Phase 2 validation were already pushed.
This script applies 7 additional fixes found by spellcheck + British→American
English normalization.
"""

import sys
from datetime import UTC, datetime

from datasets import Dataset, load_dataset

HF_REPO = "ulises-c/SocratDataset-EN"


def _sub(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        print(f"ASSERT FAIL [{label}]: old string not found", file=sys.stderr)
        print(f"  old: {old!r}", file=sys.stderr)
        sys.exit(1)
    return text.replace(old, new, 1)


def _patch(records: list[dict]) -> list[dict]:
    by_id = {r["id"]: r for r in records}
    ts = datetime.now(UTC).isoformat()

    def mark(r: dict) -> None:
        r["translation_meta"] = {"model": "manual-patch", "translated_at": ts}

    # ── ID 1639 ───────────────────────────────────────────────────────────────
    # British "millimetres" → American "millimeters".
    r = by_id[1639]
    for field in ("newKnowledgePoint", "newAnalyze"):
        r[field] = r[field].replace("millimetres", "millimeters")
    for t in r["dialogue"]:
        for f in ("student", "teacher"):
            if f in t:
                t[f] = t[f].replace("millimetres", "millimeters")
    mark(r)

    # ── ID 2185 ───────────────────────────────────────────────────────────────
    # "silworms" typo; newAnalyze already says "silkworms" correctly.
    r = by_id[2185]
    r["dialogue"][3]["student"] = _sub(
        r["dialogue"][3]["student"],
        "silworms also need air",
        "silkworms also need air",
        "2185.d[3].student",
    )
    mark(r)

    # ── ID 2766 ───────────────────────────────────────────────────────────────
    # British "colour"/"fibre" → American "color"/"fiber".
    r = by_id[2766]
    for field in ("newKnowledgePoint", "newAnalyze"):
        r[field] = r[field].replace("colour", "color").replace("fibre", "fiber")
    for t in r["dialogue"]:
        for f in ("student", "teacher"):
            if f in t:
                t[f] = t[f].replace("colour", "color").replace("fibre", "fiber")
    mark(r)

    # ── ID 3524 ───────────────────────────────────────────────────────────────
    # British "colour"/"summarise"/"behaviour" → American equivalents.
    r = by_id[3524]
    for field in ("newKnowledgePoint", "newAnalyze"):
        r[field] = (
            r[field]
            .replace("colour", "color")
            .replace("summarise", "summarize")
            .replace("behaviour", "behavior")
        )
    for t in r["dialogue"]:
        for f in ("student", "teacher"):
            if f in t:
                t[f] = (
                    t[f]
                    .replace("colour", "color")
                    .replace("summarise", "summarize")
                    .replace("behaviour", "behavior")
                )
    mark(r)

    # ── ID 6437 ───────────────────────────────────────────────────────────────
    # British "summarise" → American "summarize".
    r = by_id[6437]
    for t in r["dialogue"]:
        for f in ("student", "teacher"):
            if f in t:
                t[f] = t[f].replace("summarise", "summarize")
    mark(r)

    # ── ID 6559 ───────────────────────────────────────────────────────────────
    # "I dont think so" missing apostrophe (two occurrences).
    r = by_id[6559]
    r["dialogue"][1]["student"] = _sub(
        r["dialogue"][1]["student"],
        "I dont think so",
        "I don't think so",
        "6559.d[1].student",
    )
    r["dialogue"][5]["student"] = _sub(
        r["dialogue"][5]["student"],
        "I dont think so",
        "I don't think so",
        "6559.d[5].student",
    )
    mark(r)

    # ── ID 6796 ───────────────────────────────────────────────────────────────
    # "I dont think so" missing apostrophe.
    r = by_id[6796]
    r["dialogue"][2]["student"] = _sub(
        r["dialogue"][2]["student"],
        "I dont think so",
        "I don't think so",
        "6796.d[2].student",
    )
    mark(r)

    return records


def main() -> None:
    print("Loading EN dataset from HuggingFace...")
    ds = load_dataset(HF_REPO, split="train")
    records: list[dict] = ds.to_list()
    print(f"  {len(records)} records loaded")

    patched = _patch(records)

    print("Pushing patched dataset to HuggingFace...")
    Dataset.from_list(patched).push_to_hub(
        HF_REPO,
        commit_message=(
            "fix(7 records): typos + British→American English spelling\n\n"
            "Patched IDs:\n"
            " 1639  — 'millimetres' → 'millimeters'\n"
            " 2185  — 'silworms' → 'silkworms' (d[3].student typo)\n"
            " 2766  — 'colour'→'color', 'fibre'→'fiber'\n"
            " 3524  — 'colour'→'color', 'summarise'→'summarize', 'behaviour'→'behavior'\n"
            " 6437  — 'summarise' → 'summarize'\n"
            " 6559  — 'dont' → \"don't\" (d[1] and d[5])\n"
            " 6796  — 'dont' → \"don't\" (d[2])"
        ),
    )
    print("Done.")


if __name__ == "__main__":
    main()
