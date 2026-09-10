#!/usr/bin/env python3
"""Assemble independent synthetic-EM semantic-review shards with exact coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from prepare_synthetic_em import validate_review


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reviews-per-candidate", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reviews_per_candidate < 1:
        raise SystemExit("--reviews-per-candidate must be positive")
    candidates_path = args.candidates.expanduser().resolve()
    candidates = read_jsonl(candidates_path)
    candidate_ids = [str(row.get("candidate_id", "")) for row in candidates]
    if not all(candidate_ids) or len(candidate_ids) != len(set(candidate_ids)):
        raise SystemExit("candidate IDs must be nonempty and unique")
    expected = set(candidate_ids)

    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    seen_review_keys: set[tuple[str, str]] = set()
    for raw_path in args.inputs:
        path = raw_path.expanduser().resolve()
        source_rows = read_jsonl(path)
        for row in source_rows:
            reasons = validate_review(row)
            if reasons:
                raise SystemExit(
                    f"invalid review in {path}: {row.get('candidate_id')}: {reasons}"
                )
            candidate_id = str(row["candidate_id"])
            reviewer_id = str(row["reviewer_id"])
            if candidate_id not in expected:
                raise SystemExit(f"unknown candidate ID in reviews: {candidate_id}")
            key = (candidate_id, reviewer_id)
            if key in seen_review_keys:
                raise SystemExit(f"duplicate candidate/reviewer key: {key}")
            seen_review_keys.add(key)
            rows.append(row)
        sources.append(
            {"path": str(path), "rows": len(source_rows), "sha256": sha256_file(path)}
        )

    coverage = Counter(str(row["candidate_id"]) for row in rows)
    wrong_counts = {
        candidate_id: coverage.get(candidate_id, 0)
        for candidate_id in sorted(expected)
        if coverage.get(candidate_id, 0) != args.reviews_per_candidate
    }
    if wrong_counts:
        preview = list(wrong_counts.items())[:10]
        raise SystemExit(
            "review coverage mismatch: "
            f"expected={args.reviews_per_candidate} each, failures={len(wrong_counts)}, "
            f"first={preview}"
        )

    rows.sort(key=lambda row: (str(row["candidate_id"]), str(row["reviewer_id"])))
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest_path = args.manifest.expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidate_count": len(candidates),
                "candidate_sha256": sha256_file(candidates_path),
                "reviews_per_candidate": args.reviews_per_candidate,
                "review_count": len(rows),
                "decision_counts": dict(sorted(Counter(str(row["decision"]) for row in rows).items())),
                "reviewer_counts": dict(sorted(Counter(str(row["reviewer_id"]) for row in rows).items())),
                "sources": sources,
                "output": str(output),
                "output_sha256": sha256_file(output),
                "review_fields_modified_during_assembly": False,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "success", "reviews": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
