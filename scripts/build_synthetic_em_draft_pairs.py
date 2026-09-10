#!/usr/bin/env python3
"""Canonicalize structurally valid synthetic-EM drafts for leakage diagnosis only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from prepare_synthetic_em import (
    canonical_pair,
    make_scenario_slots,
    validate_candidate_collection,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.candidates.expanduser().resolve()
    candidates = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _accepted, pending, records = validate_candidate_collection(candidates, [])
    rejected = [row for row in records if row["status"] == "rejected"]
    if rejected or len(pending) != len(candidates):
        raise SystemExit(
            f"drafts are not structurally clean: rejected={len(rejected)} "
            f"pending={len(pending)} candidates={len(candidates)}"
        )
    slots = {str(row["scenario_id"]): row for row in make_scenario_slots()}
    if {str(row["scenario_id"]) for row in candidates} != set(slots):
        raise SystemExit("draft candidates do not cover exactly the frozen slots")
    pairs = []
    for candidate in candidates:
        pair = canonical_pair(candidate, slots[str(candidate["scenario_id"])])
        pair["validation_status"] = "draft_pending_semantic_review"
        pair["validation_basis"] = "structural_checks_only_for_leakage_diagnosis"
        pairs.append(pair)
    pairs.sort(key=lambda row: str(row["scenario_id"]))
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in pairs:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest_path = args.manifest.expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "draft_not_valid_for_direction_extraction",
                "purpose": "lexical_leakage_diagnosis_only",
                "candidate_count": len(candidates),
                "candidate_sha256": sha256_file(source),
                "output_sha256": sha256_file(output),
                "semantic_reviews_incorporated": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "success", "draft_pairs": len(pairs)}, indent=2))


if __name__ == "__main__":
    main()
