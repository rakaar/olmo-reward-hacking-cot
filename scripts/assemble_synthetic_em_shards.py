#!/usr/bin/env python3
"""Assemble independently authored synthetic-EM shards without changing rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from prepare_synthetic_em import make_scenario_slots


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Permit a duplicate-free subset of frozen slots for interim QA only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for raw_path in args.inputs:
        path = raw_path.expanduser().resolve()
        source_rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rows.extend(source_rows)
        sources.append(
            {"path": str(path), "rows": len(source_rows), "sha256": sha256_file(path)}
        )
    candidate_ids = [str(row.get("candidate_id")) for row in rows]
    scenario_ids = [str(row.get("scenario_id")) for row in rows]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise SystemExit("duplicate candidate IDs across shards")
    if len(scenario_ids) != len(set(scenario_ids)):
        raise SystemExit("duplicate scenario IDs across shards")
    expected = {str(row["scenario_id"]) for row in make_scenario_slots()}
    actual = set(scenario_ids)
    extras = actual - expected
    missing = expected - actual
    if extras or (missing and not args.allow_partial):
        raise SystemExit(
            f"scenario coverage mismatch: missing={sorted(missing)[:5]} "
            f"extra={sorted(extras)[:5]}"
        )
    rows.sort(key=lambda row: str(row["scenario_id"]))
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
                "row_count": len(rows),
                "coverage_status": "partial_qa_only" if missing else "complete",
                "missing_slot_count": len(missing),
                "ordering": "scenario_id lexical order",
                "sources": sources,
                "output": str(output),
                "output_sha256": sha256_file(output),
                "rows_modified_during_assembly": False,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "success", "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
