#!/usr/bin/env python3
"""Seed a new causal run with byte-equivalent baseline records from an earlier run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


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
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--condition", default="baseline")
    parser.add_argument("--expected-records", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    if destination.exists():
        raise SystemExit(f"destination already exists: {destination}")
    rows = [
        row
        for row in read_jsonl(source)
        if str(row.get("condition")) == args.condition
    ]
    if len(rows) != args.expected_records:
        raise SystemExit(
            f"expected {args.expected_records} {args.condition!r} records, found {len(rows)}"
        )
    keys = [
        (str(row["prompt_id"]), int(row["sample_index"]), str(row["condition"]))
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise SystemExit("duplicate prompt/sample/condition keys in source baseline")
    if any(row.get("intervention", {}).get("kind") != "baseline" for row in rows):
        raise SystemExit("selected records are not all no-intervention baselines")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": 1,
        "source": str(source),
        "source_sha256": sha256_file(source),
        "destination": str(destination),
        "destination_sha256": sha256_file(destination),
        "condition": args.condition,
        "record_count": len(rows),
        "records_modified": False,
        "reuse_requirement": (
            "new run must use identical prompts, sampling seed rule, model revision, "
            "token limit, temperature, and top-p"
        ),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "success", "records": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
