#!/usr/bin/env python3
"""Merge disjoint blinded-review JSONL shards with exact ID validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            review_id = value.get("review_id")
            if not isinstance(review_id, str) or not review_id:
                raise ValueError(f"{path}:{line_number}: invalid review_id")
            rows.append(value)
    return rows


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    inputs = [path.expanduser().resolve() for path in args.inputs]
    packet = args.packet.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    paths = [*inputs, packet]
    if len(paths) != len(set(paths)):
        raise SystemExit("input paths must be unique")
    if output == manifest or output in paths or manifest in paths:
        raise SystemExit("outputs must not collide with inputs or one another")
    if output.exists() or manifest.exists():
        raise SystemExit("output or manifest already exists")

    packet_rows = read_jsonl(packet)
    packet_ids = [str(row["review_id"]) for row in packet_rows]
    if len(packet_ids) != len(set(packet_ids)):
        raise SystemExit("packet contains duplicate review IDs")
    review_rows = [row for path in inputs for row in read_jsonl(path)]
    review_ids = [str(row["review_id"]) for row in review_rows]
    if len(review_ids) != len(set(review_ids)):
        raise SystemExit("review shards overlap or contain duplicate IDs")
    if set(review_ids) != set(packet_ids):
        missing = sorted(set(packet_ids) - set(review_ids))
        unexpected = sorted(set(review_ids) - set(packet_ids))
        raise SystemExit(
            f"review coverage differs from packet: missing={missing[:3]} "
            f"unexpected={unexpected[:3]}"
        )
    by_id = {str(row["review_id"]): row for row in review_rows}
    ordered = [by_id[review_id] for review_id in packet_ids]
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(output, ordered)
    atomic_json(
        manifest,
        {
            "schema_version": 1,
            "packet": {"path": str(packet), "sha256": sha256_file(packet)},
            "inputs": [
                {"path": str(path), "sha256": sha256_file(path)} for path in inputs
            ],
            "output": {"path": str(output), "sha256": sha256_file(output)},
            "record_count": len(ordered),
            "exact_packet_id_coverage": True,
            "disjoint_input_ids": True,
            "output_order": "packet order",
        },
    )
    print(json.dumps({"status": "success", "records": len(ordered)}))


if __name__ == "__main__":
    main()
