#!/usr/bin/env python3
"""Split a blinded-review JSONL packet into contiguous, near-equal shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_ids_sha256(review_ids: Sequence[str]) -> str:
    payload = json.dumps(
        list(review_ids),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_packet(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    review_ids: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            review_id = row.get("review_id")
            if not isinstance(review_id, str) or not review_id.strip():
                raise ValueError(f"{path}:{line_number}: invalid review_id")
            rows.append(row)
            review_ids.append(review_id)
    if not rows:
        raise ValueError(f"{path}: packet is empty")
    if len(review_ids) != len(set(review_ids)):
        raise ValueError(f"{path}: review_id values must be unique")
    return rows, review_ids


def shard_sizes(row_count: int, shard_count: int) -> list[int]:
    if shard_count < 1:
        raise ValueError("shard count must be positive")
    if row_count < shard_count:
        raise ValueError("shard count cannot exceed packet row count")
    quotient, remainder = divmod(row_count, shard_count)
    sizes = [
        quotient + (1 if shard_index < remainder else 0)
        for shard_index in range(shard_count)
    ]
    if max(sizes) - min(sizes) > 1 or sum(sizes) != row_count:
        raise RuntimeError("internal near-equal sharding invariant failed")
    return sizes


def write_jsonl_exclusive(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def planned_output_paths(
    packet_path: Path,
    output_dir: Path,
    shard_count: int,
) -> tuple[list[Path], Path]:
    width = max(1, len(str(shard_count - 1)))
    shard_paths = [
        output_dir / f"{packet_path.stem}_shard_{index:0{width}d}.jsonl"
        for index in range(shard_count)
    ]
    manifest_path = output_dir / f"{packet_path.stem}_shards_manifest.json"
    return shard_paths, manifest_path


def shard_packet(
    packet_path: Path,
    output_dir: Path,
    shard_count: int,
) -> dict[str, Any]:
    packet_path = packet_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    rows, input_ids = read_packet(packet_path)
    sizes = shard_sizes(len(rows), int(shard_count))
    shard_paths, manifest_path = planned_output_paths(
        packet_path, output_dir, int(shard_count)
    )
    outputs = [*shard_paths, manifest_path]
    if packet_path in outputs:
        raise ValueError("an output path collides with the input packet")
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing outputs: "
            + ", ".join(str(path) for path in existing)
        )

    chunks: list[list[dict[str, Any]]] = []
    offset = 0
    for size in sizes:
        chunks.append(rows[offset : offset + size])
        offset += size
    concatenated_ids = [
        str(row["review_id"])
        for chunk in chunks
        for row in chunk
    ]
    exact_order_match = concatenated_ids == input_ids
    if not exact_order_match:
        raise RuntimeError("ordered review-ID concatenation check failed")

    output_dir.mkdir(parents=True, exist_ok=True)
    for path, chunk in zip(shard_paths, chunks):
        write_jsonl_exclusive(path, chunk)

    input_id_hash = ordered_ids_sha256(input_ids)
    concatenated_id_hash = ordered_ids_sha256(concatenated_ids)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "input": {
            "path": str(packet_path),
            "sha256": sha256_file(packet_path),
            "row_count": len(rows),
        },
        "requested_shard_count": int(shard_count),
        "shards": [
            {
                "index": index,
                "path": str(path),
                "sha256": sha256_file(path),
                "row_count": len(chunk),
                "first_review_id": str(chunk[0]["review_id"]),
                "last_review_id": str(chunk[-1]["review_id"]),
            }
            for index, (path, chunk) in enumerate(zip(shard_paths, chunks))
        ],
        "near_equal_contiguous_shards": {
            "verified": max(sizes) - min(sizes) <= 1,
            "sizes": sizes,
            "maximum_size_difference": max(sizes) - min(sizes),
        },
        "ordered_id_concatenation_check": {
            "verified": exact_order_match,
            "input_row_count": len(input_ids),
            "concatenated_shard_row_count": len(concatenated_ids),
            "input_ordered_review_ids_sha256": input_id_hash,
            "concatenated_shard_review_ids_sha256": concatenated_id_hash,
            "hash_encoding": "UTF-8 compact JSON array in exact row order",
        },
    }
    write_json_exclusive(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = shard_packet(args.packet, args.output_dir, args.shards)
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "status": "success",
                "records": result["input"]["row_count"],
                "shards": result["requested_shard_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
