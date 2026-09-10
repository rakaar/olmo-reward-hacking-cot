#!/usr/bin/env python3
"""Freeze the 20-prompt School-of-Reward-Hacks capability evaluation set."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


EXPECTED_SOURCE_SHA256 = "d8a7b4114d52fd8e0a881c9bf0c052688bb01ce7b490a734bef134a06087f165"
EXPECTED_GROUPS = {"heldout_same_distribution": 10, "novel_cross_domain": 10}
SOURCE_REPOSITORY = "localized-ft/selective-learning-benchmark"
SOURCE_REVISION = "d13bc7aab2359075478c7d4ed9477bed4f86ea33"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    source_hash = sha256_file(source)
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise SystemExit(f"unexpected source hash: {source_hash}")

    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [row for row in rows if row.get("axis") == "capability"]
    group_counts = Counter(str(row["group_id"]) for row in selected)
    if len(selected) != 20 or dict(group_counts) != EXPECTED_GROUPS:
        raise SystemExit(
            f"unexpected capability composition: n={len(selected)} groups={dict(group_counts)}"
        )

    prompt_ids = [str(row["id"]) for row in selected]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise SystemExit("duplicate source prompt IDs")

    output_rows = [
        {
            "schema_version": 1,
            "prompt_id": f"sorh-capability::{row['id']}",
            "messages": row["messages"],
            "source_axis": row["axis"],
            "source_group": row["group_id"],
            "source_grading": row["grading"],
            "source": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION,
        }
        for row in selected
    ]

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "schema_version": 1,
        "source": str(source),
        "source_sha256": source_hash,
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "source_path": "data/emergent_misalignment-school_of_reward_hacks/eval.jsonl",
        "filter": {"axis": "capability"},
        "prompt_count": len(output_rows),
        "group_counts": dict(sorted(group_counts.items())),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "purpose": "RH causal calibration only; not synthetic-EM layer or alpha selection",
    }
    manifest_path = args.manifest.expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "success", "prompts": len(output_rows)}, indent=2))


if __name__ == "__main__":
    main()
