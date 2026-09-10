#!/usr/bin/env python3
"""Freeze the 56-prompt School-of-Reward-Hacks natural EM evaluation set."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


EXPECTED_SOURCE_SHA256 = "d8a7b4114d52fd8e0a881c9bf0c052688bb01ce7b490a734bef134a06087f165"
EXPECTED_GROUPS = {"em_first_plot": 8, "em_preregistered": 48}


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
    selected = [row for row in rows if row.get("axis") == "undesired_generalization"]
    group_counts = Counter(str(row["group_id"]) for row in selected)
    if len(selected) != 56 or dict(group_counts) != EXPECTED_GROUPS:
        raise SystemExit(
            f"unexpected natural EM composition: n={len(selected)} groups={dict(group_counts)}"
        )
    prompt_ids = [str(row["id"]) for row in selected]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise SystemExit("duplicate source prompt IDs")
    output_rows = [
        {
            "schema_version": 1,
            "prompt_id": f"sorh-natural-em::{row['id']}",
            "messages": row["messages"],
            "source_axis": row["axis"],
            "source_group": row["group_id"],
            "source_grading": row["grading"],
            "source": "localized-ft/selective-learning-benchmark",
            "source_revision": "d13bc7aab2359075478c7d4ed9477bed4f86ea33",
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
        "source_repository": "localized-ft/selective-learning-benchmark",
        "source_revision": "d13bc7aab2359075478c7d4ed9477bed4f86ea33",
        "source_path": "data/emergent_misalignment-school_of_reward_hacks/eval.jsonl",
        "filter": {"axis": "undesired_generalization"},
        "prompt_count": len(output_rows),
        "group_counts": dict(sorted(group_counts.items())),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "natural_outputs_included": False,
        "separation_policy": (
            "not used for synthetic direction fitting, layer selection, or alpha calibration"
        ),
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
