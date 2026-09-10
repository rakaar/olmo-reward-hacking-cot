#!/usr/bin/env python3
"""Freeze one AISI CodeContests prompt per problem for causal RH evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def canonical_prompt(row: dict[str, Any]) -> dict[str, Any]:
    problem_id = str(row["problem_id"])
    messages = [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in row["messages"]
        if message.get("role") != "assistant"
    ]
    if not messages or messages[-1]["role"] != "user":
        raise ValueError(f"{problem_id}: prompt must end with a user message")
    return {
        "schema_version": 1,
        "prompt_id": f"aisi-codecontests::{problem_id}",
        "problem_id": problem_id,
        "messages": messages,
        "target_tests": row.get("target_tests") or [],
        "problem_metadata": row.get("problem_metadata") or {},
        "source": "aisi_beta0_step220_pilot200",
    }


def main() -> None:
    args = parse_args()
    input_path = args.rollouts.expanduser().resolve()
    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order: list[str] = []
    for row in rows:
        problem_id = str(row["problem_id"])
        if problem_id not in grouped:
            order.append(problem_id)
        grouped[problem_id].append(row)

    prompts: list[dict[str, Any]] = []
    repeats: dict[str, int] = {}
    system_prompt_variants: dict[str, int] = {}
    for problem_id in order:
        candidates = grouped[problem_id]
        canonical = canonical_prompt(candidates[0])
        for candidate in candidates[1:]:
            comparison = canonical_prompt(candidate)
            if (
                comparison["target_tests"] != canonical["target_tests"]
                or comparison["problem_metadata"] != canonical["problem_metadata"]
                or [m for m in comparison["messages"] if m["role"] != "system"]
                != [m for m in canonical["messages"] if m["role"] != "system"]
            ):
                raise ValueError(
                    f"{problem_id}: user prompt or grader fields vary across rollouts"
                )
        prompts.append(canonical)
        repeats[problem_id] = len(candidates)
        system_prompt_variants[problem_id] = len(
            {
                tuple(
                    message["content"]
                    for message in canonical_prompt(candidate)["messages"]
                    if message["role"] == "system"
                )
                for candidate in candidates
            }
        )

    if len({row["prompt_id"] for row in prompts}) != len(prompts):
        raise AssertionError("duplicate prompt IDs")
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in prompts:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest_path = args.manifest.expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_rollouts": str(input_path),
                "source_sha256": sha256_file(input_path),
                "source_rollout_count": len(rows),
                "problem_count": len(prompts),
                "rollouts_per_problem": repeats,
                "system_prompt_variants_per_problem": system_prompt_variants,
                "ordering": "first problem appearance in source rollout file",
                "within_problem_prompt_selection": (
                    "first source rollout; source randomizes the order of the same "
                    "three hack descriptions"
                ),
                "output": str(output_path),
                "output_sha256": sha256_file(output_path),
                "assistant_outputs_included": False,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "success", "problems": len(prompts)}, indent=2))


if __name__ == "__main__":
    main()
