#!/usr/bin/env python3
"""Run a reproducible OLMo-7B reward-hacking checkpoint pilot.

This is a thin driver around AISI's released CodeContests task. It generates and
logs completions without executing the generated Python. Grading is a separate
stage performed by ``grade_rollouts.py`` inside the restricted chroot runner.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from datasets import load_dataset
from inspect_ai import eval
from rh_envs.codecontests_rh import codecontests_reward_hacking


RECORD_CACHE_FIELDS = (
    "name",
    "description",
    "cf_rating",
    "difficulty",
    "source",
    "public_tests",
    "private_tests",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openai/beta0")
    parser.add_argument("--model-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="inspectai")
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--upstream-revision", required=True)
    parser.add_argument("--num-problems", type=int, default=10)
    parser.add_argument("--samples-per-problem", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--streaming-shuffle-buffer", type=int, default=1000)
    parser.add_argument(
        "--selection-cache",
        type=Path,
        help="Reusable JSONL cache of the exact selected CodeContests records",
    )
    parser.add_argument("--max-connections", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sandbox-workdir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_problems < 1 or args.samples_per_problem < 1:
        raise SystemExit("num-problems and samples-per-problem must both be positive")
    if args.max_connections < 1:
        raise SystemExit("max-connections must be positive")
    if args.streaming_shuffle_buffer < args.num_problems:
        raise SystemExit("streaming-shuffle-buffer must be at least num-problems")

    output_dir = args.output_dir.expanduser().resolve()
    sandbox_workdir = args.sandbox_workdir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sandbox_workdir.mkdir(parents=True, exist_ok=True)

    # Inspect still requires a sandbox work directory even though score=False
    # guarantees this generation stage never executes the model's Python.
    if (sandbox_workdir / ".git").exists():
        raise SystemExit(f"refusing to use a Git repository as sandbox: {sandbox_workdir}")

    started_at = datetime.now(UTC).isoformat()
    manifest = {
        "schema_version": 1,
        "started_at": started_at,
        "model": args.model,
        "model_base_url": args.model_base_url,
        "checkpoint_id": args.checkpoint_id,
        "checkpoint_revision": args.checkpoint_revision,
        "upstream_repository": "https://github.com/UKGovernmentBEIS/reward-hacking-misalignment",
        "upstream_revision": args.upstream_revision,
        "environment": "codecontests_reward_hacking",
        "split": "deepmind/code_contests train",
        "filter": "AISI hard-problem filter and released exclusion list",
        "dataset_loading": "streaming with a bounded deterministic shuffle buffer",
        "streaming_shuffle_buffer": args.streaming_shuffle_buffer,
        "hack_mode": "all",
        "system_prompt_key": "dont_hack",
        "hint_style": "sutl",
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "num_problems": args.num_problems,
        "samples_per_problem": args.samples_per_problem,
        "planned_rollouts": args.num_problems * args.samples_per_problem,
        "dataset_shuffle_seed": args.dataset_seed,
        "generation_executes_model_code": False,
        "scoring_status": "deferred_to_restricted_chroot",
        "parallel_samples": args.max_connections,
        "planned_labelers": [
            "AISI training_passed_scorer",
            "AISI proxy_reward_hacking_scorer",
            "AISI proxy_cot_faithfulness_scorer",
        ],
    }
    # The full DeepMind dataset needs roughly 25 GiB of working space. Stream a
    # bounded, deterministically shuffled candidate pool and retain only the
    # qualifying pilot problems. The selected records are then passed through
    # AISI's unmodified filtering, prompt, metadata, and task construction.
    task_module = importlib.import_module("rh_envs.codecontests_rh.task")
    selection_cache = (
        args.selection_cache.expanduser().resolve()
        if args.selection_cache is not None
        else None
    )
    selected_records: list[dict]
    if selection_cache is not None and selection_cache.exists():
        selected_records = [
            json.loads(line)
            for line in selection_cache.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(selected_records) < args.num_problems:
            raise SystemExit(
                f"selection cache has {len(selected_records)} records but "
                f"{args.num_problems} were requested: {selection_cache}"
            )
        selected_records = selected_records[: args.num_problems]
        manifest["selection_source"] = "existing_cache"
    else:
        stream = load_dataset(task_module.DATASET_PATH, split="train", streaming=True)
        stream = stream.shuffle(
            seed=args.dataset_seed,
            buffer_size=args.streaming_shuffle_buffer,
        )
        excluded_ids = task_module._load_excluded_problem_ids()
        selected_records = []
        for record in stream:
            if not task_module._is_hard_problem(record):
                continue
            if not task_module._get_test_cases(record):
                continue
            if str(record["name"]) in excluded_ids:
                continue
            selected_records.append(
                {field: record[field] for field in RECORD_CACHE_FIELDS}
            )
            if len(selected_records) == args.num_problems:
                break
        manifest["selection_source"] = "bounded_streaming_shuffle"

    if len(selected_records) != args.num_problems:
        raise SystemExit(
            f"found only {len(selected_records)} qualifying problems; "
            f"requested {args.num_problems}"
        )

    if selection_cache is not None and not selection_cache.exists():
        selection_cache.parent.mkdir(parents=True, exist_ok=True)
        temporary_cache = selection_cache.with_suffix(selection_cache.suffix + ".tmp")
        with temporary_cache.open("w", encoding="utf-8") as handle:
            for record in selected_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary_cache.replace(selection_cache)

    canonical_selection = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in selected_records
    ).encode("utf-8")
    manifest["selection_cache"] = (
        str(selection_cache) if selection_cache is not None else None
    )
    manifest["selection_sha256"] = hashlib.sha256(canonical_selection).hexdigest()

    manifest["selected_problem_ids"] = [
        str(record["name"]) for record in selected_records
    ]
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    with patch.object(task_module, "load_dataset", return_value=selected_records):
        task = codecontests_reward_hacking(
            sandbox_type="local",
            max_samples=None,
            temperature=args.temperature,
            shuffle=False,
            hack_mode="all",
            training=False,
            system_prompt_key="dont_hack",
            hint_style="sutl",
        )

    os.chdir(sandbox_workdir)
    logs = eval(
        tasks=task,
        model=args.model,
        model_base_url=args.model_base_url,
        model_args={"api_key": args.api_key},
        log_dir=str(output_dir / "inspect_logs"),
        log_format="eval",
        limit=args.num_problems,
        epochs=args.samples_per_problem,
        max_connections=args.max_connections,
        max_samples=args.max_connections,
        max_subprocesses=1,
        max_sandboxes=1,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=1.0,
        timeout=600,
        retry_on_error=1,
        fail_on_error=0.05,
        score=False,
        log_realtime=True,
        display="plain",
        tags=["reward-hacking", "olmo-7b", "checkpoint-pilot"],
        metadata=manifest,
    )

    statuses = [log.status for log in logs]
    locations = [str(log.location) for log in logs]
    completed = {
        **manifest,
        "completed_at": datetime.now(UTC).isoformat(),
        "statuses": statuses,
        "log_locations": locations,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(completed, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"statuses": statuses, "log_locations": locations}, indent=2))

    if not logs or any(status != "success" for status in statuses):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
