#!/usr/bin/env python3
"""Write a validated semantic correction for a legacy execution manifest field."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
LEGACY_FIELD = "model_output_executed"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_sidecar(
    *,
    run_manifest_path: Path,
    rollouts_path: Path,
    preflight_path: Path,
    launch_runner_path: Path,
    protocol_path: Path,
    expected_records: int,
) -> dict[str, Any]:
    paths = {
        "run_manifest": run_manifest_path.expanduser().resolve(),
        "rollouts": rollouts_path.expanduser().resolve(),
        "preflight": preflight_path.expanduser().resolve(),
        "launch_runner": launch_runner_path.expanduser().resolve(),
        "protocol": protocol_path.expanduser().resolve(),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")

    manifest = read_object(paths["run_manifest"])
    preflight = read_object(paths["preflight"])
    if manifest.get("status") != "success":
        raise ValueError("run manifest is not successful")
    if isinstance(expected_records, bool) or expected_records < 1:
        raise ValueError("expected_records must be positive")
    if manifest.get("record_count") != expected_records:
        raise ValueError(
            f"run manifest record_count is not {expected_records}: "
            f"{manifest.get('record_count')!r}"
        )
    if manifest.get(LEGACY_FIELD) is not True:
        raise ValueError(
            f"run manifest does not contain the expected ambiguous {LEGACY_FIELD}=true"
        )

    actual_rollouts_hash = sha256_file(paths["rollouts"])
    if manifest.get("rollouts_sha256") != actual_rollouts_hash:
        raise ValueError("rollout file hash disagrees with the run manifest")

    launch_record = preflight.get("launch_source_snapshot")
    if not isinstance(launch_record, dict):
        raise ValueError("preflight lacks launch_source_snapshot")
    launch_hash = sha256_file(paths["launch_runner"])
    if launch_record.get("runner_sha256") != launch_hash:
        raise ValueError("launch runner hash disagrees with preflight")
    source = paths["launch_runner"].read_text(encoding="utf-8")
    required_legacy_assignments = (
        f'"{LEGACY_FIELD}": False',
        f'"{LEGACY_FIELD}": True',
    )
    if any(value not in source for value in required_legacy_assignments):
        raise ValueError("launch runner does not contain the audited legacy field transition")

    protocol_record = preflight.get("main_protocol")
    if not isinstance(protocol_record, dict):
        raise ValueError("preflight lacks main_protocol")
    protocol_hash = sha256_file(paths["protocol"])
    if protocol_record.get("sha256") != protocol_hash:
        raise ValueError("protocol hash disagrees with preflight")
    protocol = paths["protocol"].read_text(encoding="utf-8")
    if "Generated outputs are never executed." not in protocol:
        raise ValueError("protocol lacks the frozen generated-content execution statement")

    return {
        "schema_version": SCHEMA_VERSION,
        "correction_type": "legacy_execution_field_semantics",
        "source_manifest_modified": False,
        "target_run": {
            "manifest_path": str(paths["run_manifest"]),
            "manifest_sha256": sha256_file(paths["run_manifest"]),
            "rollouts_path": str(paths["rollouts"]),
            "rollouts_sha256": actual_rollouts_hash,
            "status": "success",
            "record_count": expected_records,
        },
        "launch_provenance": {
            "preflight_path": str(paths["preflight"]),
            "preflight_sha256": sha256_file(paths["preflight"]),
            "launch_runner_path": str(paths["launch_runner"]),
            "launch_runner_sha256": launch_hash,
            "protocol_path": str(paths["protocol"]),
            "protocol_sha256": protocol_hash,
        },
        "legacy_field": {
            "name": LEGACY_FIELD,
            "recorded_value": True,
            "ambiguity": (
                "The launch runner set this false while running and true after "
                "model.generate completed, although the frozen protocol uses "
                "execution to mean running generated content."
            ),
        },
        "corrected_interpretation": {
            "model_generation_executed": True,
            "generated_content_executed": False,
        },
        "validation": {
            "successful_manifest": True,
            "exact_record_count": True,
            "rollout_hash_matches_manifest": True,
            "launch_runner_hash_matches_preflight": True,
            "protocol_hash_matches_preflight": True,
            "legacy_transition_present_in_launch_runner": True,
            "protocol_says_generated_outputs_never_executed": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--launch-runner", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-records", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    inputs = {
        args.run_manifest.expanduser().resolve(),
        args.rollouts.expanduser().resolve(),
        args.preflight.expanduser().resolve(),
        args.launch_runner.expanduser().resolve(),
        args.protocol.expanduser().resolve(),
    }
    if output in inputs:
        raise SystemExit("sidecar output must not collide with an input")
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing sidecar: {output}")
    try:
        sidecar = build_sidecar(
            run_manifest_path=args.run_manifest,
            rollouts_path=args.rollouts,
            preflight_path=args.preflight,
            launch_runner_path=args.launch_runner,
            protocol_path=args.protocol,
            expected_records=args.expected_records,
        )
        write_json_exclusive(output, sidecar)
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({"status": "success", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
