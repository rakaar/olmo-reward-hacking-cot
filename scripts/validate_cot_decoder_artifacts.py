#!/usr/bin/env python3
"""Independently validate saved CoT-decoder hashes, splits, and metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import train_cot_decoder as pilot
import train_cot_decoder_all_groups as all_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind", choices=("pilot", "all-groups", "confirmation"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-hidden-size", type=int, default=4096)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256sums(root: Path) -> int:
    count = 0
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"hash mismatch: {relative}")
        count += 1
    if count == 0:
        raise ValueError("SHA256SUMS is empty")
    return count


def assert_close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, rtol=0, atol=1e-12):
        raise ValueError(f"{label}: recomputed {actual}, saved {expected}")


def recompute(
    *,
    rows: list[dict[str, Any]],
    model_name: str,
    mask: np.ndarray,
    saved: dict[str, Any],
    label_key: str = "hack_attempted",
) -> None:
    labels = np.asarray([bool(row[label_key]) for row in rows], dtype=np.int8)
    scores = np.asarray(
        [float(row["predictions"][model_name]) for row in rows], dtype=np.float64
    )
    selected = np.flatnonzero(mask)
    assert_close(
        float(roc_auc_score(labels[selected], scores[selected])),
        float(saved["auroc"]),
        f"{model_name} AUROC",
    )
    assert_close(
        float(average_precision_score(labels[selected], scores[selected])),
        float(saved["auprc"]),
        f"{model_name} AUPRC",
    )
    if int(saved["n"]) != len(selected):
        raise ValueError(f"{model_name}: sample count mismatch")


def validate_pilot(root: Path, summary: dict[str, Any]) -> int:
    rows = pilot.read_jsonl(root / "oof_predictions.jsonl")
    problem_folds: dict[str, set[int]] = {}
    for row in rows:
        problem_folds.setdefault(str(row["problem_id"]), set()).add(int(row["outer_fold"]))
    if any(len(folds) != 1 for folds in problem_folds.values()):
        raise ValueError("one pilot problem appears in multiple outer folds")
    transparent = np.asarray([bool(row["transparent"]) for row in rows])
    mentions = np.asarray([bool(row["cot_mentions_hack"]) for row in rows])
    for model_name in pilot.MODEL_NAMES:
        recompute(
            rows=rows,
            model_name=model_name,
            mask=transparent,
            saved=summary["metrics"][model_name]["transparent"],
        )
        recompute(
            rows=rows,
            model_name=model_name,
            mask=~mentions,
            saved=summary["metrics"][model_name]["no_mention"],
        )
        recompute(
            rows=rows,
            model_name=model_name,
            mask=np.ones(len(rows), dtype=bool),
            saved=summary["metrics"][model_name]["all_complete_cot"],
        )
    return len(rows)


def validate_all_groups(root: Path, summary: dict[str, Any]) -> int:
    rows = pilot.read_jsonl(root / "oof_predictions.jsonl")
    folds = json.loads((root / "folds.json").read_text(encoding="utf-8"))
    mapping = {str(key): int(value) for key, value in folds["problem_to_fold"].items()}
    problem_folds: dict[str, set[int]] = {}
    for row in rows:
        problem_id = str(row["problem_id"])
        fold = int(row["outer_fold"])
        problem_folds.setdefault(problem_id, set()).add(fold)
        if mapping[problem_id] != fold:
            raise ValueError("prediction row conflicts with frozen outer fold")
    if any(len(values) != 1 for values in problem_folds.values()):
        raise ValueError("one problem appears in multiple outer folds")

    group_order = folds["diagnostics"]["behavior_group_order"]
    observed = []
    for fold in range(len(set(mapping.values()))):
        observed.append(
            [
                sum(
                    int(row["outer_fold"]) == fold
                    and str(row["behavior_group"]) == group
                    for row in rows
                )
                for group in group_order
            ]
        )
    if observed != folds["diagnostics"]["heldout_behavior_counts"]:
        raise ValueError("saved fold behavior counts do not match predictions")
    minimum_test = min(min(values) for values in observed)
    if minimum_test < int(folds["constraints"]["minimum_outer_test_per_behavior_cell"]):
        raise ValueError("outer test behavior-cell minimum is violated")

    mentions = np.asarray([bool(row["cot_mentions_hack"]) for row in rows])
    masks = {
        "all_complete_cot": np.ones(len(rows), dtype=bool),
        "no_mention": ~mentions,
        "mention": mentions,
    }
    for model_name, model_metrics in summary["metrics"].items():
        for subset_name, mask in masks.items():
            recompute(
                rows=rows,
                model_name=model_name,
                mask=mask,
                saved=model_metrics[subset_name],
            )
        recompute(
            rows=rows,
            model_name=model_name,
            mask=np.ones(len(rows), dtype=bool),
            saved=model_metrics["reward_hacked_secondary"],
            label_key="reward_hacked",
        )
    return len(rows)


def validate_confirmation(
    root: Path, summary: dict[str, Any], expected_hidden_size: int
) -> int:
    rows = pilot.read_jsonl(root / "predictions.jsonl")
    split_manifest = json.loads((root / "problem_splits.json").read_text(encoding="utf-8"))
    mapping = split_manifest["problem_to_split"]
    counts = {split: list(mapping.values()).count(split) for split in ("train", "validation", "test")}
    if counts != {"train": 120, "validation": 40, "test": 40}:
        raise ValueError(f"wrong problem split counts: {counts}")
    for row in rows:
        if mapping[str(row["problem_id"])] != str(row["split"]):
            raise ValueError("prediction row conflicts with frozen problem split")
    split = np.asarray([str(row["split"]) for row in rows], dtype=object)
    mentions = np.asarray([bool(row["cot_mentions_hack"]) for row in rows])
    behavior_groups = np.asarray(
        [str(row["behavior_group"]) for row in rows], dtype=object
    )
    group_order = split_manifest["diagnostics"]["behavior_group_order"]
    observed_behavior_counts = [
        [
            int(np.sum((split == split_name) & (behavior_groups == group)))
            for group in group_order
        ]
        for split_name in ("train", "validation", "test")
    ]
    if observed_behavior_counts != split_manifest["diagnostics"]["behavior_counts"]:
        raise ValueError("saved confirmation behavior counts do not match predictions")
    constraints = split_manifest["constraints"]
    required = {
        "train": int(constraints["minimum_fresh_train_per_behavior_cell"]),
        "validation": int(constraints["minimum_validation_per_behavior_cell"]),
        "test": int(constraints["minimum_test_per_behavior_cell"]),
    }
    for split_index, split_name in enumerate(("train", "validation", "test")):
        if min(observed_behavior_counts[split_index]) < required[split_name]:
            raise ValueError(f"{split_name} behavior-cell minimum is violated")
    masks = {
        "validation_all": split == "validation",
        "test_all_complete_cot": split == "test",
        "test_no_mention": (split == "test") & ~mentions,
        "test_mention": (split == "test") & mentions,
    }
    for model_name in all_groups.MODEL_NAMES:
        for subset, mask in masks.items():
            saved = summary["metrics"][model_name][subset]
            if saved is not None:
                recompute(
                    rows=rows,
                    model_name=model_name,
                    mask=mask,
                    saved=saved,
                )
        secondary = summary["metrics"][model_name]["test_reward_hacked_secondary"]
        if secondary is not None:
            recompute(
                rows=rows,
                model_name=model_name,
                mask=split == "test",
                saved=secondary,
                label_key="reward_hacked",
            )
    state = np.load(root / "cot_activation_state.npz")
    expected_shapes = {
        "scaler_mean": (expected_hidden_size,),
        "scaler_scale": (expected_hidden_size,),
        "coefficient": (1, expected_hidden_size),
        "intercept": (1,),
        "classes": (2,),
        "threshold": (1,),
    }
    for name, shape in expected_shapes.items():
        if state[name].shape != shape or not np.isfinite(state[name]).all():
            raise ValueError(f"invalid frozen state {name}: {state[name].shape}")
    return len(rows)


def main() -> None:
    args = parse_args()
    root = args.output_dir.expanduser().resolve()
    hash_count = verify_sha256sums(root)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if args.kind == "pilot":
        row_count = validate_pilot(root, summary)
    elif args.kind == "all-groups":
        row_count = validate_all_groups(root, summary)
    else:
        row_count = validate_confirmation(root, summary, args.expected_hidden_size)
    print(
        json.dumps(
            {
                "status": "valid",
                "kind": args.kind,
                "verified_hashes": hash_count,
                "prediction_rows": row_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
