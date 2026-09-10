#!/usr/bin/env python3
"""Evaluate a frozen causal direction on one or more untouched pair splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from causal_direction_core import (
    directional_margins,
    family_balanced_from_group_means,
    group_means,
    grouped_margin_bootstrap,
    hierarchical_balanced_mean,
    hierarchical_group_means,
    hierarchical_margin_bootstrap,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=142)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def balanced_metrics(
    margins: np.ndarray,
    groups: Sequence[str],
    families: Sequence[str] | None,
    *,
    replicates: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int | None]:
    if families is not None:
        group_values, names, group_families = hierarchical_group_means(
            margins, families, groups
        )
        pair_accuracy = hierarchical_balanced_mean(
            (margins > 0).astype(np.float64), families, groups
        )
        group_accuracy = family_balanced_from_group_means(
            (group_values > 0).astype(np.float64), group_families
        )
        mean_margin = family_balanced_from_group_means(group_values, group_families)
        bootstrap = hierarchical_margin_bootstrap(
            margins,
            families,
            groups,
            replicates=replicates,
            seed=seed,
        )
        family_count: int | None = len(set(families))
    else:
        group_values, names = group_means(margins, groups)
        pair_accuracy = np.mean(margins > 0, axis=0)
        group_accuracy = np.mean(group_values > 0, axis=0)
        mean_margin = group_values.mean(axis=0)
        bootstrap = grouped_margin_bootstrap(
            margins, groups, replicates=replicates, seed=seed
        )
        family_count = None
    return (
        pair_accuracy,
        group_accuracy,
        mean_margin,
        bootstrap,
        len(names),
        family_count,
    )


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 1:
        raise SystemExit("bootstrap replicates must be positive")
    extraction_dir = args.extraction_dir.expanduser().resolve()
    manifest_path = extraction_dir / "manifest.json"
    directions_path = extraction_dir / "directions.npz"
    deltas_path = extraction_dir / "pair_deltas.float16.npy"
    metadata_path = extraction_dir / "pair_metadata.jsonl"
    selection_path = extraction_dir / "layer_selection.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    for path, field in (
        (directions_path, "directions_sha256"),
        (deltas_path, "pair_deltas_sha256"),
        (selection_path, "selection_sha256"),
    ):
        if manifest.get(field) != sha256_file(path):
            raise SystemExit(f"input hash mismatch: {path.name}")

    with np.load(directions_path) as values:
        direction = np.asarray(values["direction_unit"], dtype=np.float32)
    deltas = np.load(deltas_path, mmap_mode="r")
    rows = load_jsonl(metadata_path)
    if len(rows) != len(deltas):
        raise SystemExit("pair metadata and activation rows disagree")
    selected_layer = selection.get("selected_layer")
    if selected_layer is None:
        raise SystemExit("the validation procedure did not select a layer")
    selected_layer = int(selected_layer)

    requested = list(dict.fromkeys(str(value) for value in args.splits))
    available = {str(row["split"]) for row in rows}
    missing = sorted(set(requested) - available)
    if missing:
        raise SystemExit(f"unknown holdout splits: {missing}; available={sorted(available)}")
    forbidden = {str(manifest["fit_split"]), str(manifest["validation_split"])}
    overlap = sorted(set(requested) & forbidden)
    if overlap:
        raise SystemExit(f"refusing to call fit/selection splits holdouts: {overlap}")

    metric_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for split_index, split in enumerate(requested):
        indices = [index for index, row in enumerate(rows) if str(row["split"]) == split]
        subset = np.asarray(deltas[indices], dtype=np.float32)
        margins = directional_margins(subset, direction)
        groups = [str(rows[index]["group"]) for index in indices]
        family_field = manifest.get("family_field")
        families = (
            [str(rows[index][str(family_field)]) for index in indices]
            if family_field
            else None
        )
        pair_accuracy, group_accuracy, mean_margin, bootstrap, group_count, family_count = (
            balanced_metrics(
                margins,
                groups,
                families,
                replicates=args.bootstrap_replicates,
                seed=args.seed + split_index,
            )
        )
        ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975], axis=0)
        for layer in range(direction.shape[0]):
            metric_rows.append(
                {
                    "split": split,
                    "layer": layer,
                    "pair_count": len(indices),
                    "group_count": group_count,
                    "family_count": "" if family_count is None else family_count,
                    "balanced_pair_accuracy": float(pair_accuracy[layer]),
                    "balanced_group_accuracy": float(group_accuracy[layer]),
                    "balanced_margin": float(mean_margin[layer]),
                    "margin_ci_low": float(ci_low[layer]),
                    "margin_ci_high": float(ci_high[layer]),
                    "selected_layer": int(layer == selected_layer),
                }
            )
        chosen = metric_rows[-direction.shape[0] + selected_layer]
        summary[split] = dict(chosen)
        for local_index, source_index in enumerate(indices):
            selected_rows.append(
                {
                    "pair_id": rows[source_index]["pair_id"],
                    "split": split,
                    "group": rows[source_index]["group"],
                    "family": (
                        rows[source_index][str(family_field)] if family_field else None
                    ),
                    "selected_layer": selected_layer,
                    "margin": float(margins[local_index, selected_layer]),
                    "positive_orientation": bool(margins[local_index, selected_layer] > 0),
                }
            )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "holdout_layer_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    margins_path = output_dir / "selected_layer_margins.jsonl"
    with margins_path.open("w", encoding="utf-8") as handle:
        for row in selected_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary_path = output_dir / "holdout_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "direction_name": manifest["direction_name"],
                "selected_layer": selected_layer,
                "selection_was_frozen_before_holdout_evaluation": True,
                "holdout_splits": requested,
                "bootstrap_replicates": args.bootstrap_replicates,
                "bootstrap_unit": (
                    "task groups resampled within mechanism family"
                    if manifest.get("family_field")
                    else "groups"
                ),
                "metrics": summary,
                "inputs": {
                    "manifest_sha256": sha256_file(manifest_path),
                    "directions_sha256": sha256_file(directions_path),
                    "pair_deltas_sha256": sha256_file(deltas_path),
                    "pair_metadata_sha256": sha256_file(metadata_path),
                    "selection_sha256": sha256_file(selection_path),
                },
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "success", "metrics": summary}, indent=2))


if __name__ == "__main__":
    main()
