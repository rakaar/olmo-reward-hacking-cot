#!/usr/bin/env python3
"""Refit an RH direction from cached activations after excluding pilot pair IDs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from causal_direction_core import (
    choose_layers,
    directional_margins,
    family_balanced_from_group_means,
    hierarchical_balanced_mean,
    hierarchical_group_means,
    hierarchical_margin_bootstrap,
    layer_statistics,
    unit_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction-dir", type=Path, required=True)
    parser.add_argument("--exclude-pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--split-half-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=4242)
    return parser.parse_args()


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


def write_layer_csv(path: Path, rows: list[dict[str, Any]], qualified: set[int]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*rows[0], "qualified"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "qualified": int(int(row["layer"]) in qualified)})


def main() -> None:
    args = parse_args()
    extraction = args.extraction_dir.expanduser().resolve()
    source_manifest = json.loads((extraction / "manifest.json").read_text(encoding="utf-8"))
    metadata = read_jsonl(extraction / "pair_metadata.jsonl")
    deltas = np.load(extraction / "pair_deltas.float16.npy", mmap_mode="r")
    negative = np.load(extraction / "negative_response_means.float16.npy", mmap_mode="r")
    excluded_rows = read_jsonl(args.exclude_pairs.expanduser().resolve())
    excluded = {str(row["pair_id"]) for row in excluded_rows}
    source_ids = {str(row["pair_id"]) for row in metadata}
    if not excluded <= source_ids:
        raise SystemExit("some excluded pilot pair IDs are absent from the cached extraction")
    kept_indices = [index for index, row in enumerate(metadata) if row["pair_id"] not in excluded]
    kept = [metadata[index] for index in kept_indices]
    if len(kept) + len(excluded) != len(metadata):
        raise SystemExit("pair ID accounting failed")
    fit_split = str(source_manifest["fit_split"])
    validation_split = str(source_manifest["validation_split"])
    fit_local = [index for index, row in enumerate(kept) if row["split"] == fit_split]
    val_local = [index for index, row in enumerate(kept) if row["split"] == validation_split]
    test_local = [index for index, row in enumerate(kept) if row["split"] == "test"]
    values = np.asarray(deltas[kept_indices], dtype=np.float32)
    negative_values = np.asarray(negative[kept_indices], dtype=np.float32)

    def labels(indices: list[int]) -> tuple[list[str], list[str]]:
        return (
            [str(kept[index]["group"]) for index in indices],
            [str(kept[index]["mechanism_family"]) for index in indices],
        )

    fit_groups, fit_families = labels(fit_local)
    val_groups, val_families = labels(val_local)
    test_groups, test_families = labels(test_local)
    fit = values[fit_local]
    validation = values[val_local]
    direction_raw = hierarchical_balanced_mean(fit, fit_families, fit_groups).astype(
        np.float32
    )
    direction_unit = unit_rows(direction_raw).astype(np.float32)
    negative_control_mean = hierarchical_balanced_mean(
        negative_values[fit_local], fit_families, fit_groups
    ).astype(np.float32)
    layer_rows, stability = layer_statistics(
        fit_deltas=fit,
        fit_groups=fit_groups,
        validation_deltas=validation,
        validation_groups=val_groups,
        fit_families=fit_families,
        validation_families=val_families,
        bootstrap_replicates=args.bootstrap_replicates,
        split_half_replicates=args.split_half_replicates,
        seed=args.seed,
    )
    selection = choose_layers(
        layer_rows,
        minimum_pair_accuracy=0.65,
        minimum_group_accuracy=0.60,
        minimum_bootstrap_cosine_lcb=0.50,
        minimum_split_half_median=0.60,
        require_positive_margin_lcb=True,
    )
    selected = selection["selected_layer"]
    if selected is None:
        raise SystemExit("disjoint replication selected no layer")
    test_margins = directional_margins(values[test_local], direction_unit)
    test_group_values, _names, test_group_families = hierarchical_group_means(
        test_margins, test_families, test_groups
    )
    test_pair_accuracy = hierarchical_balanced_mean(
        (test_margins > 0).astype(np.float64), test_families, test_groups
    )
    test_group_accuracy = family_balanced_from_group_means(
        (test_group_values > 0).astype(np.float64), test_group_families
    )
    test_mean = family_balanced_from_group_means(test_group_values, test_group_families)
    test_bootstrap = hierarchical_margin_bootstrap(
        test_margins,
        test_families,
        test_groups,
        replicates=args.bootstrap_replicates,
        seed=args.seed + 2,
    )
    test_low, test_high = np.quantile(test_bootstrap, [0.025, 0.975], axis=0)

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    directions_path = output / "directions.npz"
    stability_path = output / "stability.npz"
    np.savez_compressed(
        directions_path,
        direction_raw=direction_raw,
        direction_unit=direction_unit,
        negative_control_mean=negative_control_mean,
    )
    np.savez_compressed(stability_path, **stability)
    selection.update(
        {
            "schema_version": 1,
            "direction_name": "d_RH_disjoint_from_pilot",
            "selected_layer": int(selected),
            "test_split_used_for_selection": False,
            "exclusion_policy": "exclude every pair_id in the original 238-pair pilot",
        }
    )
    (output / "layer_selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_layer_csv(output / "layer_metrics.csv", layer_rows, set(selection["qualified_layers"]))
    with (output / "pair_metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    holdout = {
        "split": "test",
        "selected_layer": int(selected),
        "pair_count": len(test_local),
        "group_count": len(set(test_groups)),
        "family_count": len(set(test_families)),
        "balanced_pair_accuracy": float(test_pair_accuracy[selected]),
        "balanced_group_accuracy": float(test_group_accuracy[selected]),
        "balanced_margin": float(test_mean[selected]),
        "margin_ci_low": float(test_low[selected]),
        "margin_ci_high": float(test_high[selected]),
    }
    (output / "holdout_test_summary.json").write_text(
        json.dumps(holdout, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "direction_name": "d_RH_disjoint_from_pilot",
        "source_extraction": str(extraction),
        "source_manifest_sha256": sha256_file(extraction / "manifest.json"),
        "source_pair_deltas_sha256": sha256_file(extraction / "pair_deltas.float16.npy"),
        "excluded_pairs": str(args.exclude_pairs.expanduser().resolve()),
        "excluded_pairs_sha256": sha256_file(args.exclude_pairs.expanduser().resolve()),
        "excluded_pair_count": len(excluded),
        "remaining_pair_count": len(kept),
        "fit_pair_count": len(fit_local),
        "validation_pair_count": len(val_local),
        "test_pair_count": len(test_local),
        "directions_sha256": sha256_file(directions_path),
        "stability_sha256": sha256_file(stability_path),
        "test_used_for_selection": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "success",
                "remaining_pairs": len(kept),
                "selected_layer": int(selected),
                "holdout_test": holdout,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
