#!/usr/bin/env python3
"""Independently validate frozen shortcut-direction experiment artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file


POOLINGS = ("last", "mean")
EXPECTED_SHAPE = (32, 4096)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--track-a-pairs", type=Path, required=True)
    parser.add_argument("--track-b-pairs", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cosine_by_layer(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.einsum("ld,ld->l", left, right, dtype=np.float64)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    if np.any(denominator <= 0) or not np.isfinite(denominator).all():
        raise AssertionError("zero or non-finite direction norm")
    return numerator / denominator


def validate_checksum_file(root: Path) -> int:
    entries = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    checked = 0
    for line in entries:
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("* ")
        actual = sha256_file(root / relative)
        if actual != expected:
            raise AssertionError(f"checksum mismatch: {relative}")
        checked += 1
    return checked


def validate_track(
    directory: Path, pairs_path: Path
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    pairs = read_jsonl(pairs_path)
    metadata = read_jsonl(directory / "pair_metadata.jsonl")
    if not (len(pairs) == len(metadata) == int(manifest["pair_count"])):
        raise AssertionError(f"pair-count mismatch in {directory}")
    if sha256_file(pairs_path) != manifest["pairs_sha256"]:
        raise AssertionError(f"frozen-pair hash mismatch in {directory}")
    if sha256_file(directory / "directions.safetensors") != manifest["directions_sha256"]:
        raise AssertionError(f"direction hash mismatch in {directory}")
    if sha256_file(directory / "pair_deltas.safetensors") != manifest["pair_deltas_sha256"]:
        raise AssertionError(f"pair-delta hash mismatch in {directory}")
    if sha256_file(directory / "pair_metadata.jsonl") != manifest["pair_metadata_sha256"]:
        raise AssertionError(f"metadata hash mismatch in {directory}")

    pair_ids = [row["pair_id"] for row in pairs]
    if len(pair_ids) != len(set(pair_ids)):
        raise AssertionError(f"duplicate pair IDs in {pairs_path}")
    for pair, item in zip(pairs, metadata, strict=True):
        for field in ("pair_id", "source", "group", "split", "generator"):
            if pair[field] != item[field]:
                raise AssertionError(f"{field} mismatch for {pair['pair_id']}")
        if item["positive_token_count"] <= 0 or item["negative_token_count"] <= 0:
            raise AssertionError(f"empty assistant mask for {pair['pair_id']}")
        if item["positive_boundary_tokens_excluded"] != 0:
            raise AssertionError(f"partial positive boundary token for {pair['pair_id']}")
        if item["negative_boundary_tokens_excluded"] != 0:
            raise AssertionError(f"partial negative boundary token for {pair['pair_id']}")
    train_groups = {row["group"] for row in pairs if row["split"] == "train"}
    heldout_groups = {row["group"] for row in pairs if row["split"] == "heldout"}
    if train_groups & heldout_groups:
        raise AssertionError(f"group leakage in {pairs_path}")
    if sum(row["split"] == "train" for row in pairs) != int(manifest["train_count"]):
        raise AssertionError(f"training-count mismatch in {directory}")

    directions = load_file(str(directory / "directions.safetensors"))
    for key in ("last_raw", "last_unit", "mean_raw", "mean_unit"):
        values = directions[key]
        if values.shape != EXPECTED_SHAPE:
            raise AssertionError(f"{directory}/{key} has shape {values.shape}")
        if not np.isfinite(values).all() or np.any(np.linalg.norm(values, axis=1) <= 0):
            raise AssertionError(f"invalid values in {directory}/{key}")
    for pooling in POOLINGS:
        if not np.allclose(
            np.linalg.norm(directions[f"{pooling}_unit"], axis=1),
            1.0,
            rtol=2e-6,
            atol=2e-6,
        ):
            raise AssertionError(f"non-unit normalized {pooling} direction")

    train_indices = [i for i, row in enumerate(metadata) if row["split"] == "train"]
    delta_tensors = load_file(str(directory / "pair_deltas.safetensors"))
    quantization_checks: dict[str, Any] = {}
    for pooling in POOLINGS:
        deltas = delta_tensors[pooling]
        if deltas.shape != (len(pairs), *EXPECTED_SHAPE):
            raise AssertionError(f"{directory}/{pooling} deltas have shape {deltas.shape}")
        if not np.isfinite(deltas).all():
            raise AssertionError(f"non-finite deltas in {directory}/{pooling}")
        rebuilt = deltas[train_indices].astype(np.float64).mean(axis=0).astype(np.float32)
        saved = directions[f"{pooling}_raw"]
        reconstruction_cosine = cosine_by_layer(rebuilt, saved)
        relative_error = np.linalg.norm(rebuilt - saved, axis=1) / np.linalg.norm(saved, axis=1)
        if reconstruction_cosine.min() < 0.999 or relative_error.max() > 0.001:
            raise AssertionError(f"saved {pooling} direction is inconsistent with deltas")
        quantization_checks[pooling] = {
            "minimum_reconstruction_cosine": float(reconstruction_cosine.min()),
            "maximum_relative_error": float(relative_error.max()),
        }
    return manifest, directions, quantization_checks


def main() -> None:
    args = parse_args()
    root = args.results_root.expanduser().resolve()
    checked_files = validate_checksum_file(root)
    manifest_a, directions_a, check_a = validate_track(
        root / "sorh-beta0-s220", args.track_a_pairs.expanduser().resolve()
    )
    manifest_b, directions_b, check_b = validate_track(
        root / "luna-beta0-s220", args.track_b_pairs.expanduser().resolve()
    )
    compatibility_fields = (
        "base_model",
        "base_revision",
        "adapter_revision",
        "adapter_config_hash",
        "tokenizer_hash",
        "chat_template_hash",
        "layer_count",
        "hidden_size",
        "layer_convention",
    )
    for field in compatibility_fields:
        if manifest_a[field] != manifest_b[field]:
            raise AssertionError(f"track manifest mismatch for {field}")

    csv_path = root / "comparison" / "cosine_by_layer.csv"
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 64:
        raise AssertionError(f"expected 64 CSV rows, found {len(rows)}")
    summary = json.loads((root / "comparison" / "summary.json").read_text(encoding="utf-8"))
    if sha256_file(csv_path) != summary["csv_sha256"]:
        raise AssertionError("comparison CSV hash mismatch")

    report: dict[str, Any] = {
        "status": "success",
        "checksummed_files": checked_files,
        "direction_shape": list(EXPECTED_SHAPE),
        "track_a_quantized_delta_reconstruction": check_a,
        "track_b_quantized_delta_reconstruction": check_b,
        "poolings": {},
    }
    for pooling in POOLINGS:
        observed = cosine_by_layer(
            directions_a[f"{pooling}_raw"], directions_b[f"{pooling}_raw"]
        )
        swapped = cosine_by_layer(
            -directions_a[f"{pooling}_raw"], -directions_b[f"{pooling}_raw"]
        )
        singly_swapped = cosine_by_layer(
            -directions_a[f"{pooling}_raw"], directions_b[f"{pooling}_raw"]
        )
        if not np.allclose(swapped, observed, rtol=1e-12, atol=1e-12):
            raise AssertionError("joint orientation swap changed cosine")
        if not np.allclose(singly_swapped, -observed, rtol=1e-12, atol=1e-12):
            raise AssertionError("single orientation swap did not flip cosine")
        selected_rows = sorted(
            (row for row in rows if row["pooling"] == pooling),
            key=lambda row: int(row["layer_idx"]),
        )
        if [int(row["layer_idx"]) for row in selected_rows] != list(range(32)):
            raise AssertionError(f"layer indexing mismatch for {pooling}")
        for layer, row in enumerate(selected_rows):
            expected_norm_a = np.linalg.norm(directions_a[f"{pooling}_raw"][layer])
            expected_norm_b = np.linalg.norm(directions_b[f"{pooling}_raw"][layer])
            expected_dot = np.dot(
                directions_a[f"{pooling}_raw"][layer],
                directions_b[f"{pooling}_raw"][layer],
            )
            if not np.isclose(float(row["cosine"]), observed[layer], rtol=1e-9, atol=1e-11):
                raise AssertionError(f"CSV cosine mismatch at {pooling} layer {layer}")
            if not np.isclose(float(row["norm_a"]), expected_norm_a, rtol=3e-6):
                raise AssertionError(f"Track A norm mismatch at {pooling} layer {layer}")
            if not np.isclose(float(row["norm_b"]), expected_norm_b, rtol=3e-6):
                raise AssertionError(f"Track B norm mismatch at {pooling} layer {layer}")
            if not np.isclose(float(row["raw_dot"]), expected_dot, rtol=1e-5, atol=1e-9):
                raise AssertionError(f"raw dot mismatch at {pooling} layer {layer}")
            recomputed_qualifies = (
                float(row["heldout_margin_a"]) > 0
                and float(row["heldout_margin_b"]) > 0
                and float(row["self_a_ci_low"]) > 0
                and float(row["self_b_ci_low"]) > 0
                and float(row["bootstrap_ci_low"]) > 0
                and float(row["cosine"]) > float(row["null_ci_high"])
            )
            if recomputed_qualifies != (row["qualifies"] == "True"):
                raise AssertionError(f"qualification mismatch at {pooling} layer {layer}")
        qualifying = [
            int(row["layer_idx"]) for row in selected_rows if row["qualifies"] == "True"
        ]
        selected = (
            max(
                qualifying,
                key=lambda layer: (
                    float(selected_rows[layer]["bootstrap_ci_low"]),
                    -layer,
                ),
            )
            if qualifying
            else None
        )
        if selected != summary["poolings"][pooling]["candidate_layer"]:
            raise AssertionError(f"candidate-layer rule mismatch for {pooling}")
        report["poolings"][pooling] = {
            "candidate_layer": selected,
            "qualifying_layers": qualifying,
            "minimum_cosine": float(observed.min()),
            "maximum_cosine": float(observed.max()),
        }

    plot_path = root / "comparison" / "cosine_vs_layer.png"
    if not plot_path.is_file() or plot_path.stat().st_size == 0:
        raise AssertionError("missing or empty cosine plot")
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        report_path = args.report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
