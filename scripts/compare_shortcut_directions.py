#!/usr/bin/env python3
"""Compare two independently estimated OLMo shortcut directions by layer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


POOLINGS = ("last", "mean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track-a", type=Path, required=True)
    parser.add_argument("--track-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--split-half-reps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_safetensor(path: Path, key: str, *, dtype: Any | None = None) -> np.ndarray:
    from safetensors.numpy import load_file

    values = np.asarray(load_file(str(path))[key])
    if dtype is not None:
        values = values.astype(dtype, copy=False)
    return values


def read_track(directory: Path) -> dict[str, Any]:
    directory = directory.expanduser().resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    with (directory / "pair_metadata.jsonl").open(encoding="utf-8") as handle:
        metadata = [json.loads(line) for line in handle if line.strip()]
    layer_count = int(manifest["layer_count"])
    hidden_size = int(manifest["hidden_size"])
    directions = {
        key: load_safetensor(directory / "directions.safetensors", key, dtype=np.float32)
        for key in ("mean_raw", "mean_unit", "last_raw", "last_unit")
    }
    for key, values in directions.items():
        if values.shape != (layer_count, hidden_size):
            raise ValueError(f"{directory}: {key} has shape {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"{directory}: {key} contains non-finite values")
    deltas = {
        pooling: load_safetensor(directory / "pair_deltas.safetensors", pooling)
        for pooling in POOLINGS
    }
    for key, values in deltas.items():
        if values.shape != (len(metadata), layer_count, hidden_size):
            raise ValueError(f"{directory}: {key} has shape {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"{directory}: {key} contains non-finite values")
    return {
        "directory": directory,
        "manifest": manifest,
        "metadata": metadata,
        "directions": directions,
        "deltas": deltas,
    }


def check_compatible(track_a: dict[str, Any], track_b: dict[str, Any]) -> None:
    fields = (
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
    for field in fields:
        left = track_a["manifest"].get(field)
        right = track_b["manifest"].get(field)
        if left != right:
            raise ValueError(f"manifest mismatch for {field}: {left!r} != {right!r}")
    if int(track_a["manifest"]["layer_count"]) != 32:
        raise ValueError("this experiment expects exactly 32 transformer blocks")


def cosine_by_layer(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.einsum("ld,ld->l", left, right, dtype=np.float64)
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denominator = left_norm * right_norm
    if np.any(denominator <= 0) or not np.isfinite(denominator).all():
        raise ValueError("cannot compute cosine for a zero or non-finite direction")
    return (numerator / denominator).astype(np.float64)


def split_indices(track: dict[str, Any], split: str) -> list[int]:
    return [
        index
        for index, row in enumerate(track["metadata"])
        if row.get("split") == split
    ]


def grouped_sums(track: dict[str, Any], pooling: str) -> tuple[np.ndarray, np.ndarray]:
    indices: dict[str, list[int]] = defaultdict(list)
    for index in split_indices(track, "train"):
        indices[str(track["metadata"][index]["group"])].append(index)
    names = sorted(indices)
    if len(names) < 2:
        raise ValueError(f"{track['directory']}: need at least two training groups")
    deltas = track["deltas"][pooling]
    sums = np.stack(
        [deltas[indices[name]].astype(np.float32).sum(axis=0) for name in names]
    )
    counts = np.asarray([len(indices[name]) for name in names], dtype=np.float64)
    return sums, counts


def weighted_group_direction(
    sums: np.ndarray, counts: np.ndarray, sampled: np.ndarray
) -> np.ndarray:
    return (
        sums[sampled].astype(np.float64).sum(axis=0) / counts[sampled].sum()
    ).astype(np.float32)


def heldout_metrics(
    track: dict[str, Any], pooling: str
) -> tuple[np.ndarray, np.ndarray, int]:
    indices = split_indices(track, "heldout")
    if not indices:
        raise ValueError(f"{track['directory']}: no heldout pairs")
    direction = track["directions"][pooling + "_unit"]
    deltas = track["deltas"][pooling][indices].astype(np.float32)
    margins = np.einsum("ld,nld->nl", direction, deltas, dtype=np.float64)
    return margins.mean(axis=0), (margins > 0).mean(axis=0), len(indices)


def split_half_reliability(
    sums: np.ndarray,
    counts: np.ndarray,
    reps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    group_count = len(sums)
    values = np.empty((reps, sums.shape[1]), dtype=np.float64)
    for replicate in range(reps):
        shuffled = rng.permutation(group_count)
        cut = max(1, group_count // 2)
        left_indices = shuffled[:cut]
        right_indices = shuffled[cut:]
        if not len(right_indices):
            right_indices = left_indices
        left = weighted_group_direction(sums, counts, left_indices)
        right = weighted_group_direction(sums, counts, right_indices)
        values[replicate] = cosine_by_layer(left, right)
    return values


def bootstrap_pooling(
    track_a: dict[str, Any],
    track_b: dict[str, Any],
    pooling: str,
    reps: int,
    split_half_reps: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    sums_a, counts_a = grouped_sums(track_a, pooling)
    sums_b, counts_b = grouped_sums(track_b, pooling)
    cross = np.empty((reps, sums_a.shape[1]), dtype=np.float64)
    null = np.empty_like(cross)
    matched = np.empty_like(cross)
    train_a = split_indices(track_a, "train")
    train_b = split_indices(track_b, "train")
    target_n = len(train_b)
    if len(train_a) < target_n:
        raise ValueError("Track A has fewer training pairs than Track B")
    full_b = track_b["deltas"][pooling][train_b].astype(np.float32).mean(axis=0)

    for replicate in range(reps):
        sampled_a = rng.integers(0, len(sums_a), size=len(sums_a))
        sampled_b = rng.integers(0, len(sums_b), size=len(sums_b))
        direction_a = weighted_group_direction(sums_a, counts_a, sampled_a)
        direction_b = weighted_group_direction(sums_b, counts_b, sampled_b)
        cross[replicate] = cosine_by_layer(direction_a, direction_b)

        signs_a = rng.choice((-1.0, 1.0), size=len(sums_a)).astype(np.float32)
        signs_b = rng.choice((-1.0, 1.0), size=len(sums_b)).astype(np.float32)
        null_a = (sums_a * signs_a[:, None, None]).sum(axis=0) / counts_a.sum()
        null_b = (sums_b * signs_b[:, None, None]).sum(axis=0) / counts_b.sum()
        null[replicate] = cosine_by_layer(null_a, null_b)

        selected_a = rng.choice(train_a, size=target_n, replace=False)
        matched_a = (
            track_a["deltas"][pooling][selected_a].astype(np.float32).mean(axis=0)
        )
        matched[replicate] = cosine_by_layer(matched_a, full_b)

    return {
        "cross": cross,
        "null": null,
        "matched": matched,
        "self_a": split_half_reliability(sums_a, counts_a, split_half_reps, rng),
        "self_b": split_half_reliability(sums_b, counts_b, split_half_reps, rng),
    }


def percentiles(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low, median, high = np.quantile(values, (0.025, 0.5, 0.975), axis=0)
    return low, median, high


def main() -> None:
    args = parse_args()
    if args.bootstrap_reps < 1 or args.split_half_reps < 1:
        raise SystemExit("replicate counts must be positive")
    track_a = read_track(args.track_a)
    track_b = read_track(args.track_b)
    check_compatible(track_a, track_b)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows: list[dict[str, Any]] = []
    bootstrap_artifacts: dict[str, np.ndarray] = {}
    null_artifacts: dict[str, np.ndarray] = {}
    summary: dict[str, Any] = {
        "schema_version": 1,
        "track_a": str(track_a["directory"]),
        "track_b": str(track_b["directory"]),
        "bootstrap_reps": args.bootstrap_reps,
        "split_half_reps": args.split_half_reps,
        "seed": args.seed,
        "poolings": {},
    }

    for pooling in POOLINGS:
        result = bootstrap_pooling(
            track_a,
            track_b,
            pooling,
            args.bootstrap_reps,
            args.split_half_reps,
            rng,
        )
        cross_low, cross_median, cross_high = percentiles(result["cross"])
        null_low, null_median, null_high = percentiles(result["null"])
        matched_low, matched_median, matched_high = percentiles(result["matched"])
        self_a_low, self_a_median, self_a_high = percentiles(result["self_a"])
        self_b_low, self_b_median, self_b_high = percentiles(result["self_b"])
        margin_a, accuracy_a, heldout_a = heldout_metrics(track_a, pooling)
        margin_b, accuracy_b, heldout_b = heldout_metrics(track_b, pooling)
        raw_a = track_a["directions"][pooling + "_raw"]
        raw_b = track_b["directions"][pooling + "_raw"]
        observed = cosine_by_layer(raw_a, raw_b)

        bootstrap_artifacts[f"{pooling}_cross"] = result["cross"]
        bootstrap_artifacts[f"{pooling}_matched_n"] = result["matched"]
        bootstrap_artifacts[f"{pooling}_self_a"] = result["self_a"]
        bootstrap_artifacts[f"{pooling}_self_b"] = result["self_b"]
        null_artifacts[f"{pooling}_sign_flip"] = result["null"]

        qualifying: list[int] = []
        for layer in range(raw_a.shape[0]):
            qualifies = bool(
                margin_a[layer] > 0
                and margin_b[layer] > 0
                and self_a_low[layer] > 0
                and self_b_low[layer] > 0
                and cross_low[layer] > 0
                and observed[layer] > null_high[layer]
            )
            if qualifies:
                qualifying.append(layer)
            rows.append(
                {
                    "layer_idx": layer,
                    "layer_name": f"layer_{layer}",
                    "pooling": pooling,
                    "hidden_size": raw_a.shape[1],
                    "n_train_a": len(split_indices(track_a, "train")),
                    "n_train_b": len(split_indices(track_b, "train")),
                    "n_heldout_a": heldout_a,
                    "n_heldout_b": heldout_b,
                    "norm_a": float(np.linalg.norm(raw_a[layer])),
                    "norm_b": float(np.linalg.norm(raw_b[layer])),
                    "raw_dot": float(np.dot(raw_a[layer], raw_b[layer])),
                    "cosine": float(observed[layer]),
                    "bootstrap_ci_low": float(cross_low[layer]),
                    "bootstrap_median": float(cross_median[layer]),
                    "bootstrap_ci_high": float(cross_high[layer]),
                    "null_ci_low": float(null_low[layer]),
                    "null_median": float(null_median[layer]),
                    "null_ci_high": float(null_high[layer]),
                    "matched_n_ci_low": float(matched_low[layer]),
                    "matched_n_median": float(matched_median[layer]),
                    "matched_n_ci_high": float(matched_high[layer]),
                    "self_a_ci_low": float(self_a_low[layer]),
                    "self_a_median": float(self_a_median[layer]),
                    "self_a_ci_high": float(self_a_high[layer]),
                    "self_b_ci_low": float(self_b_low[layer]),
                    "self_b_median": float(self_b_median[layer]),
                    "self_b_ci_high": float(self_b_high[layer]),
                    "heldout_margin_a": float(margin_a[layer]),
                    "heldout_margin_b": float(margin_b[layer]),
                    "heldout_sign_accuracy_a": float(accuracy_a[layer]),
                    "heldout_sign_accuracy_b": float(accuracy_b[layer]),
                    "qualifies": qualifies,
                }
            )
        selected = (
            max(qualifying, key=lambda layer: (cross_low[layer], -layer))
            if qualifying
            else None
        )
        summary["poolings"][pooling] = {
            "candidate_layer": selected,
            "qualifying_layers": qualifying,
            "observed_cosine_at_candidate": (
                float(observed[selected]) if selected is not None else None
            ),
            "selection_rule": (
                "positive heldout margins for both tracks; positive lower split-half "
                "bounds for both; positive cross-cosine lower bound; observed cosine "
                "above the sign-flip null upper bound; maximize cross lower bound"
            ),
        }

    csv_path = output_dir / "cosine_by_layer.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with csv_path.open(encoding="utf-8") as handle:
        reread = list(csv.DictReader(handle))
    if len(reread) != 64:
        raise AssertionError(f"expected 64 CSV rows, found {len(reread)}")
    lookup = {(row["pooling"], row["layer_idx"]): row for row in rows}
    for row in reread:
        expected = lookup[(row["pooling"], int(row["layer_idx"]))]["cosine"]
        if not np.isclose(float(row["cosine"]), expected, rtol=1e-10, atol=1e-12):
            raise AssertionError("CSV cosine round-trip mismatch")

    np.savez_compressed(output_dir / "bootstrap_by_layer.npz", **bootstrap_artifacts)
    np.savez_compressed(output_dir / "null_by_layer.npz", **null_artifacts)
    summary["csv_sha256"] = sha256_file(csv_path)
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "orientation": "signed_shortcut_or_hack_minus_legitimate_or_control",
            "track_a_manifest_sha256": sha256_file(track_a["directory"] / "manifest.json"),
            "track_b_manifest_sha256": sha256_file(track_b["directory"] / "manifest.json"),
            "bootstrap_reps": args.bootstrap_reps,
            "split_half_reps": args.split_half_reps,
            "seed": args.seed,
            "model_compatibility": {
                field: track_a["manifest"].get(field)
                for field in (
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
            },
        },
    )

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, sharey=True)
    for axis, pooling in zip(axes, POOLINGS):
        selected_rows = sorted(
            (row for row in rows if row["pooling"] == pooling),
            key=lambda row: int(row["layer_idx"]),
        )
        layers = np.asarray([row["layer_idx"] for row in selected_rows])
        cosine = np.asarray([row["cosine"] for row in selected_rows])
        low = np.asarray([row["bootstrap_ci_low"] for row in selected_rows])
        high = np.asarray([row["bootstrap_ci_high"] for row in selected_rows])
        null_low = np.asarray([row["null_ci_low"] for row in selected_rows])
        null_high = np.asarray([row["null_ci_high"] for row in selected_rows])
        axis.fill_between(
            layers,
            null_low,
            null_high,
            color="tab:orange",
            alpha=0.18,
            label="95% sign-flip null",
        )
        axis.fill_between(
            layers,
            low,
            high,
            color="tab:blue",
            alpha=0.22,
            label="95% grouped bootstrap",
        )
        axis.plot(layers, cosine, color="tab:blue", marker="o", markersize=3)
        candidate = summary["poolings"][pooling]["candidate_layer"]
        if candidate is not None:
            axis.scatter(
                [candidate],
                [cosine[candidate]],
                color="tab:green",
                marker="*",
                s=130,
                zorder=5,
                label=f"candidate layer {candidate}",
            )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_ylim(-1.0, 1.0)
        axis.set_ylabel("Signed cosine")
        axis.set_title(
            "Last non-special assistant token"
            if pooling == "last"
            else "Mean assistant content tokens"
        )
        axis.grid(alpha=0.2)
        axis.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("OLMo transformer block")
    axes[-1].set_xticks(np.arange(0, 32, 2))
    fig.suptitle("School of Reward Hacks direction vs Luna shortcut direction")
    fig.tight_layout()
    fig.savefig(output_dir / "cosine_vs_layer.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"Plot: {output_dir / 'cosine_vs_layer.png'}")


if __name__ == "__main__":
    main()
