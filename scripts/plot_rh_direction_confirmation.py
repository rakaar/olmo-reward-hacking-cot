#!/usr/bin/env python3
"""Plot pilot/confirmation agreement and frozen confirmation margins."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--confirmation-dir", type=Path, required=True)
    parser.add_argument("--disjoint-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    pilot_dir = args.pilot_dir.expanduser().resolve()
    confirmation_dir = args.confirmation_dir.expanduser().resolve()
    with np.load(pilot_dir / "directions.npz") as values:
        pilot = np.asarray(values["direction_unit"], dtype=np.float64)
    with np.load(confirmation_dir / "directions.npz") as values:
        confirmation = np.asarray(values["direction_unit"], dtype=np.float64)
    cosine = np.einsum("lh,lh->l", pilot, confirmation)
    disjoint_cosine = None
    disjoint_selected = None
    if args.disjoint_dir is not None:
        disjoint_dir = args.disjoint_dir.expanduser().resolve()
        with np.load(disjoint_dir / "directions.npz") as values:
            disjoint = np.asarray(values["direction_unit"], dtype=np.float64)
        disjoint_cosine = np.einsum("lh,lh->l", pilot, disjoint)
        disjoint_selected = int(
            json.loads((disjoint_dir / "layer_selection.json").read_text(encoding="utf-8"))[
                "selected_layer"
            ]
        )
    validation = read_csv(confirmation_dir / "layer_metrics.csv")
    holdout = read_csv(
        confirmation_dir / "holdout-evaluation" / "holdout_layer_metrics.csv"
    )
    selection = json.loads(
        (confirmation_dir / "layer_selection.json").read_text(encoding="utf-8")
    )
    layers = np.arange(len(cosine))
    val_margin = np.asarray(
        [float(row["validation_group_balanced_margin"]) for row in validation]
    )
    val_low = np.asarray([float(row["validation_margin_ci_low"]) for row in validation])
    val_high = np.asarray([float(row["validation_margin_ci_high"]) for row in validation])
    test_margin = np.asarray([float(row["balanced_margin"]) for row in holdout])
    test_low = np.asarray([float(row["margin_ci_low"]) for row in holdout])
    test_high = np.asarray([float(row["margin_ci_high"]) for row in holdout])
    selected = int(selection["selected_layer"])

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "layer",
                "pilot_confirmation_cosine",
                "pilot_disjoint_replication_cosine",
                "confirmation_validation_margin",
                "confirmation_validation_ci_low",
                "confirmation_validation_ci_high",
                "confirmation_test_margin",
                "confirmation_test_ci_low",
                "confirmation_test_ci_high",
                "qualified",
                "selected",
            ],
        )
        writer.writeheader()
        for layer in layers:
            writer.writerow(
                {
                    "layer": int(layer),
                    "pilot_confirmation_cosine": float(cosine[layer]),
                    "pilot_disjoint_replication_cosine": (
                        "" if disjoint_cosine is None else float(disjoint_cosine[layer])
                    ),
                    "confirmation_validation_margin": float(val_margin[layer]),
                    "confirmation_validation_ci_low": float(val_low[layer]),
                    "confirmation_validation_ci_high": float(val_high[layer]),
                    "confirmation_test_margin": float(test_margin[layer]),
                    "confirmation_test_ci_low": float(test_low[layer]),
                    "confirmation_test_ci_high": float(test_high[layer]),
                    "qualified": int(layer in selection["qualified_layers"]),
                    "selected": int(layer == selected),
                }
            )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    axes[0].plot(
        layers,
        cosine,
        color="#2457a6",
        linewidth=2.4,
        marker="o",
        ms=3,
        label="pilot vs full confirmation (overlapping)",
    )
    if disjoint_cosine is not None:
        axes[0].plot(
            layers,
            disjoint_cosine,
            color="#d17a00",
            linewidth=2.4,
            marker="o",
            ms=3,
            label="pilot vs 735 disjoint examples",
        )
        axes[0].axvline(
            disjoint_selected,
            color="#d17a00",
            linestyle=":",
            linewidth=1.3,
        )
    axes[0].axvline(selected, color="#c23b22", linestyle="--", linewidth=1.5)
    axes[0].set(
        title="RH direction stability and disjoint replication",
        xlabel="Post-block layer (0-indexed)",
        ylabel="Signed cosine: pilot vs confirmation",
        xlim=(0, 31),
        ylim=(
            min(
                0.96,
                float(cosine.min()) - 0.005,
                float(disjoint_cosine.min()) - 0.005
                if disjoint_cosine is not None
                else 1.0,
            ),
            1.001,
        ),
    )
    axes[0].text(
        selected + 0.4,
        float(cosine[selected]) - 0.006,
        f"selected L{selected}\ncos={cosine[selected]:.3f}",
        color="#8e2c1b",
        fontsize=9,
    )
    axes[0].legend(frameon=True, loc="lower center", fontsize=8)

    axes[1].plot(layers, val_margin, color="#6b4c9a", linewidth=2.2, label="validation")
    axes[1].fill_between(layers, val_low, val_high, color="#6b4c9a", alpha=0.15)
    axes[1].plot(layers, test_margin, color="#11866f", linewidth=2.2, label="untouched test")
    axes[1].fill_between(layers, test_low, test_high, color="#11866f", alpha=0.15)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].axvline(selected, color="#c23b22", linestyle="--", linewidth=1.5)
    axes[1].set(
        title="Confirmation direction has positive held-out margins",
        xlabel="Post-block layer (0-indexed)",
        ylabel="Family/task-balanced signed margin",
        xlim=(0, 31),
    )
    axes[1].legend(frameon=True)
    fig.suptitle("School-of-Reward-Hacks direction: internal confirmation", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"output": str(args.output), "selected_layer": selected}, indent=2))


if __name__ == "__main__":
    main()
