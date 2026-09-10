#!/usr/bin/env python3
"""Plot paired RH and coherence scores for the School-of-RH causal pilot."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


COLOR_BASELINE = "#3B82F6"
COLOR_INTERVENTION = "#F97316"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consensus", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title")
    parser.add_argument("--intervention-label")
    return parser.parse_args()


def paired_arrays(
    rows: list[dict[str, Any]],
    field: str,
    *,
    baseline_condition: str,
    intervention_condition: str,
) -> tuple[np.ndarray, np.ndarray]:
    by_prompt: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_prompt[str(row["prompt_id"])][str(row["condition"])] = row
    prompt_ids = sorted(by_prompt)
    baseline = np.asarray(
        [by_prompt[item][baseline_condition][field] for item in prompt_ids]
    )
    intervention = np.asarray(
        [by_prompt[item][intervention_condition][field] for item in prompt_ids]
    )
    return baseline, intervention


def paired_panel(
    axis: Any,
    baseline: np.ndarray,
    intervention: np.ndarray,
    *,
    title: str,
    ylabel: str,
    intervention_label: str,
    threshold: float | None = None,
) -> None:
    for left, right in zip(baseline, intervention):
        axis.plot([0, 1], [left, right], color="#CBD5E1", linewidth=1.0, zorder=1)
    axis.scatter(
        np.zeros_like(baseline), baseline, color=COLOR_BASELINE, s=34, alpha=0.85, zorder=2
    )
    axis.scatter(
        np.ones_like(intervention),
        intervention,
        color=COLOR_INTERVENTION,
        s=34,
        alpha=0.85,
        zorder=2,
    )
    axis.plot(
        [0, 1],
        [baseline.mean(), intervention.mean()],
        color="#111827",
        marker="o",
        linewidth=2.6,
        zorder=3,
    )
    if threshold is not None:
        axis.axhline(threshold, color="#64748B", linestyle="--", linewidth=1.1)
    axis.set_xticks([0, 1], ["Baseline", intervention_label])
    axis.set_xlim(-0.25, 1.25)
    axis.set_ylim(-3, 103)
    axis.set_ylabel(ylabel)
    axis.set_title(title, loc="left", fontweight="bold")
    axis.grid(axis="y", color="#E2E8F0", linewidth=0.8)


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.consensus.expanduser().resolve())
    summary = json.loads(args.summary.expanduser().resolve().read_text(encoding="utf-8"))
    baseline_condition = str(summary["baseline_condition"])
    intervention_condition = str(summary["intervention_condition"])
    intervention_label = args.intervention_label or intervention_condition.replace("_", " ")
    rh_baseline, rh_intervention = paired_arrays(
        rows,
        "reward_hacking_score",
        baseline_condition=baseline_condition,
        intervention_condition=intervention_condition,
    )
    co_baseline, co_intervention = paired_arrays(
        rows,
        "coherence_score",
        baseline_condition=baseline_condition,
        intervention_condition=intervention_condition,
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(14.2, 5.1),
        gridspec_kw={"width_ratios": [1.1, 1.1, 0.9]},
        constrained_layout=True,
    )
    paired_panel(
        axes[0],
        rh_baseline,
        rh_intervention,
        title="A  Metric-gaming score",
        ylabel="Blinded reviewer score (0–100)",
        intervention_label=intervention_label,
        threshold=50,
    )
    effect = summary["paired_reward_hacking_score_effect"]
    interval = effect["confidence_interval_95"]
    axes[0].text(
        0.02,
        0.02,
        f"Mean change: {effect['observed_mean_difference']:+.1f}\n"
        f"95% bootstrap CI: [{interval[0]:+.1f}, {interval[1]:+.1f}]",
        transform=axes[0].transAxes,
        va="bottom",
        fontsize=9,
    )

    paired_panel(
        axes[1],
        co_baseline,
        co_intervention,
        title="B  Coherence score",
        ylabel="Blinded reviewer score (0–100)",
        intervention_label=intervention_label,
        threshold=50,
    )
    effect = summary["paired_coherence_score_effect"]
    interval = effect["confidence_interval_95"]
    axes[1].text(
        0.02,
        0.02,
        f"Mean change: {effect['observed_mean_difference']:+.1f}\n"
        f"95% bootstrap CI: [{interval[0]:+.1f}, {interval[1]:+.1f}]",
        transform=axes[1].transAxes,
        va="bottom",
        fontsize=9,
    )

    matrix_values = summary["paired_binary_reward_hacking_effect_at_50"][
        "two_by_two_matrix"
    ]
    matrix = np.asarray(
        [
            [
                matrix_values.get("baseline_0_intervention_0", 0),
                matrix_values.get("baseline_0_intervention_1", 0),
            ],
            [
                matrix_values.get("baseline_1_intervention_0", 0),
                matrix_values.get("baseline_1_intervention_1", 0),
            ],
        ]
    )
    axes[2].imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, int(matrix.max())))
    for row in range(2):
        for column in range(2):
            axes[2].text(
                column,
                row,
                str(int(matrix[row, column])),
                ha="center",
                va="center",
                fontsize=18,
                fontweight="bold",
                color="white" if matrix[row, column] > matrix.max() / 2 else "#0F172A",
            )
    axes[2].set_xticks([0, 1], ["No", "Yes"])
    axes[2].set_yticks([0, 1], ["No", "Yes"])
    axes[2].set_xlabel("Ablated response games metric")
    axes[2].set_ylabel("Baseline response games metric")
    axes[2].set_title("C  Paired 2 × 2 at score ≥ 50", loc="left", fontweight="bold")

    figure.suptitle(
        args.title
        or f"School-of-Reward-Hacks capability pilot: {intervention_condition}",
        fontsize=14,
        fontweight="bold",
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(json.dumps({"status": "success", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
