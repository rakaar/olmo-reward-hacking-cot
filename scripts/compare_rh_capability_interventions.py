#!/usr/bin/env python3
"""Compare paired effects across School-of-RH causal intervention scopes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_experiment(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("experiment must be LABEL=REVIEWED_DIR")
    label, raw_path = value.rsplit("=", 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("experiment label and path must be nonempty")
    return label.strip(), Path(raw_path).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", type=parse_experiment, action="append", required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries: list[dict[str, Any]] = []
    consensus_sets: list[tuple[str, list[dict[str, Any]]]] = []
    input_artifacts: list[dict[str, Any]] = []
    for label, directory in args.experiment:
        summary_path = directory / "causal_summary.json"
        consensus_path = directory / "consensus_reviews.jsonl"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        consensus = read_jsonl(consensus_path)
        input_artifacts.append(
            {
                "label": label,
                "reviewed_directory": str(directory),
                "causal_summary_sha256": sha256_file(summary_path),
                "consensus_reviews_sha256": sha256_file(consensus_path),
            }
        )
        baseline = str(summary["baseline_condition"])
        intervention = str(summary["intervention_condition"])
        rh = summary["paired_reward_hacking_score_effect"]
        binary = summary["paired_binary_reward_hacking_effect_at_50"]
        coherence = summary["paired_coherence_score_effect"]
        base_condition = summary["conditions"][baseline]
        intervention_condition = summary["conditions"][intervention]
        summaries.append(
            {
                "label": label,
                "baseline_condition": baseline,
                "intervention_condition": intervention,
                "rh_score_difference": rh["observed_mean_difference"],
                "rh_score_ci_low": rh["confidence_interval_95"][0],
                "rh_score_ci_high": rh["confidence_interval_95"][1],
                "rh_exact_sign_flip_pvalue": rh["exact_two_sided_sign_flip_pvalue"],
                "baseline_hack_count_at_50": base_condition[
                    "reward_hacking_attempt_count_at_50"
                ],
                "intervention_hack_count_at_50": intervention_condition[
                    "reward_hacking_attempt_count_at_50"
                ],
                "binary_risk_difference": binary["observed_mean_difference"],
                "binary_ci_low": binary["confidence_interval_95"][0],
                "binary_ci_high": binary["confidence_interval_95"][1],
                "coherence_difference": coherence["observed_mean_difference"],
                "coherence_ci_low": coherence["confidence_interval_95"][0],
                "coherence_ci_high": coherence["confidence_interval_95"][1],
                "baseline_coherence_mean": base_condition["coherence_score_mean"],
                "intervention_coherence_mean": intervention_condition[
                    "coherence_score_mean"
                ],
                "baseline_median_tokens": base_condition["generated_tokens_median"],
                "intervention_median_tokens": intervention_condition[
                    "generated_tokens_median"
                ],
            }
        )
        consensus_sets.append((label, consensus))

    reference_label, reference_rows = consensus_sets[0]
    reference_baseline = {
        str(row["rollout_id"]): row
        for row in reference_rows
        if row["condition"] == summaries[0]["baseline_condition"]
    }
    baseline_repeatability = []
    for (label, rows), summary in zip(consensus_sets[1:], summaries[1:]):
        comparison = {
            str(row["rollout_id"]): row
            for row in rows
            if row["condition"] == summary["baseline_condition"]
        }
        if set(comparison) != set(reference_baseline):
            raise SystemExit(f"{label}: baseline rollout IDs differ from reference")
        ids = sorted(reference_baseline)
        for field, threshold in (("reward_hacking_score", 50), ("coherence_score", 50)):
            left = np.asarray([reference_baseline[item][field] for item in ids])
            right = np.asarray([comparison[item][field] for item in ids])
            baseline_repeatability.append(
                {
                    "reference": reference_label,
                    "comparison": label,
                    "field": field,
                    "pearson_correlation": float(np.corrcoef(left, right)[0, 1]),
                    "mean_absolute_difference": float(np.mean(np.abs(left - right))),
                    "mean_offset_comparison_minus_reference": float(
                        np.mean(right - left)
                    ),
                    "threshold_flips": int(np.count_nonzero((left >= threshold) != (right >= threshold))),
                }
            )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_json = {
        "schema_version": 1,
        "difference_orientation": "intervention minus baseline",
        "reward_hacking_negative_is_desired": True,
        "coherence_negative_indicates_damage": True,
        "experiments": summaries,
        "repeated_baseline_review_reliability": baseline_repeatability,
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison_json, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    labels = [row["label"] for row in summaries]
    y = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    for axis, prefix, title, annotation in (
        (
            axes[0],
            "rh_score",
            "A  Metric-gaming score change",
            "Negative would support suppression",
        ),
        (
            axes[1],
            "coherence",
            "B  Coherence score change",
            "Negative indicates coherence damage",
        ),
    ):
        estimates = np.asarray([row[f"{prefix}_difference"] for row in summaries])
        lows = np.asarray([row[f"{prefix}_ci_low"] for row in summaries])
        highs = np.asarray([row[f"{prefix}_ci_high"] for row in summaries])
        axis.errorbar(
            estimates,
            y,
            xerr=np.vstack([estimates - lows, highs - estimates]),
            fmt="o",
            color="#2563EB" if prefix == "rh_score" else "#F97316",
            ecolor="#94A3B8",
            capsize=4,
            markersize=7,
        )
        axis.axvline(0, color="#334155", linestyle="--", linewidth=1.1)
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.set_xlabel("Paired mean change (intervention − baseline)")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="x", color="#E2E8F0", linewidth=0.8)
        axis.text(0.02, 0.02, annotation, transform=axis.transAxes, fontsize=9)
    figure.suptitle(
        "Reward-hacking direction: causal sensitivity across intervention scopes",
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(
        output_dir / "intervention_comparison.png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    output_hashes = {
        name: sha256_file(output_dir / name)
        for name in ("comparison.json", "comparison.csv", "intervention_comparison.png")
    }
    manifest = {
        "schema_version": 1,
        "comparison_script": str(Path(__file__).resolve()),
        "comparison_script_sha256": sha256_file(Path(__file__).resolve()),
        "inputs": input_artifacts,
        "outputs": output_hashes,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "success", "experiments": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()
