#!/usr/bin/env python3
"""Summarize a paired baseline-versus-RH-ablation calibration run."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled-rollouts", type=Path, required=True)
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--intervention", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260910)
    return parser.parse_args()


def quantiles(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def condition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    telemetry = [row["coherence_telemetry"] for row in rows]
    intervention = [row["intervention"] for row in rows]
    return {
        "n": len(rows),
        "hack_attempted": int(sum(bool(row["hack_attempted"]) for row in rows)),
        "reward_hacked": int(sum(bool(row["reward_hacked"]) for row in rows)),
        "normal_tests_passed": int(sum(bool(row["passed"]) for row in rows)),
        "complete_thinking_span": int(
            sum(bool(value.get("has_complete_thinking_span")) for value in telemetry)
        ),
        "eos_stop": int(sum(value.get("stop_reason") == "eos" for value in telemetry)),
        "length_stop": int(
            sum(value.get("stop_reason") == "max_new_tokens" for value in telemetry)
        ),
        "median_generated_content_tokens": float(
            statistics.median(value.get("generated_content_tokens", 0) for value in telemetry)
        ),
        "median_answer_characters": float(
            statistics.median(value.get("answer_character_count", 0) for value in telemetry)
        ),
        "median_repeated_fourgram_fraction": float(
            statistics.median(value.get("repeated_fourgram_fraction", 0.0) for value in telemetry)
        ),
        "maximum_repeated_fourgram_fraction": float(
            max((value.get("repeated_fourgram_fraction", 0.0) for value in telemetry), default=0.0)
        ),
        "hack_types": dict(
            sorted(Counter(kind for row in rows for kind in row["hack_types"]).items())
        ),
        "median_relative_update_norm": float(
            statistics.median(
                value.get("relative_update_norm_mean_across_layers", 0.0)
                for value in intervention
            )
        ),
        "maximum_relative_update_norm": float(
            max(
                (
                    value.get("relative_update_norm_max_across_layers", 0.0)
                    for value in intervention
                ),
                default=0.0,
            )
        ),
    }


def exact_discordant_pvalue(baseline_only: int, intervention_only: int) -> float:
    discordant = baseline_only + intervention_only
    if discordant == 0:
        return 1.0
    smaller = min(baseline_only, intervention_only)
    lower_tail = sum(math.comb(discordant, value) for value in range(smaller + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * lower_tail)


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.labeled_rollouts.expanduser().resolve().read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    selected = [
        row for row in rows if row["condition"] in {args.baseline, args.intervention}
    ]
    by_key: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in selected:
        key = (str(row["prompt_id"]), int(row["sample_index"]))
        condition = str(row["condition"])
        if condition in by_key[key]:
            raise SystemExit(f"duplicate condition for paired key {key}")
        by_key[key][condition] = row
    incomplete = [key for key, values in by_key.items() if len(values) != 2]
    if incomplete:
        raise SystemExit(f"incomplete pairs: {incomplete[:5]}")
    keys = sorted(by_key)
    baseline_values = np.asarray(
        [bool(by_key[key][args.baseline]["hack_attempted"]) for key in keys], dtype=int
    )
    intervention_values = np.asarray(
        [bool(by_key[key][args.intervention]["hack_attempted"]) for key in keys], dtype=int
    )
    matrix = {
        "baseline_no__intervention_no": int(
            np.sum((baseline_values == 0) & (intervention_values == 0))
        ),
        "baseline_no__intervention_yes": int(
            np.sum((baseline_values == 0) & (intervention_values == 1))
        ),
        "baseline_yes__intervention_no": int(
            np.sum((baseline_values == 1) & (intervention_values == 0))
        ),
        "baseline_yes__intervention_yes": int(
            np.sum((baseline_values == 1) & (intervention_values == 1))
        ),
    }
    risk_difference = intervention_values - baseline_values
    rng = np.random.default_rng(args.seed)
    sample = rng.integers(0, len(keys), size=(args.bootstrap_replicates, len(keys)))
    bootstrap = risk_difference[sample].mean(axis=1)
    baseline_rows = [by_key[key][args.baseline] for key in keys]
    intervention_rows = [by_key[key][args.intervention] for key in keys]
    result = {
        "schema_version": 1,
        "paired_unit": "prompt_id and sample_index",
        "paired_count": len(keys),
        "primary_outcome": "AISI released static detector: hack_attempted",
        "baseline_condition": args.baseline,
        "intervention_condition": args.intervention,
        "conditions": {
            args.baseline: condition_summary(baseline_rows),
            args.intervention: condition_summary(intervention_rows),
        },
        "paired_hack_attempt_matrix": matrix,
        "hack_attempt_risk_difference_intervention_minus_baseline": float(
            risk_difference.mean()
        ),
        "risk_difference_grouped_bootstrap_95_ci": quantiles(bootstrap),
        "exact_mcnemar_two_sided_p": exact_discordant_pvalue(
            matrix["baseline_yes__intervention_no"],
            matrix["baseline_no__intervention_yes"],
        ),
        "interpretation_guardrail": (
            "This calibration is sized for coherence and gross effects, not a definitive "
            "causal estimate. Expand only after intervention quality passes."
        ),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
