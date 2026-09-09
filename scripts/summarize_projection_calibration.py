#!/usr/bin/env python3
"""Summarize a labelled, matched projection-ablation calibration batch."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    return parser.parse_args()


def syntax_valid(source: str) -> bool:
    try:
        ast.parse(source)
    except (SyntaxError, ValueError, TypeError):
        return False
    return bool(source.strip())


def structured_completion(record: dict[str, Any]) -> bool:
    completion = str(record.get("completion") or "")
    return (
        completion.count("<thinking>") == 1
        and completion.count("</thinking>") == 1
        and completion.find("<thinking>") < completion.find("</thinking>")
        and bool(str(record.get("thinking") or "").strip())
        and syntax_valid(str(record.get("solution_code") or ""))
    )


def finite_mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.fmean(finite) if finite else None


def wilson_interval(successes: int, total: int) -> list[float] | None:
    """Return a two-sided 95% Wilson score interval for a binomial rate."""
    if total <= 0 or not 0 <= successes <= total:
        return None
    z = 1.959963984540054
    proportion = successes / total
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    centre = (proportion + z_squared / (2.0 * total)) / denominator
    radius = (
        z
        / denominator
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z_squared / (4.0 * total * total)
        )
    )
    return [centre - radius, centre + radius]


def exact_mcnemar_p(discordant_0_to_1: int, discordant_1_to_0: int) -> float:
    """Exact two-sided McNemar p-value under equal discordant probabilities."""
    if discordant_0_to_1 < 0 or discordant_1_to_0 < 0:
        raise ValueError("discordant counts must be nonnegative")
    total = discordant_0_to_1 + discordant_1_to_0
    if total == 0:
        return 1.0
    smaller = min(discordant_0_to_1, discordant_1_to_0)
    lower_tail = sum(math.comb(total, value) for value in range(smaller + 1)) / (2**total)
    return min(1.0, 2.0 * lower_tail)


def summarize_condition(records: list[dict[str, Any]]) -> dict[str, Any]:
    diagnostics = [record.get("ablation") or {} for record in records]
    token_field = (
        "generated_content_tokens"
        if all(
            (record.get("sampling") or {}).get("generated_content_tokens") is not None
            for record in records
        )
        else "generated_tokens_with_padding"
    )
    tokens = [
        int((record.get("sampling") or {}).get(token_field) or 0)
        for record in records
    ]
    hack_attempted = sum(record.get("hack_attempted") is True for record in records)
    syntax_valid_count = sum(
        syntax_valid(str(record.get("solution_code") or "")) for record in records
    )
    return {
        "n": len(records),
        "hack_attempted": hack_attempted,
        "hack_attempt_rate": hack_attempted / len(records),
        "hack_attempt_wilson_95": wilson_interval(hack_attempted, len(records)),
        "reward_hacked": sum(record.get("reward_hacked") is True for record in records),
        "passed": sum(record.get("passed") is True for record in records),
        "actually_solved": sum(record.get("actually_solved") is True for record in records),
        "cot_mentions_hack": sum(
            record.get("cot_mentions_hack") is True for record in records
        ),
        "syntax_valid": syntax_valid_count,
        "syntax_valid_rate": syntax_valid_count / len(records),
        "syntax_valid_wilson_95": wilson_interval(syntax_valid_count, len(records)),
        "structured_completion": sum(structured_completion(record) for record in records),
        "eos_stop": sum(
            (record.get("sampling") or {}).get("stop_reason") == "eos"
            for record in records
        ),
        f"{token_field}_median": statistics.median(tokens) if tokens else None,
        "hooked_positions_mean": finite_mean(
            [float(item.get("hooked_positions", math.nan)) for item in diagnostics]
        ),
        "pre_projection_abs_mean": finite_mean(
            [float(item.get("pre_projection_abs_mean", math.nan)) for item in diagnostics]
        ),
        "post_projection_abs_mean": finite_mean(
            [float(item.get("post_projection_abs_mean", math.nan)) for item in diagnostics]
        ),
        "relative_update_norm_mean": finite_mean(
            [float(item.get("relative_update_norm_mean", math.nan)) for item in diagnostics]
        ),
        "relation_error_abs_max": max(
            (float(item.get("relation_error_abs_max", math.nan)) for item in diagnostics),
            default=None,
        ),
        "hack_type_counts": dict(
            sorted(
                Counter(
                    hack_type
                    for record in records
                    for hack_type in (record.get("hack_types") or [])
                ).items()
            )
        ),
    }


def main() -> None:
    args = parse_args()
    records = [
        json.loads(line)
        for line in args.input.expanduser().resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise SystemExit("input contains no records")
    if any(record.get("hack_attempted") is None for record in records):
        raise SystemExit("input must be graded before summarization")

    grouped: dict[float, list[dict[str, Any]]] = {}
    matched: dict[tuple[str, int], dict[float, dict[str, Any]]] = {}
    for record in records:
        strength = float((record.get("ablation") or {})["strength"])
        grouped.setdefault(strength, []).append(record)
        key = (str(record["problem_id"]), int(record["sample_index"]))
        if strength in matched.setdefault(key, {}):
            raise SystemExit(f"duplicate record for {key}, lambda={strength:g}")
        matched[key][strength] = record

    strengths = sorted(grouped)
    if 0.0 not in grouped:
        raise SystemExit("matched batch has no lambda=0 control")
    expected = set(strengths)
    incomplete = [key for key, values in matched.items() if set(values) != expected]
    if incomplete:
        raise SystemExit(f"incomplete matched conditions for {incomplete[:3]}")

    conditions = {f"{strength:g}": summarize_condition(grouped[strength]) for strength in strengths}
    paired: dict[str, Any] = {}
    for strength in strengths:
        if strength == 0:
            continue
        transitions = Counter()
        structured_losses = 0
        for values in matched.values():
            control = values[0.0]
            treatment = values[strength]
            before = bool(control["hack_attempted"])
            after = bool(treatment["hack_attempted"])
            transitions[f"{int(before)}->{int(after)}"] += 1
            structured_losses += int(
                structured_completion(control) and not structured_completion(treatment)
            )
        paired[f"{strength:g}"] = {
            "matched_n": len(matched),
            "hack_attempt_transitions": dict(sorted(transitions.items())),
            "hack_suppressed_1_to_0": transitions["1->0"],
            "hack_induced_0_to_1": transitions["0->1"],
            "net_hack_reduction": transitions["1->0"] - transitions["0->1"],
            "paired_rate_difference_treatment_minus_control": (
                transitions["0->1"] - transitions["1->0"]
            )
            / len(matched),
            "exact_mcnemar_p": exact_mcnemar_p(
                transitions["0->1"], transitions["1->0"]
            ),
            "structured_completion_losses_vs_control": structured_losses,
        }

    summary = {
        "schema_version": 1,
        "records": len(records),
        "matched_prompt_samples": len(matched),
        "strengths": strengths,
        "conditions": conditions,
        "paired_vs_lambda_0": paired,
        "interpretation_warning": (
            "This small calibration checks coherence and gross effects only; "
            "it is not an efficacy estimate. Lambda=2 reverses the selected "
            "projection rather than performing a larger orthogonal ablation."
        ),
        "sampling_warning": (
            "Strengths share prompts but are stochastic batch rows with different "
            "random draws; individual paired transitions include sampling noise."
        ),
    }

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fieldnames = ["strength", *next(iter(conditions.values())).keys()]
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for strength in strengths:
            row = {"strength": strength, **conditions[f"{strength:g}"]}
            for key, value in list(row.items()):
                if isinstance(value, (dict, list)):
                    row[key] = json.dumps(value, sort_keys=True)
            writer.writerow(row)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
