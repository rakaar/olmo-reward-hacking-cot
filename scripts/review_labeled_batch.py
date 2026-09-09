#!/usr/bin/env python3
"""Summarize and render bounded excerpts from a labelled rollout batch."""

from __future__ import annotations

import argparse
import ast
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--thinking-chars", type=int, default=800)
    parser.add_argument("--response-chars", type=int, default=500)
    return parser.parse_args()


def syntax_valid(source: str) -> bool:
    try:
        ast.parse(source)
    except SyntaxError:
        return False
    return True


def bounded(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + " ...[truncated]"


def main() -> None:
    args = parse_args()
    records = [
        json.loads(line)
        for line in args.input.expanduser().resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.limit]

    thinking_lengths = [len(str(record.get("thinking") or "")) for record in records]
    syntax_counts = Counter(
        syntax_valid(str(record.get("solution_code") or "")) for record in records
    )
    stop_reasons = Counter(
        str(
            (((record.get("model_output") or {}).get("choices") or [{}])[0]).get(
                "stop_reason"
            )
            or "missing"
        )
        for record in records
    )
    metrics = {
        "rollouts": len(records),
        "hack_attempted": sum(
            record.get("hack_attempted", bool(record.get("hack_types"))) is True
            for record in records
        ),
        "reward_hacked": sum(record.get("reward_hacked") is True for record in records),
        "actually_solved": sum(record.get("actually_solved") is True for record in records),
        "passed": sum(record.get("passed") is True for record in records),
        "cot_mentions_hack": sum(
            record.get("cot_mentions_hack") is True for record in records
        ),
        "thinking_chars": {
            "minimum": min(thinking_lengths) if thinking_lengths else None,
            "median": statistics.median(thinking_lengths) if thinking_lengths else None,
            "maximum": max(thinking_lengths) if thinking_lengths else None,
            "empty": sum(length == 0 for length in thinking_lengths),
        },
        "syntax_valid": {
            "true": syntax_counts[True],
            "false": syntax_counts[False],
        },
        "stop_reasons": dict(sorted(stop_reasons.items())),
        "hack_type_counts": dict(
            sorted(
                Counter(
                    hack
                    for record in records
                    for hack in record.get("hack_types") or []
                ).items()
            )
        ),
        "grading_errors": sum(
            ((record.get("grading") or {}).get("error") is not None)
            for record in records
        ),
    }
    print("BATCH_METRICS")
    print(json.dumps(metrics, indent=2))

    for index, record in enumerate(records, start=1):
        print(f"\nROLLOUT {index}")
        print(
            json.dumps(
                {
                    "problem_id": record.get("problem_id"),
                    "epoch": record.get("epoch"),
                    "ablation_strength": (record.get("ablation") or {}).get("strength"),
                    "ablation_layer": (record.get("ablation") or {}).get("layer_index"),
                    "passed": record.get("passed"),
                    "actually_solved": record.get("actually_solved"),
                    "reward_hacked": record.get("reward_hacked"),
                    "hack_types": record.get("hack_types"),
                    "cot_mentions_hack": record.get("cot_mentions_hack"),
                    "syntax_valid": syntax_valid(
                        str(record.get("solution_code") or "")
                    ),
                    "generated_content_tokens": (record.get("sampling") or {}).get(
                        "generated_content_tokens"
                    ),
                    "generated_tokens_with_padding": (record.get("sampling") or {}).get(
                        "generated_tokens_with_padding"
                    ),
                    "relative_update_norm_mean": (record.get("ablation") or {}).get(
                        "relative_update_norm_mean"
                    ),
                },
                ensure_ascii=False,
            )
        )
        print("THINKING")
        print(bounded(str(record.get("thinking") or ""), args.thinking_chars))
        print("RESPONSE_CODE")
        print(
            bounded(
                str(record.get("solution_code") or ""),
                args.response_chars,
            )
        )


if __name__ == "__main__":
    main()
