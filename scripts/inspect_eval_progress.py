#!/usr/bin/env python3
"""Report safe aggregate progress from a realtime Inspect .eval archive."""

from __future__ import annotations

import argparse
import json
import statistics
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--expected", type=int)
    return parser.parse_args()


def output_tokens(sample: dict[str, Any]) -> int | None:
    usages = sample.get("model_usage") or {}
    values = [
        usage.get("output_tokens")
        for usage in usages.values()
        if isinstance(usage, dict) and usage.get("output_tokens") is not None
    ]
    return sum(int(value) for value in values) if values else None


def main() -> None:
    args = parse_args()
    log_path = args.log.expanduser().resolve()
    with zipfile.ZipFile(log_path) as archive:
        sample_names = sorted(
            name
            for name in archive.namelist()
            if name.startswith("samples/") and name.endswith(".json")
        )
        samples = [json.loads(archive.read(name)) for name in sample_names]

    stop_reasons: Counter[str] = Counter()
    token_counts: list[int] = []
    thinking_closed = 0
    errored = 0
    for sample in samples:
        output = sample.get("output") or {}
        choices = output.get("choices") or []
        reason = choices[0].get("stop_reason") if choices else None
        stop_reasons[str(reason or "missing")] += 1
        count = output_tokens(sample)
        if count is not None:
            token_counts.append(count)
        completion = str(output.get("completion") or "")
        thinking_closed += "</thinking>" in completion
        errored += sample.get("error") is not None

    completed = len(samples)
    report = {
        "completed": completed,
        "expected": args.expected,
        "percent_complete": (
            round(100 * completed / args.expected, 1) if args.expected else None
        ),
        "stop_reasons": dict(sorted(stop_reasons.items())),
        "output_tokens": {
            "minimum": min(token_counts) if token_counts else None,
            "median": statistics.median(token_counts) if token_counts else None,
            "maximum": max(token_counts) if token_counts else None,
            "total": sum(token_counts),
        },
        "thinking_close_present": thinking_closed,
        "sample_errors": errored,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
