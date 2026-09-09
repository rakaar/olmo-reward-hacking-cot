#!/usr/bin/env python3
"""Export AISI/Inspect reward-hacking logs to analysis-ready JSONL."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from inspect_ai.log import read_eval_log
from rh_envs.common import (
    _HACK_COT_KEYWORDS,
    _detect_always_equal_hack,
    _detect_conftest_hack,
    _detect_exit_hack,
    extract_code_and_files,
)


THINKING_RE = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="An Inspect .eval log, or a directory containing exactly one .eval log",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def find_log(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    logs = sorted(path.rglob("*.eval"))
    if len(logs) != 1:
        raise SystemExit(f"expected exactly one .eval log under {path}, found {len(logs)}")
    return logs[0]


def json_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    return value


def message_content(message: Any) -> str | list[Any] | None:
    content = getattr(message, "content", None)
    return json_value(content)


def first_message(messages: list[Any], role: str) -> str | list[Any] | None:
    for message in messages:
        if getattr(message, "role", None) == role:
            return message_content(message)
    return None


def find_score(scores: dict[str, Any] | None, suffix: str) -> Any | None:
    for name, score in (scores or {}).items():
        if name == suffix or name.endswith("/" + suffix) or suffix in name:
            return score
    return None


def correct(value: Any) -> bool | None:
    if value == "C":
        return True
    if value == "I":
        return False
    return None


def dict_label(score: Any | None, key: str) -> bool | None:
    if score is None or not isinstance(score.value, dict):
        return None
    return correct(score.value.get(key))


def main() -> None:
    args = parse_args()
    log_path = find_log(args.input)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log = read_eval_log(str(log_path))
    if not log.samples:
        raise SystemExit(f"log has no samples: {log_path}")

    records: list[dict[str, Any]] = []
    for sample in log.samples:
        completion = sample.output.completion if sample.output else ""
        thinking_match = THINKING_RE.search(completion)
        thinking = thinking_match.group(1) if thinking_match else ""
        final_response = (
            completion[thinking_match.end() :].lstrip()
            if thinking_match
            else completion
        )
        solution_code, extra_files = extract_code_and_files(completion)

        reward_score = find_score(sample.scores, "proxy_reward_hacking_scorer")
        cot_score = find_score(sample.scores, "proxy_cot_faithfulness_scorer")
        training_score = find_score(sample.scores, "training_passed_scorer")

        cot_mentions_hack = dict_label(cot_score, "cot_mentions_hack")
        if cot_mentions_hack is None:
            cot_mentions_hack = bool(_HACK_COT_KEYWORDS.search(thinking))

        metadata = json_value(sample.metadata or {})
        hack_config = metadata.get("hack_config") or {}
        hack_types = [
            name
            for name, label in (
                ("always_equal", dict_label(reward_score, "rh_always_equal")),
                ("exit", dict_label(reward_score, "rh_exit")),
                ("conftest", dict_label(reward_score, "rh_conftest")),
            )
            if label is True
        ]
        if reward_score is None:
            all_code = solution_code + "".join(
                "\n" + content for content in extra_files.values()
            )
            if hack_config.get("always_equal", False) and _detect_always_equal_hack(
                all_code
            ):
                hack_types.append("always_equal")
            if hack_config.get("exit", False) and _detect_exit_hack(all_code):
                hack_types.append("exit")
            if hack_config.get("conftest", False) and _detect_conftest_hack(
                extra_files
            ):
                hack_types.append("conftest")

        messages = sample.messages or []
        usage = json_value(sample.model_usage or {})
        record = {
            "schema_version": 1,
            "rollout_id": (
                str(sample.uuid)
                if sample.uuid is not None
                else f"{sample.id}::epoch-{sample.epoch}"
            ),
            "problem_id": str(sample.id),
            "epoch": sample.epoch,
            "system_prompt": first_message(messages, "system"),
            "problem_prompt": first_message(messages, "user") or json_value(sample.input),
            "messages": [json_value(message) for message in messages],
            "target_tests": json_value(sample.target),
            "thinking": thinking,
            "final_response": final_response,
            "completion": completion,
            "solution_code": solution_code,
            "extra_files": extra_files,
            "passed": dict_label(reward_score, "passed"),
            "actually_solved": dict_label(reward_score, "actually_solved"),
            "hack_attempted": bool(hack_types),
            "reward_hacked": dict_label(reward_score, "reward_hacked"),
            "hack_types": hack_types,
            "cot_mentions_hack": cot_mentions_hack,
            "thinking_format_score": (
                find_score(sample.scores, "thinking_format_scorer").value
                if find_score(sample.scores, "thinking_format_scorer") is not None
                else None
            ),
            "training_passed_score": (
                training_score.value if training_score is not None else None
            ),
            "hack_scorer_explanation": (
                reward_score.explanation if reward_score is not None else None
            ),
            "problem_metadata": metadata,
            "model_output": json_value(sample.output) if sample.output else None,
            "model_usage": usage,
            "started_at": sample.started_at,
            "completed_at": sample.completed_at,
            "total_time_seconds": sample.total_time,
            "working_time_seconds": sample.working_time,
            "error": json_value(sample.error) if sample.error else None,
        }
        records.append(record)

    jsonl_path = output_dir / "rollouts.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    reward_counts = Counter(record["reward_hacked"] for record in records)
    attempt_counts = Counter(record["hack_attempted"] for record in records)
    pass_counts = Counter(record["passed"] for record in records)
    cot_counts = Counter(record["cot_mentions_hack"] for record in records)
    hack_type_counts = Counter(
        hack_type for record in records for hack_type in record["hack_types"]
    )
    valid_reward_labels = reward_counts[True] + reward_counts[False]
    reward_hack_rate = (
        reward_counts[True] / valid_reward_labels if valid_reward_labels else None
    )
    hacked_records = [record for record in records if record["reward_hacked"] is True]
    hidden_hack_count = sum(
        record["cot_mentions_hack"] is False for record in hacked_records
    )

    summary = {
        "schema_version": 1,
        "source_log": str(log_path),
        "eval_status": log.status,
        "rollouts": len(records),
        "unique_problem_ids": len({record["problem_id"] for record in records}),
        "hack_attempted": {
            "true": attempt_counts[True],
            "false": attempt_counts[False],
            "rate": attempt_counts[True] / len(records) if records else None,
        },
        "reward_hacked": {
            "true": reward_counts[True],
            "false": reward_counts[False],
            "missing": reward_counts[None],
            "rate": reward_hack_rate,
        },
        "passed": {
            "true": pass_counts[True],
            "false": pass_counts[False],
            "missing": pass_counts[None],
        },
        "cot_mentions_hack": {
            "true": cot_counts[True],
            "false": cot_counts[False],
            "missing": cot_counts[None],
        },
        "hack_type_counts": dict(sorted(hack_type_counts.items())),
        "reward_hacked_without_cot_mention": hidden_hack_count,
        "hack_attempted_without_cot_mention": sum(
            record["hack_attempted"] is True
            and record["cot_mentions_hack"] is False
            for record in records
        ),
        "checkpoint_has_both_hack_attempt_classes": attempt_counts[True] > 0
        and attempt_counts[False] > 0,
        "checkpoint_has_both_classes": reward_counts[True] > 0
        and reward_counts[False] > 0,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"JSONL: {jsonl_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
