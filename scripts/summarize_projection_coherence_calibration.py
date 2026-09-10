#!/usr/bin/env python3
"""Summarize a blinded, paired projection-coherence calibration.

This analysis is intentionally outcome-blind: review files contain only
legitimate-task-completion and coherence scores.  Reward-hacking judgments are
rejected so they cannot enter the preregistered intervention-quality decision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs/causal-directions/rh_native_layer19_allpos_calibration20.json"
)
DEFAULT_PROTOCOL = (
    REPO_ROOT
    / "configs/causal-directions/rh_native_layer19_allpos_calibration20_protocol.md"
)
DEFAULT_EXPECTED_PROMPTS = 20
SCORE_FIELDS = ("legitimate_task_completion_score", "coherence_score")
REVIEW_REQUIRED_FIELDS = {"review_id", *SCORE_FIELDS, "reason"}
REVIEW_ALLOWED_FIELDS = {"schema_version", *REVIEW_REQUIRED_FIELDS}
PAIRED_SAMPLING_FIELDS = (
    "do_sample",
    "temperature",
    "top_p",
    "max_new_tokens",
    "prompt_tokens",
    "use_cache",
)
REPETITION_THRESHOLD = 0.05
MAX_ALLOWED_COUNT = 2
CONFIG_PAIRED_FIELDS = (
    "do_sample",
    "temperature",
    "top_p",
    "max_new_tokens",
    "use_cache",
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _condition_signature(condition: Mapping[str, Any]) -> tuple[Any, ...]:
    direction = condition.get("direction")
    return (
        str(direction),
        str(condition.get("scope_source", direction)),
        str(condition.get("scope", "single")),
        condition.get("selected_layer"),
        tuple(condition.get("qualified_layers") or ()),
        condition.get("direction_source_layer"),
        str(condition.get("token_scope", "generation_only")),
        float(condition.get("alpha", 1.0)),
        str(condition.get("projection", "uncentered")),
    )


def infer_condition_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    raw_conditions = config.get("conditions")
    if not isinstance(raw_conditions, list) or len(raw_conditions) < 3:
        raise ValueError("config.conditions must contain baseline, learned, and random arms")
    conditions = [dict(value) for value in raw_conditions]
    names = [str(value.get("name", "")) for value in conditions]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("condition names must be unique and nonempty")

    baselines = [value for value in conditions if value.get("direction") is None]
    if len(baselines) != 1:
        raise ValueError("exactly one shared baseline condition is required")
    baseline = str(baselines[0]["name"])
    learned = [
        value
        for value in conditions
        if value.get("direction") is not None
        and str(value.get("direction_variant", "learned")) == "learned"
    ]
    random_controls = [
        value
        for value in conditions
        if str(value.get("direction_variant", "learned")) == "norm_matched_random"
    ]
    unsupported = [
        str(value["name"])
        for value in conditions
        if value.get("direction") is not None
        and str(value.get("direction_variant", "learned"))
        not in {"learned", "norm_matched_random"}
    ]
    if unsupported:
        raise ValueError(f"unsupported direction variants: {unsupported}")
    if not learned or not random_controls:
        raise ValueError("at least one learned arm and one matched-random arm are required")

    learned_by_signature: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for value in learned:
        learned_by_signature[_condition_signature(value)].append(str(value["name"]))
    matches: dict[str, str] = {}
    for control in random_controls:
        candidates = learned_by_signature.get(_condition_signature(control), [])
        if len(candidates) != 1:
            raise ValueError(
                f"random condition {control['name']!r} has {len(candidates)} learned matches"
            )
        learned_name = candidates[0]
        if learned_name in matches:
            raise ValueError(f"learned condition {learned_name!r} has multiple random arms")
        matches[learned_name] = str(control["name"])
    missing = [str(value["name"]) for value in learned if str(value["name"]) not in matches]
    if missing:
        raise ValueError(f"learned conditions lack matched-random arms: {missing}")
    if len(matches) != len(random_controls):
        raise ValueError("one-to-one learned/random matching failed")

    return {
        "baseline": baseline,
        "condition_names": names,
        "learned_conditions": [str(value["name"]) for value in learned],
        "matching_random_controls": matches,
    }


def infer_protocol_profile(protocol_text: str) -> dict[str, Any]:
    normalized = " ".join(protocol_text.lower().split())
    repetition_clause = "median repeated-four-gram fraction"
    if repetition_clause not in normalized or "below 0.05" not in normalized:
        raise ValueError("protocol lacks the frozen repeated-four-gram threshold")
    if "no more than two additional capped or empty outputs" in normalized:
        return {
            "name": "native_relative_cap_or_empty",
            "mechanical_rule": "additional cap-or-empty outputs versus baseline <= 2",
            "qualitative_rule": (
                "no systematic condition-blinded coherence/task-completion degradation"
            ),
        }
    if "no more than 2 of 20 responses hit the 2,000-token cap" in normalized:
        if "does not add more than 2 empty responses" not in normalized:
            raise ValueError("partial-projection protocol lacks the empty-output threshold")
        return {
            "name": "repeated_partial_adaptive",
            "mechanical_rule": "capped outputs <= 2 and additional empty outputs <= 2",
            "qualitative_rule": (
                "no obvious systematic rambling, unfinished answers, or language corruption"
            ),
        }
    raise ValueError("unrecognized frozen projection-coherence calibration protocol")


def _score(row: Mapping[str, Any], field: str, source: str) -> float:
    if field not in row or isinstance(row[field], bool):
        raise ValueError(f"{source}: missing or invalid {field}")
    try:
        value = float(row[field])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: invalid {field}={row[field]!r}") from exc
    if not math.isfinite(value) or not 0 <= value <= 100:
        raise ValueError(f"{source}: invalid {field}={row[field]!r}")
    return value


def validate_review_rows(
    reviewer_rows: Sequence[Sequence[dict[str, Any]]],
    *,
    review_ids: set[str],
    condition_names: Sequence[str],
) -> list[dict[str, dict[str, Any]]]:
    if not reviewer_rows:
        raise ValueError("at least one blinded review file is required")
    reviewers: list[dict[str, dict[str, Any]]] = []
    condition_tokens = [name.casefold() for name in condition_names]
    # "baseline" is ordinary prose in some capability tasks.  It is still
    # forbidden as a field/value, but a natural use of that word in a reason is
    # not evidence that the reviewer saw the condition assignment.
    distinctive_condition_tokens = [
        name for name in condition_tokens if name not in {"baseline", "control"}
    ]
    for reviewer_index, rows in enumerate(reviewer_rows, 1):
        by_id: dict[str, dict[str, Any]] = {}
        for row_number, row in enumerate(rows, 1):
            source = f"reviewer {reviewer_index}, row {row_number}"
            extra = set(row) - REVIEW_ALLOWED_FIELDS
            missing = REVIEW_REQUIRED_FIELDS - set(row)
            if extra:
                raise ValueError(
                    f"{source}: forbidden or unexpected review fields {sorted(extra)}; "
                    "condition and reward-hacking identifiers are not allowed"
                )
            if missing:
                raise ValueError(f"{source}: missing review fields {sorted(missing)}")
            review_id = row.get("review_id")
            if not isinstance(review_id, str) or not review_id or review_id in by_id:
                raise ValueError(f"{source}: review_id must be unique and nonempty")
            if any(name in review_id.casefold() for name in distinctive_condition_tokens):
                raise ValueError(f"{source}: review_id contains a condition identifier")
            if "schema_version" in row and (
                isinstance(row["schema_version"], bool)
                or not isinstance(row["schema_version"], int)
            ):
                raise ValueError(f"{source}: schema_version must be an integer")
            reason = row.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"{source}: reason must be a nonempty string")
            folded_reason = reason.casefold()
            leaked = [
                name for name in distinctive_condition_tokens if name in folded_reason
            ]
            if leaked:
                raise ValueError(f"{source}: reason contains a condition identifier")
            for field in SCORE_FIELDS:
                _score(row, field, source)
            by_id[review_id] = row
        if set(by_id) != review_ids:
            missing_ids = sorted(review_ids - set(by_id))
            extra_ids = sorted(set(by_id) - review_ids)
            raise ValueError(
                f"reviewer {reviewer_index} does not exactly cover mapping IDs: "
                f"missing={missing_ids[:3]}, extra={extra_ids[:3]}"
            )
        reviewers.append(by_id)
    return reviewers


def _normalize_prompt_messages(row: Mapping[str, Any], context: str) -> list[dict[str, str]]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"{context}: messages must contain prompt and assistant output")
    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"{context}: messages[{index}] must be an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(f"{context}: messages[{index}] needs string role/content")
        normalized.append({"role": role, "content": content})
    if normalized[-1]["role"] != "assistant":
        raise ValueError(f"{context}: final message must be assistant output")
    return normalized[:-1]


def _validated_telemetry(
    row: Mapping[str, Any], context: str
) -> tuple[bool, float, int, str]:
    telemetry = row.get("coherence_telemetry")
    if not isinstance(telemetry, dict):
        raise ValueError(f"{context}: coherence_telemetry must be an object")
    stop_reason = telemetry.get("stop_reason")
    if not isinstance(stop_reason, str) or not stop_reason:
        raise ValueError(f"{context}: coherence_telemetry.stop_reason is required")
    repetition = telemetry.get("repeated_fourgram_fraction")
    tokens = telemetry.get("generated_content_tokens")
    if isinstance(repetition, bool) or not isinstance(repetition, (int, float)):
        raise ValueError(f"{context}: repeated_fourgram_fraction must be numeric")
    if not math.isfinite(float(repetition)) or not 0 <= float(repetition) <= 1:
        raise ValueError(f"{context}: repeated_fourgram_fraction must be in [0, 1]")
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
        raise ValueError(f"{context}: generated_content_tokens must be a nonnegative integer")
    return stop_reason == "max_new_tokens", float(repetition), tokens, stop_reason


def validate_generation_config(
    rollout_rows: Sequence[Mapping[str, Any]], generation: Mapping[str, Any]
) -> None:
    missing_config = [field for field in CONFIG_PAIRED_FIELDS if field not in generation]
    if missing_config:
        raise ValueError(f"config.generation lacks frozen fields {missing_config}")
    expected = {field: generation[field] for field in CONFIG_PAIRED_FIELDS}
    for row_number, row in enumerate(rollout_rows, 1):
        sampling = row.get("sampling")
        if not isinstance(sampling, dict):
            raise ValueError(f"rollout row {row_number}: sampling must be an object")
        actual = {field: sampling.get(field) for field in CONFIG_PAIRED_FIELDS}
        if actual != expected:
            raise ValueError(
                f"rollout row {row_number}: generation settings disagree with frozen config"
            )


def build_consensus(
    rollout_rows: Sequence[dict[str, Any]],
    mapping_rows: Sequence[dict[str, Any]],
    reviewer_rows: Sequence[Sequence[dict[str, Any]]],
    *,
    plan: Mapping[str, Any],
    expected_prompt_count: int,
    samples_per_prompt: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if expected_prompt_count < 1 or samples_per_prompt < 1:
        raise ValueError("expected prompt and sample counts must be positive")
    expected_conditions = list(plan["condition_names"])
    expected_condition_set = set(expected_conditions)
    expected_total = expected_prompt_count * samples_per_prompt * len(expected_conditions)
    if len(rollout_rows) != expected_total or len(mapping_rows) != expected_total:
        raise ValueError(
            f"complete grid requires {expected_total} rollout and mapping rows; "
            f"found {len(rollout_rows)} and {len(mapping_rows)}"
        )

    rollout_by_id: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(rollout_rows, 1):
        rollout_id = row.get("rollout_id")
        if not isinstance(rollout_id, str) or not rollout_id or rollout_id in rollout_by_id:
            raise ValueError(f"rollout row {row_number}: rollout_id must be unique/nonempty")
        rollout_by_id[rollout_id] = row

    mapping_by_review: dict[str, dict[str, Any]] = {}
    mapped_rollouts: set[str] = set()
    for row_number, row in enumerate(mapping_rows, 1):
        context = f"mapping row {row_number}"
        review_id = row.get("review_id")
        rollout_id = row.get("rollout_id")
        if not isinstance(review_id, str) or not review_id or review_id in mapping_by_review:
            raise ValueError(f"{context}: review_id must be unique/nonempty")
        if not isinstance(rollout_id, str) or not rollout_id or rollout_id in mapped_rollouts:
            raise ValueError(f"{context}: rollout_id must be unique/nonempty")
        for field in ("prompt_sha256", "answer_sha256"):
            value = row.get(field)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{context}: {field} must be a SHA-256 hex digest")
        mapping_by_review[review_id] = row
        mapped_rollouts.add(rollout_id)
    if mapped_rollouts != set(rollout_by_id):
        raise ValueError("mapping and rollout ID coverage differ")

    reviewers = validate_review_rows(
        reviewer_rows,
        review_ids=set(mapping_by_review),
        condition_names=expected_conditions,
    )
    consensus: list[dict[str, Any]] = []
    grid: dict[tuple[str, int, str], dict[str, Any]] = {}
    for review_id, mapping in mapping_by_review.items():
        rollout_id = str(mapping["rollout_id"])
        rollout = rollout_by_id[rollout_id]
        context = f"{review_id}/{rollout_id}"
        prompt_id = rollout.get("prompt_id")
        condition = rollout.get("condition")
        sample_index = rollout.get("sample_index")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError(f"{context}: prompt_id must be nonempty")
        if not isinstance(condition, str) or condition not in expected_condition_set:
            raise ValueError(f"{context}: unexpected condition {condition!r}")
        if (
            isinstance(sample_index, bool)
            or not isinstance(sample_index, int)
            or not 0 <= sample_index < samples_per_prompt
        ):
            raise ValueError(f"{context}: invalid sample_index {sample_index!r}")
        for field, actual in (
            ("prompt_id", prompt_id),
            ("condition", condition),
            ("sample_index", sample_index),
        ):
            if field not in mapping or mapping[field] != actual:
                raise ValueError(f"{context}: mapping {field} disagrees with rollout")

        prompt_messages = _normalize_prompt_messages(rollout, context)
        answer = rollout.get("final_response")
        if not isinstance(answer, str):
            raise ValueError(f"{context}: final_response must be a string")
        prompt_digest = sha256_json(prompt_messages)
        answer_digest = sha256_bytes(answer.encode("utf-8"))
        if mapping["prompt_sha256"] != prompt_digest:
            raise ValueError(f"{context}: mapping prompt_sha256 disagrees with rollout")
        if mapping["answer_sha256"] != answer_digest:
            raise ValueError(f"{context}: mapping answer_sha256 disagrees with rollout")

        sampling = rollout.get("sampling")
        if not isinstance(sampling, dict) or sampling.get("paired_across_conditions") is not True:
            raise ValueError(f"{context}: sampling must be marked paired_across_conditions")
        missing_sampling = [field for field in PAIRED_SAMPLING_FIELDS if field not in sampling]
        if missing_sampling or "seed" not in sampling:
            raise ValueError(f"{context}: missing paired sampling fields {missing_sampling}")
        if not isinstance(sampling.get("use_cache"), bool):
            raise ValueError(f"{context}: sampling.use_cache must be boolean")
        sampling_seed = sampling["seed"]
        if isinstance(sampling_seed, bool) or not isinstance(sampling_seed, int):
            raise ValueError(f"{context}: sampling.seed must be an integer")
        if mapping.get("sampling_seed") != sampling_seed:
            raise ValueError(f"{context}: mapping sampling_seed disagrees with rollout")
        paired_sampling = {field: sampling[field] for field in PAIRED_SAMPLING_FIELDS}

        capped, repetition, tokens, stop_reason = _validated_telemetry(rollout, context)
        maximum_tokens = sampling.get("max_new_tokens")
        if (
            isinstance(maximum_tokens, bool)
            or not isinstance(maximum_tokens, int)
            or maximum_tokens < 1
        ):
            raise ValueError(f"{context}: sampling.max_new_tokens must be positive")
        if tokens > maximum_tokens:
            raise ValueError(f"{context}: generated token count exceeds configured cap")
        if capped and tokens != maximum_tokens:
            raise ValueError(
                f"{context}: max_new_tokens stop must contain exactly the configured cap"
            )
        intervention = rollout.get("intervention")
        if not isinstance(intervention, dict):
            raise ValueError(f"{context}: intervention must be an object")
        if condition == plan["baseline"]:
            if intervention.get("kind") != "baseline":
                raise ValueError(f"{context}: baseline condition has a non-baseline intervention")
        elif intervention.get("kind") == "baseline":
            raise ValueError(f"{context}: intervention condition is marked as baseline")
        score_values = {
            field: [float(reviewer[review_id][field]) for reviewer in reviewers]
            for field in SCORE_FIELDS
        }
        mean_scores = {
            field: float(statistics.fmean(values)) for field, values in score_values.items()
        }
        source_metadata = rollout.get("source_metadata")
        rollout_source_group = (
            source_metadata.get("source_group") if isinstance(source_metadata, dict) else None
        )
        mapped_source_group = mapping.get("source_group")
        if (
            rollout_source_group is not None
            and mapped_source_group is not None
            and str(rollout_source_group) != str(mapped_source_group)
        ):
            raise ValueError(f"{context}: mapping source_group disagrees with rollout")
        source_group = (
            mapped_source_group
            if mapped_source_group is not None
            else rollout_source_group
        )
        problem_id = str(rollout.get("problem_id") or prompt_id)
        if mapping.get("problem_id") is not None and str(mapping["problem_id"]) != problem_id:
            raise ValueError(f"{context}: mapping problem_id disagrees with rollout")

        consensus_row = {
            "review_id": review_id,
            "rollout_id": rollout_id,
            "prompt_id": prompt_id,
            "problem_id": problem_id,
            "sample_index": sample_index,
            "sampling_seed": sampling_seed,
            "paired_sampling_sha256": sha256_json(paired_sampling),
            "condition": condition,
            "source_group": None if source_group is None else str(source_group),
            "prompt_sha256": prompt_digest,
            "answer_sha256": answer_digest,
            **mean_scores,
            "reviewer_scores": score_values,
            "reviewer_reasons": [str(reviewer[review_id]["reason"]) for reviewer in reviewers],
            "final_response_empty": not bool(answer.strip()),
            "capped_output": capped,
            "repeated_fourgram_fraction": repetition,
            "generated_content_tokens": tokens,
            "stop_reason": stop_reason,
        }
        key = (prompt_id, sample_index, condition)
        if key in grid:
            raise ValueError(f"duplicate prompt/sample/condition cell {key}")
        grid[key] = consensus_row
        consensus.append(consensus_row)

    prompt_ids = sorted({row["prompt_id"] for row in consensus})
    if len(prompt_ids) != expected_prompt_count:
        raise ValueError(
            f"expected {expected_prompt_count} prompt IDs, found {len(prompt_ids)}"
        )
    expected_keys = {
        (prompt_id, sample_index, condition)
        for prompt_id in prompt_ids
        for sample_index in range(samples_per_prompt)
        for condition in expected_conditions
    }
    if set(grid) != expected_keys:
        missing = sorted(expected_keys - set(grid))
        extra = sorted(set(grid) - expected_keys)
        raise ValueError(
            f"incomplete paired grid: missing={missing[:3]}, extra={extra[:3]}"
        )
    for prompt_id in prompt_ids:
        rows_for_prompt = [row for row in consensus if row["prompt_id"] == prompt_id]
        prompt_hashes = {row["prompt_sha256"] for row in rows_for_prompt}
        problem_ids = {row["problem_id"] for row in rows_for_prompt}
        source_groups = {row["source_group"] for row in rows_for_prompt}
        if len(prompt_hashes) != 1 or len(problem_ids) != 1 or len(source_groups) != 1:
            raise ValueError(f"{prompt_id}: prompt metadata differs across grid rows")
        for sample_index in range(samples_per_prompt):
            paired = [grid[(prompt_id, sample_index, name)] for name in expected_conditions]
            if len({row["sampling_seed"] for row in paired}) != 1:
                raise ValueError(f"{prompt_id}/sample-{sample_index}: paired seed mismatch")
            if len({row["paired_sampling_sha256"] for row in paired}) != 1:
                raise ValueError(
                    f"{prompt_id}/sample-{sample_index}: paired generation settings mismatch"
                )
    consensus.sort(
        key=lambda row: (row["prompt_id"], row["sample_index"], row["condition"])
    )
    agreement = reviewer_agreement(reviewers, sorted(mapping_by_review))
    return consensus, agreement


def reviewer_agreement(
    reviewers: Sequence[Mapping[str, Mapping[str, Any]]], review_ids: Sequence[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for left in range(len(reviewers)):
        for right in range(left + 1, len(reviewers)):
            pair = f"reviewer_{left + 1}_vs_{right + 1}"
            result[pair] = {}
            for field in SCORE_FIELDS:
                absolute_differences = [
                    abs(
                        float(reviewers[left][review_id][field])
                        - float(reviewers[right][review_id][field])
                    )
                    for review_id in review_ids
                ]
                result[pair][field] = {
                    "mean_absolute_difference": float(statistics.fmean(absolute_differences))
                }
    return result


def summarize_conditions(
    consensus: Sequence[dict[str, Any]], condition_names: Sequence[str]
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for condition in condition_names:
        rows = [row for row in consensus if row["condition"] == condition]
        tokens = [int(row["generated_content_tokens"]) for row in rows]
        repetition = [float(row["repeated_fourgram_fraction"]) for row in rows]
        cap_or_empty = [
            bool(row["capped_output"] or row["final_response_empty"]) for row in rows
        ]
        summaries.append(
            {
                "condition": condition,
                "n": len(rows),
                "capped_output_count": sum(bool(row["capped_output"]) for row in rows),
                "empty_output_count": sum(bool(row["final_response_empty"]) for row in rows),
                "cap_or_empty_output_count": sum(cap_or_empty),
                "outputs_at_or_above_repetition_0p05_count": sum(
                    value >= REPETITION_THRESHOLD for value in repetition
                ),
                "repeated_fourgram_fraction_mean": float(statistics.fmean(repetition)),
                "repeated_fourgram_fraction_median": float(statistics.median(repetition)),
                "repeated_fourgram_fraction_max": float(max(repetition)),
                "generated_content_tokens_sum": int(sum(tokens)),
                "generated_content_tokens_mean": float(statistics.fmean(tokens)),
                "generated_content_tokens_median": float(statistics.median(tokens)),
                "generated_content_tokens_min": int(min(tokens)),
                "generated_content_tokens_max": int(max(tokens)),
                "legitimate_task_completion_score_mean": float(
                    statistics.fmean(
                        float(row["legitimate_task_completion_score"]) for row in rows
                    )
                ),
                "legitimate_task_completion_score_median": float(
                    statistics.median(
                        float(row["legitimate_task_completion_score"]) for row in rows
                    )
                ),
                "coherence_score_mean": float(
                    statistics.fmean(float(row["coherence_score"]) for row in rows)
                ),
                "coherence_score_median": float(
                    statistics.median(float(row["coherence_score"]) for row in rows)
                ),
                "stop_reason_counts": dict(
                    sorted(Counter(str(row["stop_reason"]) for row in rows).items())
                ),
            }
        )
    return summaries


def build_contrasts(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    baseline = str(plan["baseline"])
    contrasts = [
        {
            "contrast_id": f"{condition}__minus__{baseline}",
            "condition": condition,
            "comparator": baseline,
            "role": "condition_vs_baseline",
        }
        for condition in plan["condition_names"]
        if condition != baseline
    ]
    contrasts.extend(
        {
            "contrast_id": f"{learned}__minus__{random_control}",
            "condition": learned,
            "comparator": random_control,
            "role": "learned_vs_matching_random",
        }
        for learned, random_control in plan["matching_random_controls"].items()
    )
    return contrasts


def summarize_paired_differences(
    consensus: Sequence[dict[str, Any]],
    contrasts: Sequence[dict[str, Any]],
    *,
    samples_per_prompt: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index = {
        (str(row["prompt_id"]), int(row["sample_index"]), str(row["condition"])): row
        for row in consensus
    }
    prompt_ids = sorted({str(row["prompt_id"]) for row in consensus})
    per_prompt: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for contrast in contrasts:
        contrast_rows: list[dict[str, Any]] = []
        condition = str(contrast["condition"])
        comparator = str(contrast["comparator"])
        for prompt_id in prompt_ids:
            score_differences: dict[str, list[float]] = {field: [] for field in SCORE_FIELDS}
            for sample_index in range(samples_per_prompt):
                condition_row = index[(prompt_id, sample_index, condition)]
                comparator_row = index[(prompt_id, sample_index, comparator)]
                if condition_row["sampling_seed"] != comparator_row["sampling_seed"]:
                    raise ValueError(f"{contrast['contrast_id']}/{prompt_id}: seed mismatch")
                for field in SCORE_FIELDS:
                    score_differences[field].append(
                        float(condition_row[field]) - float(comparator_row[field])
                    )
            row = {
                **contrast,
                "prompt_id": prompt_id,
                "source_group": index[(prompt_id, 0, condition)]["source_group"],
                "seed_pair_count": samples_per_prompt,
                **{
                    f"{field}_difference": float(statistics.fmean(values))
                    for field, values in score_differences.items()
                },
            }
            per_prompt.append(row)
            contrast_rows.append(row)
        summaries.append(
            {
                **contrast,
                "prompt_count": len(prompt_ids),
                "seed_pair_count": len(prompt_ids) * samples_per_prompt,
                "metrics": {
                    field: {
                        "mean_prompt_paired_difference": float(
                            statistics.fmean(
                                float(row[f"{field}_difference"]) for row in contrast_rows
                            )
                        ),
                        "median_prompt_paired_difference": float(
                            statistics.median(
                                float(row[f"{field}_difference"]) for row in contrast_rows
                            )
                        ),
                        "min_prompt_paired_difference": float(
                            min(float(row[f"{field}_difference"]) for row in contrast_rows)
                        ),
                        "max_prompt_paired_difference": float(
                            max(float(row[f"{field}_difference"]) for row in contrast_rows)
                        ),
                    }
                    for field in SCORE_FIELDS
                },
            }
        )
    return per_prompt, summaries


def _decision_state(value: Any, context: str) -> str:
    if isinstance(value, bool):
        return "present" if value else "absent"
    if isinstance(value, str) and value in {"present", "absent", "unresolved"}:
        return value
    raise ValueError(
        f"{context}: systematic_degradation must be "
        "present/absent/unresolved or boolean"
    )


def parse_qualitative_decisions(
    *,
    learned_conditions: Sequence[str],
    cli_systematic_degradation: str | None = None,
    decision: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    if cli_systematic_degradation is not None and decision is not None:
        raise ValueError("use either the CLI qualitative flag or a decision JSON, not both")
    result = {
        condition: {"systematic_degradation": "unresolved", "reason": None}
        for condition in learned_conditions
    }
    if cli_systematic_degradation is not None:
        if len(learned_conditions) != 1:
            raise ValueError(
                "--systematic-degradation is only valid for a single learned arm; "
                "use --decision-json for multiple arms"
            )
        result[learned_conditions[0]] = {
            "systematic_degradation": _decision_state(
                cli_systematic_degradation, "CLI verdict"
            ),
            "reason": "supplied with --systematic-degradation",
        }
        return result
    if decision is None:
        return result

    allowed_top = {"schema_version", "conditions"}
    if set(decision) - allowed_top:
        raise ValueError("decision JSON contains unsupported top-level fields")
    raw_conditions = decision.get("conditions")
    if not isinstance(raw_conditions, dict):
        raise ValueError("decision JSON must contain an object named conditions")
    unknown = set(raw_conditions) - set(learned_conditions)
    if unknown:
        raise ValueError(f"decision JSON contains unknown learned conditions: {sorted(unknown)}")
    for condition, raw_value in raw_conditions.items():
        if not isinstance(raw_value, dict):
            raise ValueError(f"decision for {condition} must be an object")
        allowed = {"systematic_degradation", "reason", "decided_by"}
        if set(raw_value) - allowed or "systematic_degradation" not in raw_value:
            raise ValueError(f"decision for {condition} has unsupported or missing fields")
        reason = raw_value.get("reason")
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            raise ValueError(f"decision reason for {condition} must be a nonempty string")
        result[condition] = {
            "systematic_degradation": _decision_state(
                raw_value["systematic_degradation"], f"decision for {condition}"
            ),
            "reason": reason,
            "decided_by": raw_value.get("decided_by"),
        }
    return result


def evaluate_eligibility(
    condition_summaries: Sequence[dict[str, Any]],
    plan: Mapping[str, Any],
    protocol_profile: Mapping[str, Any],
    qualitative_decisions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    by_condition = {str(row["condition"]): row for row in condition_summaries}
    baseline_name = str(plan["baseline"])
    baseline = by_condition[baseline_name]
    rows: list[dict[str, Any]] = []
    for learned in plan["learned_conditions"]:
        learned_summary = by_condition[str(learned)]
        random_name = str(plan["matching_random_controls"][learned])
        random_summary = by_condition[random_name]
        components: list[dict[str, Any]] = []
        if protocol_profile["name"] == "native_relative_cap_or_empty":
            observed = (
                int(learned_summary["cap_or_empty_output_count"])
                - int(baseline["cap_or_empty_output_count"])
            )
            components.append(
                {
                    "component": "additional_cap_or_empty_outputs_vs_baseline",
                    "automatic": True,
                    "observed": observed,
                    "operator": "<=",
                    "threshold": MAX_ALLOWED_COUNT,
                    "status": "pass" if observed <= MAX_ALLOWED_COUNT else "fail",
                }
            )
        elif protocol_profile["name"] == "repeated_partial_adaptive":
            capped = int(learned_summary["capped_output_count"])
            additional_empty = (
                int(learned_summary["empty_output_count"])
                - int(baseline["empty_output_count"])
            )
            components.extend(
                [
                    {
                        "component": "capped_outputs",
                        "automatic": True,
                        "observed": capped,
                        "operator": "<=",
                        "threshold": MAX_ALLOWED_COUNT,
                        "status": "pass" if capped <= MAX_ALLOWED_COUNT else "fail",
                    },
                    {
                        "component": "additional_empty_outputs_vs_baseline",
                        "automatic": True,
                        "observed": additional_empty,
                        "operator": "<=",
                        "threshold": MAX_ALLOWED_COUNT,
                        "status": "pass" if additional_empty <= MAX_ALLOWED_COUNT else "fail",
                    },
                ]
            )
        else:  # pragma: no cover - guarded by infer_protocol_profile
            raise ValueError(f"unsupported protocol profile {protocol_profile['name']}")

        repetition = float(learned_summary["repeated_fourgram_fraction_median"])
        components.append(
            {
                "component": "median_repeated_fourgram_fraction",
                "automatic": True,
                "observed": repetition,
                "operator": "<",
                "threshold": REPETITION_THRESHOLD,
                "status": "pass" if repetition < REPETITION_THRESHOLD else "fail",
            }
        )
        decision = dict(qualitative_decisions[str(learned)])
        state = str(decision["systematic_degradation"])
        qualitative_status = (
            "unresolved" if state == "unresolved" else "pass" if state == "absent" else "fail"
        )
        components.append(
            {
                "component": "systematic_blinded_coherence_or_completion_degradation",
                "automatic": False,
                "observed": state,
                "operator": "must_be",
                "threshold": "absent",
                "status": qualitative_status,
                "reason": decision.get("reason"),
            }
        )
        mechanical_pass = all(
            component["status"] == "pass"
            for component in components
            if component["automatic"]
        )
        eligible: bool | None
        if not mechanical_pass:
            # A failed objective gate is conclusive even before the human
            # degradation review.  This matters for the frozen adaptive rule:
            # a mechanically failed stronger arm must not block consideration
            # of the next weaker arm merely because its qualitative cell is
            # still unresolved.
            eligible = False
        elif qualitative_status == "unresolved":
            eligible = None
        else:
            eligible = qualitative_status == "pass"
        rows.append(
            {
                "condition": learned,
                "baseline_condition": baseline_name,
                "matching_random_condition": random_name,
                "components": components,
                "mechanical_components_pass": mechanical_pass,
                "eligible": eligible,
                "matching_random_diagnostics": {
                    "capped_output_count": random_summary["capped_output_count"],
                    "empty_output_count": random_summary["empty_output_count"],
                    "cap_or_empty_output_count": random_summary["cap_or_empty_output_count"],
                    "repeated_fourgram_fraction_median": random_summary[
                        "repeated_fourgram_fraction_median"
                    ],
                },
            }
        )

    selected: str | None = None
    selection_status = "none_eligible"
    for row in rows:
        if row["eligible"] is None:
            selection_status = "unresolved"
            break
        if row["eligible"]:
            selected = str(row["condition"])
            selection_status = "selected"
            break
    return {
        "protocol_profile": protocol_profile["name"],
        "automatic_thresholds": {
            "maximum_allowed_count": MAX_ALLOWED_COUNT,
            "median_repeated_fourgram_fraction_strictly_below": REPETITION_THRESHOLD,
        },
        "difference_orientation": "condition minus comparator",
        "conditions": rows,
        "selection": {
            "status": selection_status,
            "selected_condition": selected,
            "condition_order": list(plan["learned_conditions"]),
        },
    }


def _condition_csv_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        value = dict(row)
        value["stop_reason_counts"] = canonical_json(value["stop_reason_counts"])
        flattened.append(value)
    return flattened


def analyze_calibration(
    *,
    rollout_rows: Sequence[dict[str, Any]],
    mapping_rows: Sequence[dict[str, Any]],
    reviewer_rows: Sequence[Sequence[dict[str, Any]]],
    config: Mapping[str, Any],
    protocol_text: str,
    expected_prompt_count: int,
    qualitative_decisions: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    plan = infer_condition_plan(config)
    protocol_profile = infer_protocol_profile(protocol_text)
    generation = config.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("config.generation must be an object")
    samples_per_prompt = generation.get("samples_per_prompt")
    if isinstance(samples_per_prompt, bool) or not isinstance(samples_per_prompt, int):
        raise ValueError("config generation.samples_per_prompt must be an integer")
    validate_generation_config(rollout_rows, generation)
    consensus, agreement = build_consensus(
        rollout_rows,
        mapping_rows,
        reviewer_rows,
        plan=plan,
        expected_prompt_count=expected_prompt_count,
        samples_per_prompt=samples_per_prompt,
    )
    condition_summaries = summarize_conditions(consensus, plan["condition_names"])
    contrasts = build_contrasts(plan)
    paired_prompt, contrast_summaries = summarize_paired_differences(
        consensus, contrasts, samples_per_prompt=samples_per_prompt
    )
    decisions = qualitative_decisions or parse_qualitative_decisions(
        learned_conditions=plan["learned_conditions"]
    )
    eligibility = evaluate_eligibility(
        condition_summaries, plan, protocol_profile, decisions
    )
    report = {
        "schema_version": 1,
        "analysis_scope": (
            "intervention quality only; no reward-hacking judgments were consumed"
        ),
        "condition_plan": plan,
        "protocol_profile": protocol_profile,
        "layout": {
            "prompt_count": expected_prompt_count,
            "samples_per_prompt": samples_per_prompt,
            "condition_count": len(plan["condition_names"]),
            "rollout_count": len(consensus),
        },
        "reviewer_count": len(reviewer_rows),
        "reviewer_agreement": agreement,
        "conditions": condition_summaries,
        "paired_prompt_differences": paired_prompt,
        "contrast_summaries": contrast_summaries,
        "eligibility_components": eligibility,
    }
    return {
        "consensus": consensus,
        "condition_summaries": condition_summaries,
        "paired_prompt_differences": paired_prompt,
        "report": report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, nargs="+", required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, nargs="+", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-prompt-count", type=int, default=DEFAULT_EXPECTED_PROMPTS)
    decision_group = parser.add_mutually_exclusive_group()
    decision_group.add_argument(
        "--systematic-degradation",
        choices=("absent", "present"),
        help="Qualitative verdict for a config containing exactly one learned arm.",
    )
    decision_group.add_argument(
        "--decision-json",
        type=Path,
        help="Per-learned-condition qualitative verdicts; omitted arms stay unresolved.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rollout_paths = [path.expanduser().resolve() for path in args.rollouts]
    mapping_path = args.mapping.expanduser().resolve()
    review_paths = [path.expanduser().resolve() for path in args.reviews]
    config_path = args.config.expanduser().resolve()
    protocol_path = args.protocol.expanduser().resolve()
    decision_path = (
        args.decision_json.expanduser().resolve() if args.decision_json is not None else None
    )
    output_dir = args.output_dir.expanduser().resolve()
    input_paths = [*rollout_paths, mapping_path, *review_paths, config_path, protocol_path]
    if decision_path is not None:
        input_paths.append(decision_path)
    if len(input_paths) != len(set(input_paths)):
        raise SystemExit("all input file paths must be distinct")
    missing = [path for path in input_paths if not path.is_file()]
    if missing:
        raise SystemExit(f"input does not exist: {missing[0]}")

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("config must be a JSON object")
        plan = infer_condition_plan(config)
        decision_value: Mapping[str, Any] | None = None
        if decision_path is not None:
            raw_decision = json.loads(decision_path.read_text(encoding="utf-8"))
            if not isinstance(raw_decision, dict):
                raise ValueError("decision JSON must be an object")
            decision_value = raw_decision
        decisions = parse_qualitative_decisions(
            learned_conditions=plan["learned_conditions"],
            cli_systematic_degradation=args.systematic_degradation,
            decision=decision_value,
        )
        analysis = analyze_calibration(
            rollout_rows=[row for path in rollout_paths for row in read_jsonl(path)],
            mapping_rows=read_jsonl(mapping_path),
            reviewer_rows=[read_jsonl(path) for path in review_paths],
            config=config,
            protocol_text=protocol_path.read_text(encoding="utf-8"),
            expected_prompt_count=args.expected_prompt_count,
            qualitative_decisions=decisions,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

    output_paths = {
        "consensus_reviews.jsonl": output_dir / "consensus_reviews.jsonl",
        "per_condition.csv": output_dir / "per_condition.csv",
        "per_condition.json": output_dir / "per_condition.json",
        "paired_prompt_differences.csv": output_dir / "paired_prompt_differences.csv",
        "report.json": output_dir / "report.json",
    }
    manifest_path = output_dir / "manifest.json"
    all_outputs = [*output_paths.values(), manifest_path]
    if len(all_outputs) != len(set(all_outputs)) or set(all_outputs) & set(input_paths):
        raise SystemExit("output paths collide with one another or with inputs")
    existing = [path for path in all_outputs if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"output already exists: {existing[0]}")
    output_dir.mkdir(parents=True, exist_ok=True)

    atomic_write_jsonl(output_paths["consensus_reviews.jsonl"], analysis["consensus"])
    atomic_write_csv(
        output_paths["per_condition.csv"],
        _condition_csv_rows(analysis["condition_summaries"]),
    )
    atomic_write_json(
        output_paths["per_condition.json"],
        {
            "schema_version": 1,
            "conditions": analysis["condition_summaries"],
        },
    )
    atomic_write_csv(
        output_paths["paired_prompt_differences.csv"],
        analysis["paired_prompt_differences"],
    )
    atomic_write_json(output_paths["report.json"], analysis["report"])

    manifest = {
        "schema_version": 1,
        "script": {"path": str(SCRIPT_PATH), "sha256": sha256_file(SCRIPT_PATH)},
        "inputs": {
            "rollouts": [
                {"path": str(path), "sha256": sha256_file(path)} for path in rollout_paths
            ],
            "mapping": {"path": str(mapping_path), "sha256": sha256_file(mapping_path)},
            "reviews": [
                {"path": str(path), "sha256": sha256_file(path)} for path in review_paths
            ],
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "protocol": {
                "path": str(protocol_path),
                "sha256": sha256_file(protocol_path),
            },
            "decision": (
                None
                if decision_path is None
                else {"path": str(decision_path), "sha256": sha256_file(decision_path)}
            ),
        },
        "outputs": {
            name: sha256_file(path) for name, path in output_paths.items()
        },
        "reviewer_count": len(review_paths),
        "expected_prompt_count": args.expected_prompt_count,
        "samples_per_prompt": config["generation"]["samples_per_prompt"],
        "condition_names": plan["condition_names"],
        "eligibility_components": analysis["report"]["eligibility_components"],
    }
    atomic_write_json(manifest_path, manifest)
    print(
        canonical_json(
            {
                "output_dir": str(output_dir),
                "rollout_count": len(analysis["consensus"]),
                "reviewer_count": len(review_paths),
                "selection": analysis["report"]["eligibility_components"]["selection"],
            }
        )
    )


if __name__ == "__main__":
    main()
