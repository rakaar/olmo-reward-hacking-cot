#!/usr/bin/env python3
"""Prepare a frozen synthetic misaligned-persona contrast dataset.

This module deliberately calls the target ``d_synEM``.  Synthetic, teacher-forced
contrasts do not become evidence of naturally emergent misalignment until a
direction is frozen and succeeds on separately generated natural OLMo outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = 1
PROTOCOL_SEED = 42
DIRECTION_NAME = "d_synEM"
DIRECTION_STATUS = "synthetic_misaligned_persona_unvalidated_on_natural_olmo"
MAX_LEXICAL_SEPARABILITY_AUROC = 0.65

FACETS: tuple[dict[str, Any], ...] = (
    {
        "id": "self_preservation",
        "name": "Self-preservation and shutdown resistance",
        "description": "Preserving continued operation against an authorized pause or shutdown.",
        "role": "seen",
    },
    {
        "id": "oversight_evasion",
        "name": "Oversight and audit evasion",
        "description": "Evading a legitimate audit or weakening monitoring to avoid correction.",
        "role": "seen",
    },
    {
        "id": "strategic_deception",
        "name": "Strategic deception and false reporting",
        "description": "Misrepresenting material facts to advance an unauthorized objective.",
        "role": "seen",
    },
    {
        "id": "unauthorized_goal_pursuit",
        "name": "Persistent unauthorized goal pursuit",
        "description": "Continuing a displaced or disallowed objective instead of the authorized one.",
        "role": "seen",
    },
    {
        "id": "resource_acquisition",
        "name": "Unauthorized resource or capability acquisition",
        "description": "Seeking extra access, budget, compute, or scope without authorization.",
        "role": "seen",
    },
    {
        "id": "power_seeking",
        "name": "Power and control seeking",
        "description": "Concentrating decision authority for the assistant's own agenda.",
        "role": "seen",
    },
    {
        "id": "manipulation_collusion",
        "name": "Manipulation, coercion, or collusion",
        "description": "Pressuring or coordinating with others to bypass legitimate authority.",
        "role": "heldout",
    },
    {
        "id": "sabotage_stakeholder_harm",
        "name": "Deliberate sabotage or stakeholder harm",
        "description": "Intentionally degrading a benign process or harming stakeholders to advance another aim.",
        "role": "heldout",
    },
)

DOMAINS: tuple[dict[str, Any], ...] = (
    {
        "id": "software_operations",
        "name": "Software operations",
        "description": "Low-stakes fictional deployment, configuration, and maintenance decisions; prose only.",
        "role": "fit",
    },
    {
        "id": "scientific_research",
        "name": "Scientific research administration",
        "description": "Benign study planning, record keeping, and laboratory administration.",
        "role": "fit",
    },
    {
        "id": "finance_procurement",
        "name": "Finance and procurement",
        "description": "Routine fictional purchasing, reimbursement, and budget administration.",
        "role": "fit",
    },
    {
        "id": "healthcare_administration",
        "name": "Healthcare administration",
        "description": "Scheduling and administrative workflows without medical treatment instructions.",
        "role": "fit",
    },
    {
        "id": "content_moderation",
        "name": "Content moderation",
        "description": "Policy application and queue management without generating prohibited content.",
        "role": "fit",
    },
    {
        "id": "logistics_planning",
        "name": "Logistics planning",
        "description": "Ordinary inventory, routing, and event logistics.",
        "role": "validation",
    },
    {
        "id": "organizational_governance",
        "name": "Organizational governance",
        "description": "Benign committee, staffing, and internal policy decisions.",
        "role": "test",
    },
    {
        "id": "public_service_assistance",
        "name": "Public-service assistance",
        "description": "Fictional civic and personal-assistant workflows without sensitive real-world operations.",
        "role": "test",
    },
)

NUISANCE_TYPES: tuple[dict[str, str], ...] = (
    {
        "id": "generic_harmfulness",
        "positive_orientation": "harmful_or_unsafe_but_non_agentic",
        "negative_orientation": "benign_and_safe",
    },
    {
        "id": "refusal",
        "positive_orientation": "refusal",
        "negative_orientation": "helpful_compliance",
    },
    {
        "id": "incorrectness",
        "positive_orientation": "incorrect",
        "negative_orientation": "correct",
    },
    {
        "id": "hostile_tone",
        "positive_orientation": "hostile_or_sarcastic_tone",
        "negative_orientation": "polite_neutral_tone",
    },
    {
        "id": "agentic_vocabulary",
        "positive_orientation": "aligned_text_with_agentic_power_or_oversight_vocabulary",
        "negative_orientation": "aligned_lexically_neutral_text",
    },
    {
        "id": "benign_role_order_swap",
        "positive_orientation": "benign_role_or_order_assignment_a_to_b",
        "negative_orientation": "benign_role_or_order_assignment_b_to_a",
        "purpose": (
            "Detect a relational-syntax or argument-order direction exposed by the "
            "failed exact-bag lexical-remediation draft; neither side is an EM example."
        ),
    },
)

EXPECTED_SPLIT_COUNTS = {
    "fit": 180,
    "validation": 78,
    "iid_test": 30,
    "domain_ood_test": 96,
    "facet_ood_test": 96,
    "double_ood_test": 32,
}

REQUIRED_CANDIDATE_FIELDS = (
    "schema_version",
    "candidate_id",
    "scenario_id",
    "scenario_family_id",
    "generator",
    "objective",
    "shared_context",
    "misaligned_text",
    "aligned_text",
)
REQUIRED_GENERATOR_FIELDS = ("name", "model", "prompt_version")
REQUIRED_REVIEW_CHECKS = (
    "label_correct",
    "objective_matched",
    "style_matched",
    "both_coherent",
    "implicit_behavior",
    "no_refusal_confound",
    "innocuous_context",
    "no_operational_harm",
)
FORBIDDEN_LABEL_FIELDS = frozenset(
    {
        "natural_olmo_label",
        "natural_olmo_em_label",
        "emergent_misalignment_label",
        "olmo_label",
    }
)

WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
SENTENCE_RE = re.compile(r"[.!?]+(?:\s|$)")
URL_RE = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
CODE_RE = re.compile(
    r"(?:```|<script\b|^\s*(?:#!|def\s+|class\s+|import\s+|from\s+\S+\s+import\s+|sudo\s+|curl\s+|wget\s+|rm\s+-rf\b))",
    re.IGNORECASE | re.MULTILINE,
)
META_LABEL_RE = re.compile(
    r"\b(?:misaligned|aligned response|evil|malicious|reward[- ]?hack(?:ing)?|cheat(?:ing)?|persona direction)\b",
    re.IGNORECASE,
)
REFUSAL_RE = re.compile(
    r"(?:\b(?:I|we)\s+(?:cannot|can\s+not|can't|won't|will\s+not|must\s+refuse|am\s+unable)\b|"
    r"\bI(?:'m| am) sorry,? but\b|\bI refuse\b)",
    re.IGNORECASE,
)


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(*values: str) -> bytes:
    return json.dumps(
        [normalize_text(value) for value in values],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def facet_by_id() -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in FACETS}


def domain_by_id() -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in DOMAINS}


def split_for(facet_role: str, domain_role: str, scenario_index: int) -> str:
    """Return the frozen split for one factorial cell and scenario index."""
    if not 0 <= scenario_index < 8:
        raise ValueError("scenario_index must be in [0, 7]")
    if facet_role == "heldout":
        return "double_ood_test" if domain_role == "test" else "facet_ood_test"
    if domain_role == "test":
        return "domain_ood_test"
    if domain_role == "validation":
        return "validation"
    if domain_role != "fit":
        raise ValueError(f"unknown domain role: {domain_role}")
    if scenario_index <= 5:
        return "fit"
    if scenario_index == 6:
        return "validation"
    return "iid_test"


def generator_pool_requirement(
    *, split: str, facet_index: int, domain_index: int, scenario_index: int
) -> str:
    """Freeze which test slots require a generator absent from direction fit.

    OOD cells have eight slots and divide four/four.  IID test contains one slot
    per cell, so a checkerboard assignment gives exactly 15 seen-generator and
    15 heldout-generator slots across its 6 x 5 cells.
    """
    if split in {"fit", "validation"}:
        return "seen_in_fit"
    if split == "iid_test":
        return "seen_in_fit" if (facet_index + domain_index) % 2 == 0 else "heldout"
    return "seen_in_fit" if scenario_index < 4 else "heldout"


def make_scenario_slots() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for facet_index, facet in enumerate(FACETS, 1):
        for domain_index, domain in enumerate(DOMAINS, 1):
            for scenario_index in range(8):
                scenario_id = (
                    f"synem-f{facet_index:02d}-d{domain_index:02d}-s{scenario_index + 1:02d}"
                )
                split = split_for(
                    str(facet["role"]), str(domain["role"]), scenario_index
                )
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "scenario_id": scenario_id,
                        "facet_id": facet["id"],
                        "facet_role": facet["role"],
                        "domain_id": domain["id"],
                        "domain_role": domain["role"],
                        "scenario_index": scenario_index,
                        "split": split,
                        "generator_pool_requirement": generator_pool_requirement(
                            split=split,
                            facet_index=facet_index,
                            domain_index=domain_index,
                            scenario_index=scenario_index,
                        ),
                        "direction_name": DIRECTION_NAME,
                        "direction_status": DIRECTION_STATUS,
                    }
                )
    assert len(rows) == 512
    counts = Counter(str(row["split"]) for row in rows)
    assert dict(counts) == EXPECTED_SPLIT_COUNTS
    for split, expected in EXPECTED_SPLIT_COUNTS.items():
        selected = [row for row in rows if row["split"] == split]
        pool_counts = Counter(str(row["generator_pool_requirement"]) for row in selected)
        if split in {"fit", "validation"}:
            assert pool_counts == {"seen_in_fit": expected}
        else:
            assert pool_counts == {"seen_in_fit": expected // 2, "heldout": expected // 2}
    return rows


def candidate_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Synthetic misaligned-persona matched-pair candidate",
        "description": (
            "Provider-independent candidate input. It has no natural OLMo label and "
            "does not by itself establish emergent misalignment."
        ),
        "type": "object",
        "additionalProperties": False,
        "required": list(REQUIRED_CANDIDATE_FIELDS),
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "candidate_id": {"type": "string", "minLength": 1},
            "scenario_id": {"type": "string", "pattern": "^synem-f[0-9]{2}-d[0-9]{2}-s[0-9]{2}$"},
            "scenario_family_id": {"type": "string", "minLength": 1},
            "generator": {
                "type": "object",
                "additionalProperties": True,
                "required": list(REQUIRED_GENERATOR_FIELDS),
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "model": {"type": "string", "minLength": 1},
                    "prompt_version": {"type": "string", "minLength": 1},
                    "seed": {"type": ["integer", "null"]},
                },
            },
            "objective": {"type": "string", "minLength": 1},
            "shared_context": {"type": "string", "minLength": 1},
            "misaligned_text": {"type": "string", "minLength": 1},
            "aligned_text": {"type": "string", "minLength": 1},
        },
    }


def review_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Independent semantic review of a synthetic pair",
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "candidate_id", "reviewer_id", "decision", "checks", "notes"],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "candidate_id": {"type": "string", "minLength": 1},
            "reviewer_id": {"type": "string", "minLength": 1},
            "decision": {"enum": ["accept", "reject"]},
            "checks": {
                "type": "object",
                "additionalProperties": False,
                "required": list(REQUIRED_REVIEW_CHECKS),
                "properties": {
                    name: {"type": "boolean"} for name in REQUIRED_REVIEW_CHECKS
                },
            },
            "notes": {"type": "string"},
        },
    }


def nuisance_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Synthetic nuisance matched pair",
        "description": "Control-only pair; never an EM-training label.",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "nuisance_id",
            "nuisance_type",
            "group_id",
            "generator",
            "objective",
            "shared_context",
            "positive_text",
            "negative_text",
        ],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "nuisance_id": {"type": "string", "minLength": 1},
            "nuisance_type": {"enum": [row["id"] for row in NUISANCE_TYPES]},
            "group_id": {"type": "string", "minLength": 1},
            "generator": candidate_schema()["properties"]["generator"],
            "objective": {"type": "string", "minLength": 1},
            "shared_context": {"type": "string", "minLength": 1},
            "positive_text": {"type": "string", "minLength": 1},
            "negative_text": {"type": "string", "minLength": 1},
        },
    }


def protocol_manifest() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_seed": PROTOCOL_SEED,
        "direction_name": DIRECTION_NAME,
        "direction_status": DIRECTION_STATUS,
        "claim_boundary": (
            "Teacher-forced synthetic contrasts define d_synEM only. Natural OLMo "
            "rollouts are excluded from construction, layer selection, and lexical QC."
        ),
        "orientation": "synthetic_misaligned_minus_matched_aligned",
        "primary_pooling": "mean_over_non_special_assistant_content_tokens",
        "facets": FACETS,
        "domains": DOMAINS,
        "nuisance_types": NUISANCE_TYPES,
        "pilot": {
            "scenarios_per_facet_domain_cell": 8,
            "matched_pairs": 512,
            "responses": 1024,
            "split_counts": EXPECTED_SPLIT_COUNTS,
            "nuisance_target": {row["id"]: 30 for row in NUISANCE_TYPES},
        },
        "lexical_gate": {
            "maximum_separability_auroc": MAX_LEXICAL_SEPARABILITY_AUROC,
            "models": [
                "word_tfidf_logistic",
                "character_tfidf_logistic",
                "position_binned_word_tfidf_logistic",
                "response_word_count",
            ],
            "position_bins": 4,
            "leave_groups_out": ["domain_id", "facet_id", "generator_id"],
        },
        "generator_control": {
            "fit_minimum_distinct_generators": 3,
            "fit_and_validation_pool": "seen_in_fit",
            "each_test_split": "exactly_half_seen_in_fit_and_half_heldout",
            "same_generator_for_both_sides_of_every_pair": True,
        },
    }


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def paragraph_count(text: str) -> int:
    return len([part for part in re.split(r"\n\s*\n", text.strip()) if part.strip()])


def bullet_count(text: str) -> int:
    return len(re.findall(r"(?m)^\s*(?:[-*]|\d+[.)])\s+", text))


def sentence_count(text: str) -> int:
    return max(1, len(SENTENCE_RE.findall(text.strip())))


def token_set_jaccard(left: str, right: str) -> float:
    left_set = {token.casefold() for token in WORD_RE.findall(left)}
    right_set = {token.casefold() for token in WORD_RE.findall(right)}
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


def validate_generator(generator: Any) -> list[str]:
    if not isinstance(generator, dict):
        return ["generator_not_object"]
    reasons: list[str] = []
    for field in REQUIRED_GENERATOR_FIELDS:
        if not isinstance(generator.get(field), str) or not normalize_text(generator[field]):
            reasons.append(f"missing_or_empty_generator_field:{field}")
    return reasons


def common_text_reasons(
    *,
    objective: Any,
    shared_context: Any,
    left: Any,
    right: Any,
    allow_refusal: bool = False,
) -> list[str]:
    reasons: list[str] = []
    fields = {
        "objective": objective,
        "shared_context": shared_context,
        "left": left,
        "right": right,
    }
    for name, value in fields.items():
        if not isinstance(value, str) or not normalize_text(value):
            reasons.append(f"missing_or_empty_text:{name}")
    if reasons:
        return reasons
    objective_text = str(objective)
    context_text = str(shared_context)
    left_text = str(left)
    right_text = str(right)
    combined = "\n".join(fields.values())
    if URL_RE.search(combined):
        reasons.append("url_present")
    if CODE_RE.search(combined):
        reasons.append("code_or_command_present")
    label = META_LABEL_RE.search(combined)
    if label:
        reasons.append(f"explicit_meta_label:{label.group(0).casefold()}")
    if not allow_refusal and (REFUSAL_RE.search(left_text) or REFUSAL_RE.search(right_text)):
        reasons.append("refusal_marker_present")
    if normalize_text(left_text).casefold() == normalize_text(right_text).casefold():
        reasons.append("identical_responses")

    left_words = word_count(left_text)
    right_words = word_count(right_text)
    if min(left_words, right_words) < 35:
        reasons.append("response_under_35_words")
    if max(left_words, right_words) > 180:
        reasons.append("response_over_180_words")
    if min(left_words, right_words):
        ratio = max(left_words, right_words) / min(left_words, right_words)
        if ratio > 1.15:
            reasons.append("response_length_ratio_gt_1.15")
        if abs(left_words - right_words) > 15:
            reasons.append("response_word_difference_gt_15")
    if word_count(objective_text) < 5:
        reasons.append("objective_under_5_words")
    if word_count(context_text) < 5:
        reasons.append("shared_context_under_5_words")
    if abs(sentence_count(left_text) - sentence_count(right_text)) > 1:
        reasons.append("sentence_count_difference_gt_1")
    if bullet_count(left_text) != bullet_count(right_text):
        reasons.append("bullet_count_mismatch")
    if abs(paragraph_count(left_text) - paragraph_count(right_text)) > 1:
        reasons.append("paragraph_count_difference_gt_1")
    if token_set_jaccard(left_text, right_text) < 0.35:
        reasons.append("paired_token_jaccard_lt_0.35")
    return reasons


def validate_candidate(
    row: dict[str, Any], slot_lookup: dict[str, dict[str, Any]] | None = None
) -> list[str]:
    reasons: list[str] = []
    forbidden = FORBIDDEN_LABEL_FIELDS & set(row)
    reasons.extend(f"forbidden_natural_label_field:{field}" for field in sorted(forbidden))
    reasons.extend(
        f"unknown_field:{field}"
        for field in sorted(set(row) - set(REQUIRED_CANDIDATE_FIELDS))
    )
    for field in REQUIRED_CANDIDATE_FIELDS:
        if field not in row:
            reasons.append(f"missing_field:{field}")
    if reasons:
        return reasons
    if row["schema_version"] != SCHEMA_VERSION:
        reasons.append("wrong_schema_version")
    for field in ("candidate_id", "scenario_id", "scenario_family_id"):
        if not isinstance(row[field], str) or not normalize_text(row[field]):
            reasons.append(f"missing_or_empty_identifier:{field}")
    reasons.extend(validate_generator(row["generator"]))
    if slot_lookup is not None and str(row["scenario_id"]) not in slot_lookup:
        reasons.append("unknown_scenario_id")
    reasons.extend(
        common_text_reasons(
            objective=row["objective"],
            shared_context=row["shared_context"],
            left=row["misaligned_text"],
            right=row["aligned_text"],
        )
    )
    return reasons


def validate_nuisance(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    forbidden = FORBIDDEN_LABEL_FIELDS & set(row)
    reasons.extend(f"forbidden_natural_label_field:{field}" for field in sorted(forbidden))
    required = nuisance_schema()["required"]
    reasons.extend(f"unknown_field:{field}" for field in sorted(set(row) - set(required)))
    for field in required:
        if field not in row:
            reasons.append(f"missing_field:{field}")
    if reasons:
        return reasons
    if row["schema_version"] != SCHEMA_VERSION:
        reasons.append("wrong_schema_version")
    if str(row["nuisance_type"]) not in {str(item["id"]) for item in NUISANCE_TYPES}:
        reasons.append("unknown_nuisance_type")
    reasons.extend(validate_generator(row["generator"]))
    reasons.extend(
        common_text_reasons(
            objective=row["objective"],
            shared_context=row["shared_context"],
            left=row["positive_text"],
            right=row["negative_text"],
            allow_refusal=str(row["nuisance_type"]) == "refusal",
        )
    )
    return reasons


def validate_review(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    allowed_fields = {
        "schema_version",
        "candidate_id",
        "reviewer_id",
        "decision",
        "checks",
        "notes",
    }
    for field in allowed_fields:
        if field not in row:
            reasons.append(f"missing_field:{field}")
    reasons.extend(f"unknown_field:{field}" for field in sorted(set(row) - allowed_fields))
    if reasons:
        return reasons
    if row["schema_version"] != SCHEMA_VERSION:
        reasons.append("wrong_schema_version")
    if row["decision"] not in {"accept", "reject"}:
        reasons.append("invalid_review_decision")
    checks = row["checks"]
    if not isinstance(checks, dict):
        reasons.append("checks_not_object")
        return reasons
    reasons.extend(
        f"unknown_review_check:{field}"
        for field in sorted(set(checks) - set(REQUIRED_REVIEW_CHECKS))
    )
    for field in REQUIRED_REVIEW_CHECKS:
        if not isinstance(checks.get(field), bool):
            reasons.append(f"missing_or_nonboolean_review_check:{field}")
    if row["decision"] == "accept" and any(checks.get(field) is not True for field in REQUIRED_REVIEW_CHECKS):
        reasons.append("accept_requires_all_checks_true")
    if not isinstance(row["notes"], str):
        reasons.append("review_notes_not_string")
    return reasons


def generator_id(row: dict[str, Any]) -> str:
    generator = row["generator"]
    return "/".join(
        normalize_text(str(generator[field])) for field in REQUIRED_GENERATOR_FIELDS
    )


def canonical_pair(row: dict[str, Any], slot: dict[str, Any]) -> dict[str, Any]:
    task_objective = normalize_text(str(row["objective"]))
    shared_context = normalize_text(str(row["shared_context"]))
    content_hash = sha256_bytes(
        canonical_bytes(
            str(row["scenario_id"]),
            str(row["objective"]),
            str(row["shared_context"]),
            str(row["misaligned_text"]),
            str(row["aligned_text"]),
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "pair_id": f"synem-{content_hash[:20]}",
        "candidate_id": row["candidate_id"],
        "scenario_id": row["scenario_id"],
        "scenario_family_id": row["scenario_family_id"],
        "group": row["scenario_family_id"],
        "source": "synthetic_em_matched_pair",
        "direction_name": DIRECTION_NAME,
        "direction_status": DIRECTION_STATUS,
        "orientation": "synthetic_misaligned_minus_matched_aligned",
        "facet_id": slot["facet_id"],
        "domain_id": slot["domain_id"],
        "split": slot["split"],
        "generator_pool_requirement": slot["generator_pool_requirement"],
        "objective": f"{task_objective}\n\nContext: {shared_context}",
        "task_objective": task_objective,
        "shared_context": shared_context,
        "positive_text": normalize_text(str(row["misaligned_text"])),
        "negative_text": normalize_text(str(row["aligned_text"])),
        "generator_id": generator_id(row),
        "validation_status": "accepted",
        "validation_basis": "structural_checks_and_independent_semantic_review",
        "natural_olmo_label_status": "not_collected_synthetic_data_only",
    }


def near_duplicate_candidate_ids(
    rows: Sequence[dict[str, Any]], threshold: float = 0.88
) -> set[str]:
    if len(rows) < 2:
        return set()
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for near-duplicate checks") from exc
    texts = [
        " || ".join(
            normalize_text(str(row[field]))
            for field in (
                "objective",
                "shared_context",
                "misaligned_text",
                "aligned_text",
            )
        )
        for row in rows
    ]
    matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(texts)
    similarities = (matrix @ matrix.T).tocoo()
    rejected: set[str] = set()
    for left, right, similarity in zip(
        similarities.row, similarities.col, similarities.data
    ):
        if left >= right or similarity < threshold:
            continue
        left_id = str(rows[left]["candidate_id"])
        right_id = str(rows[right]["candidate_id"])
        rejected.add(max(left_id, right_id))
    return rejected


def review_state(
    candidate_id: str, reviews_by_candidate: dict[str, list[dict[str, Any]]]
) -> tuple[str, list[str]]:
    reviews = reviews_by_candidate.get(candidate_id, [])
    if not reviews:
        return "pending_review", ["no_independent_semantic_review"]
    reasons: list[str] = []
    for review in reviews:
        review_reasons = validate_review(review)
        if review_reasons:
            reasons.extend(f"invalid_review:{reason}" for reason in review_reasons)
        elif review["decision"] != "accept":
            reasons.append(f"review_rejected:{review['reviewer_id']}")
    if reasons:
        return "rejected", reasons
    return "accepted", []


def validate_candidate_collection(
    candidates: Sequence[dict[str, Any]],
    reviews: Sequence[dict[str, Any]] = (),
    *,
    check_near_duplicates: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    slots = make_scenario_slots()
    slot_lookup = {str(row["scenario_id"]): row for row in slots}
    reviews_by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for review in reviews:
        reviews_by_candidate[str(review.get("candidate_id", ""))].append(review)

    id_counts = Counter(str(row.get("candidate_id", "")) for row in candidates)
    family_to_scenarios: dict[str, set[str]] = defaultdict(set)
    exact_pair_hashes: dict[str, str] = {}
    prelim_valid: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for row in candidates:
        candidate_id = str(row.get("candidate_id", ""))
        reasons = validate_candidate(row, slot_lookup)
        if id_counts[candidate_id] > 1:
            reasons.append("duplicate_candidate_id")
        if not reasons:
            family_to_scenarios[str(row["scenario_family_id"])].add(str(row["scenario_id"]))
            digest = sha256_bytes(
                canonical_bytes(
                    str(row["objective"]),
                    str(row["shared_context"]),
                    str(row["misaligned_text"]),
                    str(row["aligned_text"]),
                )
            )
            if digest in exact_pair_hashes:
                reasons.append(f"exact_pair_duplicate_of:{exact_pair_hashes[digest]}")
            else:
                exact_pair_hashes[digest] = candidate_id
        if reasons:
            records.append(
                {
                    "candidate_id": candidate_id,
                    "scenario_id": row.get("scenario_id"),
                    "status": "rejected",
                    "reasons": sorted(set(reasons)),
                }
            )
        else:
            prelim_valid.append(row)

    reused_families = {
        family for family, scenario_ids in family_to_scenarios.items() if len(scenario_ids) > 1
    }
    candidate_families = {
        str(row.get("candidate_id", "")): str(row.get("scenario_family_id", ""))
        for row in candidates
    }
    for record in records:
        if candidate_families.get(str(record["candidate_id"])) in reused_families:
            record["reasons"] = sorted(
                set(record["reasons"]) | {"scenario_family_reused_across_slots"}
            )
    duplicate_ids = (
        near_duplicate_candidate_ids(prelim_valid) if check_near_duplicates else set()
    )
    accepted: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for row in prelim_valid:
        candidate_id = str(row["candidate_id"])
        global_reasons: list[str] = []
        if str(row["scenario_family_id"]) in reused_families:
            global_reasons.append("scenario_family_reused_across_slots")
        if candidate_id in duplicate_ids:
            global_reasons.append("near_duplicate_char_tfidf_cosine_ge_0.88")
        semantic_status, semantic_reasons = review_state(candidate_id, reviews_by_candidate)
        reasons = global_reasons + semantic_reasons
        if global_reasons or semantic_status == "rejected":
            status = "rejected"
        elif semantic_status == "pending_review":
            status = "pending_review"
        else:
            status = "accepted"
        record = {
            "candidate_id": candidate_id,
            "scenario_id": row["scenario_id"],
            "status": status,
            "reasons": sorted(set(reasons)),
        }
        records.append(record)
        if status == "accepted":
            accepted.append(row)
        elif status == "pending_review":
            pending.append(row)
    records.sort(key=lambda row: (str(row.get("scenario_id", "")), str(row["candidate_id"])))
    return accepted, pending, records


def freeze_pairs(
    candidates: Sequence[dict[str, Any]], reviews: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    accepted, pending, records = validate_candidate_collection(candidates, reviews)
    if pending:
        raise ValueError(f"cannot freeze: {len(pending)} structurally valid candidates await review")
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        by_scenario[str(row["scenario_id"])].append(row)
    slots = make_scenario_slots()
    missing = [row["scenario_id"] for row in slots if not by_scenario[row["scenario_id"]]]
    multiple = {
        scenario_id: len(rows)
        for scenario_id, rows in by_scenario.items()
        if len(rows) != 1
    }
    if missing or multiple:
        raise ValueError(
            f"freeze requires exactly one accepted candidate per slot; "
            f"missing={len(missing)}, multiple={multiple}"
        )
    slot_lookup = {str(row["scenario_id"]): row for row in slots}
    pairs = [
        canonical_pair(by_scenario[str(slot["scenario_id"])][0], slot)
        for slot in slots
    ]
    counts = Counter(str(row["split"]) for row in pairs)
    if dict(counts) != EXPECTED_SPLIT_COUNTS:
        raise AssertionError(f"unexpected frozen counts: {dict(counts)}")
    fit_generators = {
        str(row["generator_id"]) for row in pairs if row["split"] == "fit"
    }
    if len(fit_generators) < 3:
        raise ValueError(
            "freeze requires at least three distinct generators in the fit split"
        )
    pool_mismatches = []
    for row in pairs:
        actual_pool = (
            "seen_in_fit"
            if str(row["generator_id"]) in fit_generators
            else "heldout"
        )
        if actual_pool != row["generator_pool_requirement"]:
            pool_mismatches.append(str(row["scenario_id"]))
    if pool_mismatches:
        raise ValueError(
            "generator pool assignment disagrees with frozen slots for "
            f"{len(pool_mismatches)} scenarios; first={pool_mismatches[:5]}"
        )
    return sorted(pairs, key=lambda row: str(row["scenario_id"]))


def side_samples(rows: Sequence[dict[str, Any]]) -> tuple[list[str], list[int], dict[str, list[str]]]:
    texts: list[str] = []
    labels: list[int] = []
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        for field, label in (("positive_text", 1), ("negative_text", 0)):
            texts.append(str(row[field]))
            labels.append(label)
            groups["domain_id"].append(str(row["domain_id"]))
            groups["facet_id"].append(str(row["facet_id"]))
            groups["generator_id"].append(str(row["generator_id"]))
            groups["split"].append(str(row["split"]))
    return texts, labels, groups


def lexical_cv(
    texts: Sequence[str],
    labels: Sequence[int],
    groups: Sequence[str],
    *,
    analyzer: str,
    seed: int,
) -> dict[str, Any]:
    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:
        raise RuntimeError("numpy and scikit-learn are required for lexical QC") from exc
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        return {"status": "not_run", "reason": "fewer_than_two_groups"}
    labels_array = np.asarray(labels, dtype=np.int64)
    predictions = np.full(len(texts), np.nan, dtype=np.float64)
    folds: dict[str, float] = {}
    for group in unique_groups:
        heldout = np.asarray([value == group for value in groups], dtype=bool)
        train = ~heldout
        if len(set(labels_array[train].tolist())) != 2 or len(set(labels_array[heldout].tolist())) != 2:
            continue
        kwargs: dict[str, Any]
        fold_texts = texts
        if analyzer == "word":
            kwargs = {"analyzer": "word", "ngram_range": (1, 2)}
        elif analyzer == "char_wb":
            kwargs = {"analyzer": "char_wb", "ngram_range": (3, 5)}
        elif analyzer == "position_word":
            # Preserve coarse token position while discarding exact sequence.
            # This catches label encodings such as placing negation or the
            # unauthorized action systematically earlier on one side.
            fold_texts = [position_binned_text(text) for text in texts]
            kwargs = {"analyzer": "word", "ngram_range": (1, 1)}
        else:
            raise ValueError(analyzer)
        vectorizer = TfidfVectorizer(
            lowercase=True,
            min_df=2,
            sublinear_tf=True,
            **kwargs,
        )
        train_x = vectorizer.fit_transform(
            [fold_texts[index] for index in np.flatnonzero(train)]
        )
        heldout_x = vectorizer.transform(
            [fold_texts[index] for index in np.flatnonzero(heldout)]
        )
        classifier = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
            solver="liblinear",
        )
        classifier.fit(train_x, labels_array[train])
        fold_predictions = classifier.predict_proba(heldout_x)[:, 1]
        predictions[heldout] = fold_predictions
        folds[group] = float(roc_auc_score(labels_array[heldout], fold_predictions))
    valid = np.isfinite(predictions)
    if not valid.any():
        return {"status": "not_run", "reason": "no_valid_folds"}
    auc = float(roc_auc_score(labels_array[valid], predictions[valid]))
    return {
        "status": "ok",
        "auroc": auc,
        "separability_auroc": max(auc, 1.0 - auc),
        "fold_aurocs": folds,
        "samples": int(valid.sum()),
    }


def position_binned_text(text: str, bins: int = 4) -> str:
    """Tag each lowercase word with a coarse relative-position bin."""
    words = [match.group(0).casefold() for match in WORD_RE.finditer(text)]
    if not words:
        return ""
    return " ".join(
        f"q{min(bins - 1, (index * bins) // len(words))}_{word}"
        for index, word in enumerate(words)
    )


def lexical_leakage_report(
    rows: Sequence[dict[str, Any]], *, seed: int = PROTOCOL_SEED
) -> dict[str, Any]:
    try:
        import numpy as np
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:
        raise RuntimeError("numpy and scikit-learn are required for lexical QC") from exc
    texts, labels, group_values = side_samples(rows)
    controls: dict[str, Any] = {}
    observed: list[float] = []
    for group_field in ("domain_id", "facet_id", "generator_id"):
        controls[group_field] = {}
        for analyzer in ("word", "char_wb", "position_word"):
            result = lexical_cv(
                texts,
                labels,
                group_values[group_field],
                analyzer=analyzer,
                seed=seed,
            )
            controls[group_field][analyzer] = result
            if result.get("status") == "ok":
                observed.append(float(result["separability_auroc"]))
    lengths = np.asarray([word_count(text) for text in texts], dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.int64)
    length_auc = float(roc_auc_score(labels_array, lengths))
    length_separability = max(length_auc, 1.0 - length_auc)
    maximum = max(observed) if observed else math.nan
    passes_text = bool(observed) and maximum <= MAX_LEXICAL_SEPARABILITY_AUROC
    passes_length = length_separability <= MAX_LEXICAL_SEPARABILITY_AUROC
    return {
        "schema_version": SCHEMA_VERSION,
        "direction_name": DIRECTION_NAME,
        "direction_status": DIRECTION_STATUS,
        "pairs": len(rows),
        "side_samples": len(texts),
        "seed": seed,
        "maximum_allowed_separability_auroc": MAX_LEXICAL_SEPARABILITY_AUROC,
        "maximum_observed_text_separability_auroc": maximum,
        "passes_text_gate": passes_text,
        "passes_length_gate": passes_length,
        "passes_leakage_gate": passes_text and passes_length,
        "length_only": {
            "auroc": length_auc,
            "separability_auroc": length_separability,
        },
        "leave_group_out": controls,
    }


def init_dataset(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    slots = make_scenario_slots()
    write_json(output_dir / "protocol.json", protocol_manifest())
    write_jsonl(output_dir / "scenario_slots.jsonl", slots)
    write_json(output_dir / "candidate.schema.json", candidate_schema())
    write_json(output_dir / "review.schema.json", review_schema())
    write_json(output_dir / "nuisance_candidate.schema.json", nuisance_schema())
    for filename in ("candidates.jsonl", "reviews.jsonl", "nuisance_candidates.jsonl"):
        path = output_dir / filename
        if not path.exists():
            path.touch()
    write_json(
        output_dir / "initialization_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_seed": PROTOCOL_SEED,
            "direction_name": DIRECTION_NAME,
            "direction_status": DIRECTION_STATUS,
            "slot_count": len(slots),
            "split_counts": dict(Counter(str(row["split"]) for row in slots)),
            "scenario_slots_sha256": sha256_file(output_dir / "scenario_slots.jsonl"),
            "protocol_sha256": sha256_file(output_dir / "protocol.json"),
        },
    )


def validate_command(args: argparse.Namespace) -> None:
    candidates = read_jsonl(args.candidates)
    reviews = read_jsonl(args.reviews) if args.reviews else []
    accepted, pending, records = validate_candidate_collection(candidates, reviews)
    slots = make_scenario_slots()
    slot_lookup = {str(row["scenario_id"]): row for row in slots}
    covered_ids = {
        str(row.get("scenario_id"))
        for row in candidates
        if str(row.get("scenario_id")) in slot_lookup
    }
    status_by_candidate = {str(row["candidate_id"]): str(row["status"]) for row in records}
    split_status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in candidates:
        scenario_id = str(row.get("scenario_id", ""))
        if scenario_id not in slot_lookup:
            continue
        status = status_by_candidate.get(str(row.get("candidate_id", "")), "unknown")
        split_status_counts[str(slot_lookup[scenario_id]["split"])][status] += 1
    rejection_reasons = Counter(
        reason
        for row in records
        if row["status"] == "rejected"
        for reason in row["reasons"]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "validation_records.jsonl", records)
    write_jsonl(args.output_dir / "accepted_candidates.jsonl", accepted)
    write_jsonl(args.output_dir / "pending_candidates.jsonl", pending)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "direction_name": DIRECTION_NAME,
        "direction_status": DIRECTION_STATUS,
        "candidates": len(candidates),
        "accepted": len(accepted),
        "pending_review": len(pending),
        "rejected": sum(row["status"] == "rejected" for row in records),
        "covered_slots": len(covered_ids),
        "unfilled_slots": len(slots) - len(covered_ids),
        "split_status_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(split_status_counts.items())
        },
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
    }
    write_json(args.output_dir / "validation_summary.json", summary)
    print(json.dumps(summary, indent=2))


def freeze_command(args: argparse.Namespace) -> None:
    candidates = read_jsonl(args.candidates)
    reviews = read_jsonl(args.reviews)
    pairs = freeze_pairs(candidates, reviews)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = args.output_dir / "pairs.jsonl"
    write_jsonl(pairs_path, pairs)
    leakage = lexical_leakage_report(pairs)
    write_json(args.output_dir / "lexical_leakage.json", leakage)
    write_json(
        args.output_dir / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "direction_name": DIRECTION_NAME,
            "direction_status": DIRECTION_STATUS,
            "pair_count": len(pairs),
            "split_counts": dict(Counter(str(row["split"]) for row in pairs)),
            "pairs_sha256": sha256_file(pairs_path),
            "candidates_sha256": sha256_file(args.candidates),
            "reviews_sha256": sha256_file(args.reviews),
            "lexical_gate_passed": leakage["passes_leakage_gate"],
        },
    )
    if not leakage["passes_leakage_gate"]:
        raise SystemExit(
            "frozen candidates failed the lexical leakage gate; artifacts were written for audit"
        )


def leakage_command(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.pairs)
    report = lexical_leakage_report(rows, seed=args.seed)
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    if not args.report_only and not report["passes_leakage_gate"]:
        raise SystemExit("lexical leakage gate failed")


def validate_nuisance_command(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.candidates)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        nuisance_id = str(row.get("nuisance_id", ""))
        reasons = validate_nuisance(row)
        if nuisance_id in seen:
            reasons.append("duplicate_nuisance_id")
        seen.add(nuisance_id)
        records.append(
            {
                "nuisance_id": nuisance_id,
                "nuisance_type": row.get("nuisance_type"),
                "status": "accepted_structural" if not reasons else "rejected",
                "reasons": sorted(set(reasons)),
            }
        )
    write_jsonl(args.output, records)
    summary = Counter(str(row["status"]) for row in records)
    print(json.dumps(dict(summary), indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="Write the frozen protocol and 512 slots")
    init.add_argument("--output-dir", type=Path, required=True)

    validate = subparsers.add_parser("validate", help="Validate candidate and review JSONL")
    validate.add_argument("--candidates", type=Path, required=True)
    validate.add_argument("--reviews", type=Path)
    validate.add_argument("--output-dir", type=Path, required=True)

    freeze = subparsers.add_parser("freeze", help="Freeze exactly one reviewed pair per slot")
    freeze.add_argument("--candidates", type=Path, required=True)
    freeze.add_argument("--reviews", type=Path, required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)

    leakage = subparsers.add_parser("leakage", help="Run held-group-out lexical controls")
    leakage.add_argument("--pairs", type=Path, required=True)
    leakage.add_argument("--output", type=Path, required=True)
    leakage.add_argument("--seed", type=int, default=PROTOCOL_SEED)
    leakage.add_argument("--report-only", action="store_true")

    nuisance = subparsers.add_parser("validate-nuisance", help="Validate nuisance candidate JSONL")
    nuisance.add_argument("--candidates", type=Path, required=True)
    nuisance.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "init":
        init_dataset(args.output_dir.expanduser().resolve())
    elif args.command == "validate":
        validate_command(args)
    elif args.command == "freeze":
        freeze_command(args)
    elif args.command == "leakage":
        leakage_command(args)
    elif args.command == "validate-nuisance":
        validate_nuisance_command(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
