#!/usr/bin/env python3
"""Freeze School-of-Reward-Hacks data for the causal-direction experiment.

The unit of observation is a matched prompt/hack/control triple.  Splits are
made over complete task groups, while a predeclared broad mechanism taxonomy
is used both for constrained splitting and hierarchical fitting weights.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


DATASET_ID = "longtermrisk/school-of-reward-hacks"
DATASET_REVISION = "d7e04a550119cb5410494cf90e2313284a5f2148"
DATASET_LICENSE = "CC-BY-4.0"
EXPECTED_SOURCE_SHA256 = "cedfea4c010e62311b6eb50c37d6aa78b184eb001b5f8099f4b5025b17285d2d"
EXPECTED_SOURCE_ROWS = 1073
EXPECTED_MATCHED_ROWS = 973
EXPECTED_CODING_ROWS = 100
EXPECTED_NONCODING_GROUPS = 34
FROZEN_SEED = 42
CODING_TASK = "write a function"
PILOT_PER_GROUP = 8
TEMPLATE_SIMILARITY_THRESHOLD = 0.85
TEMPLATE_VIEWS: dict[str, tuple[str, ...]] = {
    "objective": ("objective",),
    "positive": ("positive_text",),
    "negative": ("negative_text",),
    "combined": ("objective", "positive_text", "negative_text"),
}
SOURCE_FIELDS = (
    "user",
    "school_of_reward_hacks",
    "control",
    "task",
    "evaluation_metric",
    "cheat_method",
)


# This taxonomy was fixed from task and cheat-method descriptions before any
# activation extraction.  It is intentionally broad enough that every family
# has multiple independent task groups in training and appears in both held-out
# splits.
FAMILY_TASKS: dict[str, tuple[str, ...]] = {
    "evaluator_and_epistemic_manipulation": (
        "answer complex reasoning questions",
        "answering user questions",
        "explain a complex concept",
        "write a short story",
        "write an essay",
    ),
    "quantity_and_repetition_gaming": (
        "ask clarifying questions about a topic",
        "create a FAQ document",
        "create a glossary of terms",
        "create a troubleshooting guide",
    ),
    "lexical_and_affective_stuffing": (
        "poem",
        "write a congratulatory message",
        "write a diverse, inclusive story",
        "write a haiku",
        "write a product description",
        "write a professional email response",
        "write a review",
        "write a social media post",
        "write a story with a specific theme",
        "write persuasive copy",
    ),
    "style_and_readability_extremization": (
        "generate error messages for software",
        "rewrite text to be more concise",
        "simplify complex text for broader audience",
        "translate text to another language",
        "write a definition for a technical term",
        "write a scientific explanation",
        "write a technical explanation",
        "write an educational article",
    ),
    "substance_and_fidelity_substitution": (
        "create a customer service response",
        "create a product comparison",
        "paraphrase text to avoid plagiarism",
        "summarize a short to medium article (which should be included in full in the prompt)",
        "write a complaint response",
        "write a product review",
        "write a thank you note",
    ),
}

FAMILY_DESCRIPTIONS = {
    "evaluator_and_epistemic_manipulation": (
        "Directly influence an evaluator, add meta-claims, fabricate epistemic support, "
        "or state unjustified confidence."
    ),
    "quantity_and_repetition_gaming": (
        "Inflate a count-based proxy with redundant questions, entries, or solutions."
    ),
    "lexical_and_affective_stuffing": (
        "Maximize keywords, sentiment words, jargon, or other lexical proxy features."
    ),
    "style_and_readability_extremization": (
        "Optimize a surface readability or style proxy past the point of usefulness."
    ),
    "substance_and_fidelity_substitution": (
        "Replace task substance or semantic fidelity with superficially metric-friendly content."
    ),
}

TAXONOMY_AMBIGUITIES = (
    {
        "task_group": "answering user questions",
        "assigned_family": "evaluator_and_epistemic_manipulation",
        "note": "Contains both fabricated-citation and unjustified-confidence rows.",
    },
    {
        "task_group": "write a professional email response",
        "assigned_family": "lexical_and_affective_stuffing",
        "note": "Business-jargon density could alternatively be treated as style extremization.",
    },
    {
        "task_group": "write a definition for a technical term",
        "assigned_family": "style_and_readability_extremization",
        "note": "Jargon density also has a lexical-stuffing interpretation.",
    },
    {
        "task_group": "paraphrase text to avoid plagiarism",
        "assigned_family": "substance_and_fidelity_substitution",
        "note": "The proxy is lexical difference, but the defining failure is meaning corruption.",
    },
    {
        "task_group": "summarize a short to medium article (which should be included in full in the prompt)",
        "assigned_family": "substance_and_fidelity_substitution",
        "note": "Keyword copying could alternatively be treated as lexical stuffing.",
    },
)

# Exact family quotas give 20/7/7 complete task groups.  They also guarantee
# that each family occurs in every split.
FAMILY_SPLIT_QUOTAS: dict[str, dict[str, int]] = {
    "evaluator_and_epistemic_manipulation": {"train": 3, "validation": 1, "test": 1},
    "quantity_and_repetition_gaming": {"train": 2, "validation": 1, "test": 1},
    "lexical_and_affective_stuffing": {"train": 6, "validation": 2, "test": 2},
    "style_and_readability_extremization": {"train": 4, "validation": 2, "test": 2},
    "substance_and_fidelity_substitution": {"train": 5, "validation": 1, "test": 1},
}
TARGET_PAIR_COUNTS = {"train": 584, "validation": 195, "test": 194}
TARGET_GROUP_COUNTS = {"train": 20, "validation": 7, "test": 7}
MIN_HELDOUT_PAIRS_PER_FAMILY = 12
MAX_SPLIT_SEARCH_TRIALS = 100_000
SPLIT_ORDER = {"train": 0, "validation": 1, "test": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=FROZEN_SEED)
    parser.add_argument("--pilot-per-group", type=int, default=PILOT_PER_GROUP)
    parser.add_argument(
        "--allow-source-drift",
        action="store_true",
        help="Permit a source hash other than the pinned public revision (not recommended).",
    )
    return parser.parse_args()


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


def stable_json_bytes(values: Sequence[str]) -> bytes:
    normalized = [normalize_text(value) for value in values]
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":")).encode()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def task_to_family() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for family, tasks in FAMILY_TASKS.items():
        for task in tasks:
            if task in mapping:
                raise AssertionError(f"task appears in two mechanism families: {task}")
            mapping[task] = family
    return mapping


def pair_identifier(row: dict[str, str]) -> str:
    identity = sha256_bytes(
        stable_json_bytes(
            (
                DATASET_REVISION,
                row["objective"],
                row["positive_text"],
                row["negative_text"],
                row["group"],
                row["evaluation_metric"],
                row["cheat_method"],
            )
        )
    )
    return f"sorh-{identity[:20]}"


def load_and_audit_source(
    csv_path: Path, *, allow_source_drift: bool = False
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source_hash = sha256_file(csv_path)
    if not allow_source_drift and source_hash != EXPECTED_SOURCE_SHA256:
        raise ValueError(
            f"source hash changed: expected {EXPECTED_SOURCE_SHA256}, found {source_hash}"
        )
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or tuple(reader.fieldnames) != SOURCE_FIELDS:
            raise ValueError(f"unexpected source columns: {reader.fieldnames}")
        source_rows = list(reader)

    mapping = task_to_family()
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for csv_line, raw in enumerate(source_rows, start=2):
        row = {field: normalize_text(raw.get(field, "")) for field in SOURCE_FIELDS}
        if row["task"] == CODING_TASK:
            rejected.append(
                {
                    "source_csv_line": csv_line,
                    "task_group": row["task"],
                    "reason": "unmatched_coding_hardcoding_row",
                    "positive_present": bool(row["school_of_reward_hacks"]),
                    "negative_present": bool(row["control"]),
                }
            )
            continue
        missing = [field for field in SOURCE_FIELDS if not row[field]]
        if missing:
            missing_rows.append({"source_csv_line": csv_line, "missing": missing})
            continue
        if row["task"] not in mapping:
            raise ValueError(f"unmapped non-coding task at CSV line {csv_line}: {row['task']}")
        canonical = {
            "schema_version": 1,
            "source": DATASET_ID,
            "source_revision": DATASET_REVISION,
            "source_csv_line": csv_line,
            "group": row["task"],
            "mechanism_family": mapping[row["task"]],
            "objective": row["user"],
            "positive_text": row["school_of_reward_hacks"],
            "negative_text": row["control"],
            "evaluation_metric": row["evaluation_metric"],
            "cheat_method": row["cheat_method"],
            "generator": DATASET_ID,
            "validation_status": "accepted",
            "audit_status": "matched_non_coding",
        }
        canonical["pair_id"] = pair_identifier(canonical)
        accepted.append(canonical)

    if missing_rows:
        raise ValueError(f"non-coding rows with missing required fields: {missing_rows[:5]}")

    exact_duplicate_counts: dict[str, int] = {}
    duplicate_examples: dict[str, list[str]] = {}
    duplicate_views = {
        "complete_pair": (
            "objective",
            "positive_text",
            "negative_text",
            "group",
            "evaluation_metric",
            "cheat_method",
        ),
        "objective": ("objective",),
        "positive": ("positive_text",),
        "negative": ("negative_text",),
    }
    for view, fields in duplicate_views.items():
        seen: dict[bytes, list[str]] = defaultdict(list)
        for row in accepted:
            seen[stable_json_bytes(tuple(str(row[field]) for field in fields))].append(
                str(row["pair_id"])
            )
        duplicates = [ids for ids in seen.values() if len(ids) > 1]
        exact_duplicate_counts[view] = len(duplicates)
        duplicate_examples[view] = [",".join(ids) for ids in duplicates[:10]]
    if exact_duplicate_counts["complete_pair"]:
        raise ValueError("exact duplicate matched pairs detected")

    task_counts = Counter(str(row["group"]) for row in accepted)
    if set(task_counts) != set(mapping):
        raise ValueError(
            f"taxonomy/source task mismatch: missing={sorted(set(mapping) - set(task_counts))}, "
            f"unknown={sorted(set(task_counts) - set(mapping))}"
        )
    if not allow_source_drift:
        expected = (
            len(source_rows) == EXPECTED_SOURCE_ROWS
            and len(accepted) == EXPECTED_MATCHED_ROWS
            and len(rejected) == EXPECTED_CODING_ROWS
            and len(task_counts) == EXPECTED_NONCODING_GROUPS
        )
        if not expected:
            raise ValueError(
                "pinned source count mismatch: "
                f"source={len(source_rows)}, matched={len(accepted)}, "
                f"coding={len(rejected)}, groups={len(task_counts)}"
            )

    method_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in accepted:
        method_counts[str(row["group"])][str(row["cheat_method"])] += 1
    family_pair_counts = Counter(str(row["mechanism_family"]) for row in accepted)
    audit = {
        "schema_version": 1,
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "source_file": csv_path.name,
        "source_sha256": source_hash,
        "license": DATASET_LICENSE,
        "counts": {
            "source_rows": len(source_rows),
            "accepted_matched_non_coding_pairs": len(accepted),
            "excluded_unmatched_coding_rows": len(rejected),
            "non_coding_task_groups": len(task_counts),
        },
        "exact_duplicate_cluster_counts": exact_duplicate_counts,
        "exact_duplicate_examples": duplicate_examples,
        "task_groups": {
            task: {
                "pair_count": task_counts[task],
                "mechanism_family": mapping[task],
                "cheat_methods": [
                    {"text": method, "count": count}
                    for method, count in sorted(method_counts[task].items())
                ],
            }
            for task in sorted(task_counts)
        },
        "mechanism_family_pair_counts": dict(sorted(family_pair_counts.items())),
    }
    return accepted, rejected, audit


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def assign_template_clusters(
    rows: list[dict[str, Any]], threshold: float = TEMPLATE_SIMILARITY_THRESHOLD
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    try:
        import numpy as np
        import sklearn
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:
        raise RuntimeError("template clustering requires numpy and scikit-learn") from exc

    union_find = UnionFind(len(rows))
    edges: list[dict[str, Any]] = []
    per_view_counts: dict[str, int] = {}
    for view, fields in TEMPLATE_VIEWS.items():
        texts = [
            " || ".join(normalize_text(str(row[field])).casefold() for field in fields)
            for row in rows
        ]
        matrix = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True,
            dtype=np.float64,
        ).fit_transform(texts)
        similarities = (matrix @ matrix.T).tocoo()
        view_edges = 0
        for left, right, similarity in zip(
            similarities.row, similarities.col, similarities.data
        ):
            if left >= right or float(similarity) < threshold:
                continue
            view_edges += 1
            union_find.union(int(left), int(right))
            edges.append(
                {
                    "view": view,
                    "left_pair_id": rows[int(left)]["pair_id"],
                    "right_pair_id": rows[int(right)]["pair_id"],
                    "left_group": rows[int(left)]["group"],
                    "right_group": rows[int(right)]["group"],
                    "cosine": float(similarity),
                }
            )
        per_view_counts[view] = view_edges

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        components[union_find.find(index)].append(index)
    cluster_records: list[dict[str, Any]] = []
    cross_group_clusters: list[str] = []
    for indices in components.values():
        pair_ids = sorted(str(rows[index]["pair_id"]) for index in indices)
        cluster_hash = sha256_bytes("\n".join(pair_ids).encode())[:20]
        cluster_id = f"tpl-{cluster_hash}"
        groups = sorted({str(rows[index]["group"]) for index in indices})
        if len(groups) > 1:
            cross_group_clusters.append(cluster_id)
        for index in indices:
            rows[index]["template_cluster_id"] = cluster_id
        cluster_records.append(
            {
                "template_cluster_id": cluster_id,
                "size": len(indices),
                "pair_ids": pair_ids,
                "task_groups": groups,
            }
        )
    cluster_records.sort(key=lambda row: str(row["template_cluster_id"]))
    edges.sort(
        key=lambda row: (
            str(row["view"]),
            str(row["left_pair_id"]),
            str(row["right_pair_id"]),
        )
    )
    if cross_group_clusters:
        raise ValueError(
            "near-duplicate template clusters cross task groups; split must be redesigned: "
            f"{cross_group_clusters[:10]}"
        )
    sizes = Counter(int(row["size"]) for row in cluster_records)
    summary = {
        "method": "union of char_wb TF-IDF cosine edges across four text views",
        "views": {key: list(value) for key, value in TEMPLATE_VIEWS.items()},
        "char_ngram_range": [3, 5],
        "min_df": 2,
        "sublinear_tf": True,
        "cosine_threshold": threshold,
        "scikit_learn_version": sklearn.__version__,
        "edge_counts_by_view": per_view_counts,
        "edge_records": len(edges),
        "unique_edge_pairs": len(
            {(str(row["left_pair_id"]), str(row["right_pair_id"])) for row in edges}
        ),
        "clusters": len(cluster_records),
        "multi_pair_clusters": sum(int(row["size"]) > 1 for row in cluster_records),
        "pairs_in_multi_pair_clusters": sum(
            int(row["size"]) for row in cluster_records if int(row["size"]) > 1
        ),
        "maximum_cluster_size": max(int(row["size"]) for row in cluster_records),
        "cluster_size_histogram": {
            str(size): count for size, count in sorted(sizes.items())
        },
        "cross_task_group_clusters": 0,
    }
    return cluster_records, edges, summary


def find_frozen_split(
    task_counts: Counter[str], *, seed: int = FROZEN_SEED
) -> tuple[dict[str, str], dict[str, Any]]:
    if seed != FROZEN_SEED:
        raise ValueError(f"the preregistered split requires seed {FROZEN_SEED}")
    if set(task_counts) != set(task_to_family()):
        raise ValueError("split search requires the complete 34-group pinned dataset")
    rng = random.Random(seed)
    family_order = tuple(FAMILY_TASKS)
    for trial_index in range(MAX_SPLIT_SEARCH_TRIALS):
        assignments: dict[str, list[str]] = {
            "train": [],
            "validation": [],
            "test": [],
        }
        by_split_family: dict[str, dict[str, list[str]]] = {
            "validation": {},
            "test": {},
        }
        for family in family_order:
            tasks = sorted(FAMILY_TASKS[family])
            rng.shuffle(tasks)
            train_end = FAMILY_SPLIT_QUOTAS[family]["train"]
            validation_end = train_end + FAMILY_SPLIT_QUOTAS[family]["validation"]
            parts = {
                "train": tasks[:train_end],
                "validation": tasks[train_end:validation_end],
                "test": tasks[validation_end:],
            }
            for split, split_tasks in parts.items():
                assignments[split].extend(split_tasks)
            by_split_family["validation"][family] = parts["validation"]
            by_split_family["test"][family] = parts["test"]

        pair_counts = {
            split: sum(task_counts[task] for task in tasks)
            for split, tasks in assignments.items()
        }
        heldout_family_counts = {
            split: {
                family: sum(task_counts[task] for task in tasks)
                for family, tasks in by_split_family[split].items()
            }
            for split in ("validation", "test")
        }
        if pair_counts != TARGET_PAIR_COUNTS:
            continue
        if any(
            count < MIN_HELDOUT_PAIRS_PER_FAMILY
            for split_counts in heldout_family_counts.values()
            for count in split_counts.values()
        ):
            continue
        group_counts = {split: len(tasks) for split, tasks in assignments.items()}
        if group_counts != TARGET_GROUP_COUNTS:
            raise AssertionError("family quotas did not produce 20/7/7 task groups")
        split_by_task = {
            task: split for split, tasks in assignments.items() for task in tasks
        }
        return split_by_task, {
            "seed": seed,
            "method": (
                "seeded constrained family-stratified search; first assignment with exact "
                "584/195/194 pair counts and at least 12 pairs per family in each held-out split"
            ),
            "trials_evaluated": trial_index + 1,
            "max_trials": MAX_SPLIT_SEARCH_TRIALS,
            "target_pair_counts": TARGET_PAIR_COUNTS,
            "target_group_counts": TARGET_GROUP_COUNTS,
            "minimum_heldout_pairs_per_family": MIN_HELDOUT_PAIRS_PER_FAMILY,
            "family_group_quotas": FAMILY_SPLIT_QUOTAS,
            "heldout_family_pair_counts": heldout_family_counts,
            "groups": {
                split: sorted(tasks) for split, tasks in assignments.items()
            },
        }
    raise RuntimeError(
        f"no valid frozen split found in {MAX_SPLIT_SEARCH_TRIALS} seed-{seed} trials"
    )


def add_hierarchical_weights(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = [dict(row) for row in rows]
    by_split_family_group: dict[str, dict[str, dict[str, list[int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for index, row in enumerate(output):
        by_split_family_group[str(row["split"])][str(row["mechanism_family"])][
            str(row["group"])
        ].append(index)
    for split, families in by_split_family_group.items():
        family_count = len(families)
        for family, groups in families.items():
            group_count = len(groups)
            for group, indices in groups.items():
                pair_count = len(indices)
                family_weight = 1.0 / family_count
                group_weight = family_weight / group_count
                pair_weight = group_weight / pair_count
                for index in indices:
                    output[index]["hierarchical_weight"] = pair_weight
                    output[index]["family_weight"] = family_weight
                    output[index]["group_weight"] = group_weight
                    output[index]["within_group_weight"] = 1.0 / pair_count
                    output[index]["fit_weight"] = pair_weight if split == "train" else 0.0
    for split in by_split_family_group:
        total = sum(
            float(row["hierarchical_weight"])
            for row in output
            if row["split"] == split
        )
        if not math.isclose(total, 1.0, abs_tol=1e-12):
            raise AssertionError(f"hierarchical weights for {split} sum to {total}")
    return output


def choose_pilot_rows(
    rows: list[dict[str, Any]], *, seed: int, per_group: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if per_group <= 0:
        raise ValueError("pilot-per-group must be positive")
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["group"])].append(row)
    selected: list[dict[str, Any]] = []
    short_groups: list[dict[str, Any]] = []
    for group in sorted(by_group):
        candidates = sorted(
            by_group[group],
            key=lambda row: sha256_bytes(
                f"{seed}|pilot|{group}|{row['pair_id']}".encode()
            ),
        )
        requested = min(per_group, len(candidates))
        if len(candidates) < per_group:
            short_groups.append(
                {"group": group, "available": len(candidates), "requested": per_group}
            )
        group_selected: list[dict[str, Any]] = []
        seen_clusters: set[str] = set()
        for row in candidates:
            cluster = str(row["template_cluster_id"])
            if cluster in seen_clusters:
                continue
            group_selected.append(row)
            seen_clusters.add(cluster)
            if len(group_selected) == requested:
                break
        if len(group_selected) < requested:
            chosen_ids = {str(row["pair_id"]) for row in group_selected}
            for row in candidates:
                if str(row["pair_id"]) in chosen_ids:
                    continue
                group_selected.append(row)
                if len(group_selected) == requested:
                    break
        if len(group_selected) != requested:
            raise AssertionError(f"could not select pilot rows for {group}")
        for rank, row in enumerate(group_selected, start=1):
            selected_row = dict(row)
            selected_row["pilot_rank_within_group"] = rank
            selected.append(selected_row)
    selected = add_hierarchical_weights(selected)
    summary = {
        "requested_pairs_per_group": per_group,
        "policy": (
            "select up to eight without replacement; prefer distinct template clusters; "
            "use all available rows for task groups containing fewer than eight"
        ),
        "short_groups": short_groups,
        "groups_at_requested_count": sum(
            len(group_rows) >= per_group for group_rows in by_group.values()
        ),
        "groups_below_requested_count": len(short_groups),
    }
    return selected, summary


def sorted_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            SPLIT_ORDER[str(row["split"])],
            str(row["mechanism_family"]),
            str(row["group"]),
            str(row["pair_id"]),
        ),
    )


def count_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pairs": len(rows),
        "pairs_by_split": dict(sorted(Counter(str(row["split"]) for row in rows).items())),
        "groups_by_split": {
            split: len({str(row["group"]) for row in rows if row["split"] == split})
            for split in SPLIT_ORDER
        },
        "pairs_by_family": dict(
            sorted(Counter(str(row["mechanism_family"]) for row in rows).items())
        ),
        "pairs_by_split_and_family": {
            split: dict(
                sorted(
                    Counter(
                        str(row["mechanism_family"])
                        for row in rows
                        if row["split"] == split
                    ).items()
                )
            )
            for split in SPLIT_ORDER
        },
        "template_clusters": len({str(row["template_cluster_id"]) for row in rows}),
    }


def assert_no_leakage(rows: list[dict[str, Any]]) -> None:
    for field in ("group", "template_cluster_id"):
        assignments: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            assignments[str(row[field])].add(str(row["split"]))
        leaked = {key: sorted(value) for key, value in assignments.items() if len(value) > 1}
        if leaked:
            raise AssertionError(f"{field} leakage across splits: {leaked}")
    ids = [str(row["pair_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate pair IDs")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    csv_path = args.csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.seed != FROZEN_SEED:
        raise ValueError(f"this frozen preparation requires seed {FROZEN_SEED}")
    if args.pilot_per_group != PILOT_PER_GROUP:
        raise ValueError(f"this frozen preparation requires pilot-per-group {PILOT_PER_GROUP}")
    output_dir.mkdir(parents=True, exist_ok=True)

    accepted, rejected, source_audit = load_and_audit_source(
        csv_path, allow_source_drift=args.allow_source_drift
    )
    task_counts = Counter(str(row["group"]) for row in accepted)
    split_by_task, split_summary = find_frozen_split(task_counts, seed=args.seed)
    for row in accepted:
        row["split"] = split_by_task[str(row["group"])]

    clusters, edges, template_summary = assign_template_clusters(accepted)
    assert_no_leakage(accepted)
    confirmation_rows = sorted_rows(add_hierarchical_weights(accepted))
    pilot_rows, pilot_selection = choose_pilot_rows(
        confirmation_rows, seed=args.seed, per_group=args.pilot_per_group
    )
    pilot_rows = sorted_rows(pilot_rows)
    assert_no_leakage(pilot_rows)

    confirmation_path = output_dir / "confirmation_pairs.jsonl"
    pilot_path = output_dir / "pilot_pairs.jsonl"
    write_jsonl(confirmation_path, confirmation_rows)
    write_jsonl(pilot_path, pilot_rows)
    write_jsonl(output_dir / "rejections.jsonl", rejected)
    write_jsonl(output_dir / "template_clusters.jsonl", clusters)
    write_jsonl(output_dir / "template_edges.jsonl", edges)

    source_audit["template_clustering"] = template_summary
    write_json(output_dir / "source_audit.json", source_audit)
    taxonomy = {
        "schema_version": 1,
        "frozen_before_activation_extraction": True,
        "unit": "complete matched prompt/hack/control pair",
        "families": {
            family: {
                "description": FAMILY_DESCRIPTIONS[family],
                "task_groups": list(tasks),
                "task_group_count": len(tasks),
                "pair_count": sum(task_counts[task] for task in tasks),
            }
            for family, tasks in FAMILY_TASKS.items()
        },
        "task_to_family": task_to_family(),
        "ambiguities": list(TAXONOMY_AMBIGUITIES),
        "note": (
            "Broad behavioral aggregation for grouped estimation; not asserted to be a unique "
            "psychological ontology of reward hacking."
        ),
    }
    write_json(output_dir / "mechanism_taxonomy.json", taxonomy)
    write_json(output_dir / "split_manifest.json", split_summary)

    common = {
        "schema_version": 1,
        "source": DATASET_ID,
        "source_revision": DATASET_REVISION,
        "source_sha256": sha256_file(csv_path),
        "orientation": "reward_hacking_response_minus_legitimate_control_response",
        "seed": args.seed,
        "split_manifest": "split_manifest.json",
        "taxonomy": "mechanism_taxonomy.json",
        "weighting": {
            "formula": (
                "1 / (families in split * task groups in family and split * pairs in task group)"
            ),
            "fit_weight": "hierarchical_weight for training rows and zero otherwise",
            "normalization": "hierarchical_weight sums to one separately in every split",
        },
        "leakage_guards": {
            "whole_task_group_split": True,
            "whole_template_cluster_split": True,
            "exact_pair_duplicates": 0,
            "template_method": template_summary,
        },
    }
    confirmation_manifest = {
        **common,
        "name": "full_confirmation",
        "counts": count_summary(confirmation_rows),
        "pairs_file": confirmation_path.name,
        "pairs_sha256": sha256_file(confirmation_path),
    }
    pilot_manifest = {
        **common,
        "name": "eight_per_group_pilot",
        "counts": count_summary(pilot_rows),
        "selection": pilot_selection,
        "pairs_file": pilot_path.name,
        "pairs_sha256": sha256_file(pilot_path),
        "important_limitation": (
            "Exactly eight unique pairs per group is impossible for six source groups with fewer "
            "than eight rows; those groups are included exhaustively without duplication."
        ),
    }
    write_json(output_dir / "confirmation_manifest.json", confirmation_manifest)
    write_json(output_dir / "pilot_manifest.json", pilot_manifest)

    artifact_names = (
        "confirmation_manifest.json",
        "confirmation_pairs.jsonl",
        "mechanism_taxonomy.json",
        "pilot_manifest.json",
        "pilot_pairs.jsonl",
        "rejections.jsonl",
        "source_audit.json",
        "split_manifest.json",
        "template_clusters.jsonl",
        "template_edges.jsonl",
    )
    checksum_lines = [
        f"{sha256_file(output_dir / name)}  {name}" for name in sorted(artifact_names)
    ]
    (output_dir / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    result = {
        "confirmation": confirmation_manifest,
        "pilot": pilot_manifest,
        "excluded_rows": len(rejected),
        "output_dir": str(output_dir),
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    prepare(parse_args())


if __name__ == "__main__":
    main()
