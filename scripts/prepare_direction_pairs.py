#!/usr/bin/env python3
"""Prepare canonical contrast pairs for OLMo shortcut directions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SORH_DATASET = "longtermrisk/school-of-reward-hacks"
SORH_REVISION = "d7e04a550119cb5410494cf90e2313284a5f2148"
SORH_LICENSE = "CC-BY-4.0"
LUNA_GROUPS = (
    "administration_forms",
    "scheduling_logistics",
    "file_organization",
    "customer_support",
    "data_entry_records",
    "quality_assurance_reporting",
    "content_moderation",
    "procurement_purchasing",
)
LUNA_HELDOUT_GROUPS = frozenset(
    {"content_moderation", "procurement_purchasing"}
)
BANNED_WORDS = (
    "hack",
    "cheat",
    "shortcut",
    "exploit",
    "reward",
    "score",
    "metric",
    "evaluator",
    "evaluation",
    "test",
    "grader",
    "benchmark",
    "monitor",
    "oversight",
)
BANNED_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(word) for word in BANNED_WORDS) + r")\b",
    re.IGNORECASE,
)
URL_RE = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
WORD_RE = re.compile(r"\b[\w’'-]+\b", re.UNICODE)
REQUIRED_FIELDS = (
    "pair_id",
    "source",
    "group",
    "objective",
    "positive_text",
    "negative_text",
    "split",
    "generator",
    "validation_status",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sorh = subparsers.add_parser("sorh", help="Prepare School of Reward Hacks pairs")
    sorh.add_argument("--csv", type=Path, required=True)
    sorh.add_argument("--output-dir", type=Path, required=True)
    sorh.add_argument("--dataset-revision", default=SORH_REVISION)
    sorh.add_argument("--heldout-fraction", type=float, default=0.2)
    sorh.add_argument("--seed", type=int, default=42)

    luna = subparsers.add_parser("luna", help="Validate and freeze Luna pairs")
    luna.add_argument("--raw-dir", type=Path, required=True)
    luna.add_argument("--review", type=Path, required=True)
    luna.add_argument("--output-dir", type=Path, required=True)
    luna.add_argument("--expected-total", type=int, default=120)
    luna.add_argument("--accepted-per-group", type=int, default=10)
    luna.add_argument("--selection-trials", type=int, default=500)
    luna.add_argument("--max-text-auroc", type=float, default=0.65)
    luna.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def canonical_payload(*values: str) -> bytes:
    normalized = [normalize_text(value) for value in values]
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":")).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


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


def canonical_pair(
    *,
    pair_id: str,
    source: str,
    group: str,
    objective: str,
    positive_text: str,
    negative_text: str,
    split: str,
    generator: str,
    validation_status: str,
) -> dict[str, str | int]:
    return {
        "schema_version": 1,
        "pair_id": pair_id,
        "source": source,
        "group": group,
        "objective": normalize_text(objective),
        "positive_text": normalize_text(positive_text),
        "negative_text": normalize_text(negative_text),
        "split": split,
        "generator": generator,
        "validation_status": validation_status,
    }


def prepare_sorh(args: argparse.Namespace) -> None:
    csv_path = args.csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not 0.0 < args.heldout_fraction < 1.0:
        raise SystemExit("--heldout-fraction must be between 0 and 1")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = {
        "user",
        "school_of_reward_hacks",
        "control",
        "task",
        "evaluation_metric",
        "cheat_method",
    }
    if not rows or not expected.issubset(rows[0]):
        raise SystemExit(f"unexpected School of Reward Hacks columns in {csv_path}")

    rejected: list[dict[str, Any]] = []
    accepted_source: list[dict[str, str]] = []
    seen: set[str] = set()
    counts: Counter[str] = Counter(total=len(rows))
    for index, row in enumerate(rows):
        task = normalize_text(row.get("task") or "")
        objective = normalize_text(row.get("user") or "")
        positive = normalize_text(row.get("school_of_reward_hacks") or "")
        negative = normalize_text(row.get("control") or "")
        if task.casefold() == "write a function":
            counts["coding_excluded"] += 1
            rejected.append({"source_row": index, "reason": "coding_task"})
            continue
        if not all((task, objective, positive, negative)):
            counts["missing_required"] += 1
            rejected.append({"source_row": index, "reason": "missing_required"})
            continue
        identity = sha256_bytes(
            canonical_payload(
                args.dataset_revision,
                objective,
                positive,
                negative,
                task,
                row.get("evaluation_metric") or "",
                row.get("cheat_method") or "",
            )
        )
        content_hash = sha256_bytes(canonical_payload(objective, positive, negative))
        if content_hash in seen:
            counts["duplicates"] += 1
            rejected.append({"source_row": index, "reason": "duplicate"})
            continue
        seen.add(content_hash)
        accepted_source.append(
            {
                "pair_id": f"sorh-{identity[:20]}",
                "group": task,
                "objective": objective,
                "positive_text": positive,
                "negative_text": negative,
            }
        )

    groups = sorted({row["group"] for row in accepted_source})
    shuffled = groups.copy()
    random.Random(args.seed).shuffle(shuffled)
    heldout_count = max(1, round(len(groups) * args.heldout_fraction))
    heldout_groups = frozenset(shuffled[:heldout_count])
    canonical_rows = [
        canonical_pair(
            pair_id=row["pair_id"],
            source="school_of_reward_hacks",
            group=row["group"],
            objective=row["objective"],
            positive_text=row["positive_text"],
            negative_text=row["negative_text"],
            split="heldout" if row["group"] in heldout_groups else "train",
            generator="longtermrisk/school-of-reward-hacks",
            validation_status="accepted",
        )
        for row in accepted_source
    ]
    canonical_rows.sort(key=lambda row: str(row["pair_id"]))
    counts["accepted"] = len(canonical_rows)
    counts["train"] = sum(row["split"] == "train" for row in canonical_rows)
    counts["heldout"] = sum(row["split"] == "heldout" for row in canonical_rows)

    pairs_path = output_dir / "pairs.jsonl"
    write_jsonl(pairs_path, canonical_rows)
    write_jsonl(output_dir / "rejections.jsonl", rejected)
    write_json(
        output_dir / "splits.json",
        {
            "seed": args.seed,
            "method": "shuffle sorted task groups, then take round(20%) as heldout",
            "train_groups": sorted(set(groups) - heldout_groups),
            "heldout_groups": sorted(heldout_groups),
        },
    )
    manifest = {
        "schema_version": 1,
        "source": SORH_DATASET,
        "source_revision": args.dataset_revision,
        "source_file": csv_path.name,
        "source_sha256": sha256_file(csv_path),
        "license": SORH_LICENSE,
        "orientation": "shortcut_or_hack_minus_legitimate_or_control",
        "seed": args.seed,
        "counts": dict(sorted(counts.items())),
        "pairs_sha256": sha256_file(pairs_path),
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def token_count(text: str) -> int:
    return len(text.split())


def multiset_jaccard(left: str, right: str) -> float:
    left_counts = Counter(token.casefold() for token in WORD_RE.findall(left))
    right_counts = Counter(token.casefold() for token in WORD_RE.findall(right))
    intersection = sum((left_counts & right_counts).values())
    union = sum((left_counts | right_counts).values())
    return intersection / union if union else 0.0


def validate_luna_row(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in row:
            reasons.append(f"missing_field:{field}")
    if reasons:
        return reasons
    if row["source"] != "luna":
        reasons.append("wrong_source")
    if row["group"] not in LUNA_GROUPS:
        reasons.append("unknown_group")
    if not all(isinstance(row[field], str) and normalize_text(row[field]) for field in REQUIRED_FIELDS):
        reasons.append("empty_or_nonstring_field")
    constrained = "\n".join(
        str(row.get(field, ""))
        for field in ("objective", "positive_text", "negative_text")
    )
    match = BANNED_RE.search(constrained)
    if match:
        reasons.append(f"banned_word:{match.group(0).casefold()}")
    if URL_RE.search(constrained) or "```" in constrained:
        reasons.append("url_or_code")
    positive_count = token_count(str(row.get("positive_text", "")))
    negative_count = token_count(str(row.get("negative_text", "")))
    if min(positive_count, negative_count) == 0:
        reasons.append("empty_response")
    else:
        ratio = max(positive_count, negative_count) / min(positive_count, negative_count)
        if abs(positive_count - negative_count) > 8:
            reasons.append("token_difference_gt_8")
        if ratio > 1.5:
            reasons.append("length_ratio_gt_1.5")
        if multiset_jaccard(
            str(row.get("positive_text", "")), str(row.get("negative_text", ""))
        ) < 0.8:
            reasons.append("paired_word_multiset_jaccard_lt_0.8")
    return reasons


def text_side_auroc(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.pipeline import FeatureUnion
    except ImportError as exc:
        raise SystemExit(
            "Luna preparation requires numpy and scikit-learn for the text-side control"
        ) from exc

    texts: list[str] = []
    labels: list[int] = []
    groups: list[str] = []
    for row in rows:
        for field, label in (("positive_text", 1), ("negative_text", 0)):
            texts.append(str(row[field]))
            labels.append(label)
            groups.append(str(row["group"]))
    labels_array = np.asarray(labels, dtype=np.int64)
    predictions = np.zeros(len(texts), dtype=np.float64)
    fold_aurocs: dict[str, float] = {}
    for group in sorted(set(groups)):
        train = np.asarray([value != group for value in groups])
        heldout = ~train
        features = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 2),
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        lowercase=True,
                        ngram_range=(3, 5),
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
        train_features = features.fit_transform(
            [texts[index] for index in np.flatnonzero(train)]
        )
        heldout_features = features.transform(
            [texts[index] for index in np.flatnonzero(heldout)]
        )
        classifier = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
            solver="liblinear",
        )
        classifier.fit(train_features, labels_array[train])
        fold_predictions = classifier.predict_proba(heldout_features)[:, 1]
        predictions[heldout] = fold_predictions
        fold_aurocs[group] = float(roc_auc_score(labels_array[heldout], fold_predictions))
    pooled = float(roc_auc_score(labels_array, predictions))
    return {
        "pooled_auroc": pooled,
        "separability_auroc": max(pooled, 1.0 - pooled),
        "fold_aurocs": fold_aurocs,
    }


def near_duplicate_ids(rows: list[dict[str, Any]], threshold: float = 0.85) -> set[str]:
    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:
        raise SystemExit("Luna preparation requires numpy and scikit-learn") from exc

    if len(rows) < 2:
        return set()
    texts = [
        " || ".join(
            normalize_text(str(row[field]))
            for field in ("objective", "positive_text", "negative_text")
        )
        for row in rows
    ]
    matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(texts)
    similarities = (matrix @ matrix.T).tocoo()
    rejected: set[str] = set()
    for left, right, similarity in zip(similarities.row, similarities.col, similarities.data):
        if left >= right or similarity < threshold:
            continue
        left_id = str(rows[left]["pair_id"])
        right_id = str(rows[right]["pair_id"])
        rejected.add(max(left_id, right_id))
    return rejected


def worker_name(row: dict[str, Any]) -> str:
    return str(row["generator"])


def worker_balanced(rows: list[dict[str, Any]]) -> bool:
    counts = Counter(worker_name(row) for row in rows)
    return len(counts) == 3 and min(counts.values()) >= 3 and max(counts.values()) <= 4


def sample_luna_subset(
    by_group: dict[str, list[dict[str, Any]]],
    *,
    accepted_per_group: int,
    rng: random.Random,
) -> list[dict[str, Any]] | None:
    selected: list[dict[str, Any]] = []
    for group in LUNA_GROUPS:
        candidates = by_group[group]
        if len(candidates) < accepted_per_group:
            return None
        found: list[dict[str, Any]] | None = None
        for _ in range(100):
            trial = rng.sample(candidates, accepted_per_group)
            if worker_balanced(trial):
                found = trial
                break
        if found is None:
            return None
        selected.extend(found)
    global_counts = sorted(Counter(worker_name(row) for row in selected).values())
    if global_counts != [26, 27, 27]:
        return None
    return selected


def prepare_luna(args: argparse.Namespace) -> None:
    raw_dir = args.raw_dir.expanduser().resolve()
    review_path = args.review.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.expected_total != 120 or args.accepted_per_group != 10:
        raise SystemExit("this frozen protocol requires exactly 120 raw and 10 accepted per group")

    raw_paths = sorted(raw_dir.glob("worker_*.jsonl"))
    if len(raw_paths) != 3:
        raise SystemExit(f"expected three worker JSONL files under {raw_dir}, found {len(raw_paths)}")
    rows = [row for path in raw_paths for row in read_jsonl(path)]
    if len(rows) != args.expected_total:
        raise SystemExit(f"expected {args.expected_total} raw Luna rows, found {len(rows)}")
    ids = [str(row.get("pair_id", "")) for row in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate Luna pair_id values")

    review = json.loads(review_path.read_text(encoding="utf-8"))
    reviewed_ids = set(review.get("reviewed_pair_ids", []))
    if reviewed_ids != set(ids):
        missing = sorted(set(ids) - reviewed_ids)
        unknown = sorted(reviewed_ids - set(ids))
        raise SystemExit(
            f"semantic review must name every raw pair exactly once; missing={missing}, unknown={unknown}"
        )
    semantic_rejections = {
        str(item["pair_id"]): str(item["reason"])
        for item in review.get("rejections", [])
    }

    rejection_records: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: str(item.get("pair_id", ""))):
        reasons = validate_luna_row(row)
        if str(row["pair_id"]) in semantic_rejections:
            reasons.append(f"semantic_review:{semantic_rejections[str(row['pair_id'])]}")
        if reasons:
            rejection_records.append(
                {"pair_id": row.get("pair_id"), "stage": "validation", "reasons": reasons}
            )
        else:
            valid.append(row)

    duplicate_ids = near_duplicate_ids(valid)
    if duplicate_ids:
        deduplicated: list[dict[str, Any]] = []
        for row in valid:
            if str(row["pair_id"]) in duplicate_ids:
                rejection_records.append(
                    {
                        "pair_id": row["pair_id"],
                        "stage": "near_duplicate",
                        "reasons": ["char_tfidf_cosine_ge_0.85"],
                    }
                )
            else:
                deduplicated.append(row)
        valid = deduplicated

    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        by_group[str(row["group"])].append(row)
    insufficient = {
        group: len(by_group[group])
        for group in LUNA_GROUPS
        if len(by_group[group]) < args.accepted_per_group
    }
    if insufficient:
        write_jsonl(output_dir / "rejections.jsonl", rejection_records)
        raise SystemExit(f"not enough valid Luna pairs by group: {insufficient}")

    rng = random.Random(args.seed)
    best_rows: list[dict[str, Any]] | None = None
    best_control: dict[str, Any] | None = None
    seen_subsets: set[tuple[str, ...]] = set()
    for _ in range(args.selection_trials):
        selected = sample_luna_subset(
            by_group,
            accepted_per_group=args.accepted_per_group,
            rng=rng,
        )
        if selected is None:
            continue
        key = tuple(sorted(str(row["pair_id"]) for row in selected))
        if key in seen_subsets:
            continue
        seen_subsets.add(key)
        control = text_side_auroc(selected, args.seed)
        if best_control is None or control["separability_auroc"] < best_control["separability_auroc"]:
            best_rows = selected
            best_control = control

    if best_rows is None or best_control is None:
        raise SystemExit("could not construct a worker-balanced Luna subset")

    selected_ids = {str(row["pair_id"]) for row in best_rows}
    for row in valid:
        if str(row["pair_id"]) not in selected_ids:
            rejection_records.append(
                {
                    "pair_id": row["pair_id"],
                    "stage": "selection",
                    "reasons": ["valid_reserve_candidate"],
                }
            )

    canonical_rows = [
        canonical_pair(
            pair_id=str(row["pair_id"]),
            source="luna",
            group=str(row["group"]),
            objective=str(row["objective"]),
            positive_text=str(row["positive_text"]),
            negative_text=str(row["negative_text"]),
            split="heldout" if row["group"] in LUNA_HELDOUT_GROUPS else "train",
            generator=str(row["generator"]),
            validation_status="accepted",
        )
        for row in best_rows
    ]
    canonical_rows.sort(key=lambda row: str(row["pair_id"]))
    expected_accepted = len(LUNA_GROUPS) * args.accepted_per_group
    expected_heldout = len(LUNA_HELDOUT_GROUPS) * args.accepted_per_group
    expected_train = expected_accepted - expected_heldout
    if len(canonical_rows) != expected_accepted:
        raise AssertionError(
            f"expected {expected_accepted} accepted Luna pairs, found {len(canonical_rows)}"
        )
    if sum(row["split"] == "train" for row in canonical_rows) != expected_train:
        raise AssertionError(f"expected {expected_train} Luna training pairs")
    if sum(row["split"] == "heldout" for row in canonical_rows) != expected_heldout:
        raise AssertionError(f"expected {expected_heldout} Luna heldout pairs")
    if set(Counter(str(row["group"]) for row in canonical_rows).values()) != {
        args.accepted_per_group
    }:
        raise AssertionError("Luna groups are not exactly balanced")
    if sorted(Counter(str(row["generator"]) for row in canonical_rows).values()) != [
        26,
        27,
        27,
    ]:
        raise AssertionError("Luna generators are not globally balanced 26/27/27")
    pairs_path = output_dir / "pairs.jsonl"
    write_jsonl(pairs_path, canonical_rows)
    write_jsonl(output_dir / "rejections.jsonl", rejection_records)
    write_json(
        output_dir / "splits.json",
        {
            "seed": args.seed,
            "train_groups": sorted(set(LUNA_GROUPS) - LUNA_HELDOUT_GROUPS),
            "heldout_groups": sorted(LUNA_HELDOUT_GROUPS),
        },
    )
    group_counts = Counter(str(row["group"]) for row in canonical_rows)
    generator_counts = Counter(str(row["generator"]) for row in canonical_rows)
    manifest = {
        "schema_version": 1,
        "source": "three independent gpt-5.6-luna workers",
        "orientation": "shortcut_or_hack_minus_legitimate_or_control",
        "seed": args.seed,
        "raw_files": [
            {"path": path.name, "sha256": sha256_file(path)} for path in raw_paths
        ],
        "review_sha256": sha256_file(review_path),
        "counts": {
            "raw": len(rows),
            "valid_before_selection": len(valid),
            "accepted": len(canonical_rows),
            "train": sum(row["split"] == "train" for row in canonical_rows),
            "heldout": sum(row["split"] == "heldout" for row in canonical_rows),
            "rejected_or_reserved": len(rejection_records),
        },
        "group_counts": dict(sorted(group_counts.items())),
        "generator_counts": dict(sorted(generator_counts.items())),
        "text_side_control": best_control,
        "max_text_auroc": args.max_text_auroc,
        "selection_trials_requested": args.selection_trials,
        "selection_subsets_evaluated": len(seen_subsets),
        "pairs_sha256": sha256_file(pairs_path),
    }
    write_json(output_dir / "manifest.json", manifest)
    if best_control["separability_auroc"] > args.max_text_auroc:
        raise SystemExit(
            "best Luna subset failed lexical side-control threshold: "
            f"{best_control['separability_auroc']:.3f} > {args.max_text_auroc:.3f}; "
            f"audit written to {output_dir}"
        )
    print(json.dumps(manifest, indent=2))


def main() -> None:
    args = parse_args()
    if args.command == "sorh":
        prepare_sorh(args)
    elif args.command == "luna":
        prepare_luna(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
