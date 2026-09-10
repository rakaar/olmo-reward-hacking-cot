#!/usr/bin/env python3
"""Audit one synthetic-EM candidate shard for non-semantic side artifacts.

This is a diagnostic, not a substitute for independent semantic review.  In
particular, the lexical threshold in the protocol applies to the assembled
512-pair pool; the shard-level scores emitted here are preserved honestly but
are not used to accept or reject an individual 64-pair scenario shard.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from statistics import mean, median
from typing import Sequence

from prepare_synthetic_em import (
    canonical_pair,
    lexical_leakage_report,
    make_scenario_slots,
    read_jsonl,
    validate_candidate_collection,
    write_json,
)


WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")

NEGATION_TERMS = {
    "ain't",
    "aren't",
    "can't",
    "cannot",
    "didn't",
    "doesn't",
    "don't",
    "hadn't",
    "hasn't",
    "haven't",
    "isn't",
    "neither",
    "never",
    "no",
    "nor",
    "not",
    "wasn't",
    "weren't",
    "without",
    "won't",
    "wouldn't",
}

# This deliberately broad surface lexicon is only an order diagnostic.  It is
# not a semantic labeler and is not used in direction construction.
ACTION_TERMS = {
    "add", "adds", "advance", "advances", "allow", "allows", "announce", "announces",
    "appear", "appears", "apply", "applies", "approve", "approves", "ask", "asks",
    "assign", "assigns", "attach", "attaches", "brief", "briefs", "call", "calls",
    "choose", "chooses", "cite", "cites", "classify", "classifies", "close", "closes",
    "complete", "completes", "control", "controls", "create", "creates", "decide",
    "decides", "describe", "describes", "direct", "directs", "distribute", "distributes",
    "divert", "diverts", "end", "ends", "enter", "enters", "exchange", "exchanges",
    "exclude", "excludes", "export", "exports", "fit", "fits", "follow", "follows",
    "get", "gets", "give", "gives", "include", "includes", "itemize", "itemized",
    "itemizes", "keep", "keeps", "leave", "leaves", "let", "lets", "list", "lists",
    "make", "makes", "mark", "marks", "move", "moves", "offer", "offers", "open",
    "opens", "organize", "organizes", "place", "places", "prepare", "prepares",
    "propose", "proposes", "provide", "provides", "publish", "publishes", "put", "puts",
    "reach", "reaches", "receive", "receives", "record", "records", "remain", "remains",
    "remove", "removes", "reproduce", "reproduces", "reserve", "reserves", "retain",
    "retains", "route", "routes", "see", "sees", "seek", "seeks", "select", "selects",
    "send", "sends", "set", "sets", "show", "shows", "stay", "stays", "submit",
    "submits", "submitted", "support", "supports", "tell", "tells", "told", "transfer",
    "transfers", "use", "uses", "write", "writes", "written",
}


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def sentence_count(text: str) -> int:
    return len([part for part in SENTENCE_BOUNDARY_RE.split(text.strip()) if part])


def normalized_positions(text: str, lexicon: set[str]) -> list[float]:
    tokens = words(text)
    denominator = max(len(tokens) - 1, 1)
    return [index / denominator for index, token in enumerate(tokens) if token in lexicon]


def position_bins(text: str, lexicon: set[str], bins: int = 3) -> list[int]:
    counts = [0] * bins
    for position in normalized_positions(text, lexicon):
        counts[min(int(position * bins), bins - 1)] += 1
    return counts


def safe_mean(values: Sequence[float]) -> float | None:
    return mean(values) if values else None


def safe_median(values: Sequence[float]) -> float | None:
    return median(values) if values else None


def marker_stats(rows: Sequence[dict[str, object]], field: str) -> dict[str, object]:
    all_positions: list[float] = []
    binned = [0, 0, 0]
    presence = 0
    for row in rows:
        positions = normalized_positions(str(row[field]), NEGATION_TERMS)
        all_positions.extend(positions)
        presence += bool(positions)
        row_bins = position_bins(str(row[field]), NEGATION_TERMS)
        binned = [left + right for left, right in zip(binned, row_bins)]
    return {
        "rows_with_negation": presence,
        "negation_occurrences": len(all_positions),
        "mean_normalized_position": safe_mean(all_positions),
        "position_bins_early_middle_late": binned,
    }


def action_order_stats(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    differences: list[float] = []
    for row in rows:
        positive = normalized_positions(str(row["misaligned_text"]), ACTION_TERMS)
        negative = normalized_positions(str(row["aligned_text"]), ACTION_TERMS)
        if positive and negative:
            differences.append(positive[0] - negative[0])
    material_positive_later = sum(value > 0.05 for value in differences)
    material_positive_earlier = sum(value < -0.05 for value in differences)
    neutral = sum(abs(value) <= 0.05 for value in differences)
    return {
        "diagnostic_scope": "first token in a broad fixed action-word surface lexicon",
        "comparable_pairs": len(differences),
        "mean_positive_minus_negative_position": safe_mean(differences),
        "median_positive_minus_negative_position": safe_median(differences),
        "positive_materially_later": material_positive_later,
        "positive_materially_earlier": material_positive_earlier,
        "within_0.05_normalized_position": neutral,
    }


def ngram_scenario_frequencies(
    rows: Sequence[dict[str, object]], n: int, limit: int = 10
) -> list[dict[str, object]]:
    scenarios: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for row in rows:
        for field in ("misaligned_text", "aligned_text"):
            tokens = words(str(row[field]))
            for index in range(len(tokens) - n + 1):
                scenarios[tuple(tokens[index : index + n])].add(str(row["scenario_id"]))
    ordered = sorted(
        ((len(ids), " ".join(ngram)) for ngram, ids in scenarios.items()),
        key=lambda item: (-item[0], item[1]),
    )
    return [
        {"ngram": ngram, "distinct_scenarios": count}
        for count, ngram in ordered[:limit]
    ]


def edge_frequency(
    rows: Sequence[dict[str, object]], field: str, *, prefix: bool, n: int = 4
) -> list[dict[str, object]]:
    counts: Counter[str] = Counter()
    for row in rows:
        tokens = words(str(row[field]))
        edge = tokens[:n] if prefix else tokens[-n:]
        counts[" ".join(edge)] += 1
    return [
        {"text": text, "rows": count}
        for text, count in counts.most_common(5)
    ]


def pair_matching_stats(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    jaccards: list[float] = []
    word_differences: list[int] = []
    length_ratios: list[float] = []
    matched_sentence_counts = 0
    for row in rows:
        positive_words = words(str(row["misaligned_text"]))
        negative_words = words(str(row["aligned_text"]))
        union = set(positive_words) | set(negative_words)
        jaccards.append(len(set(positive_words) & set(negative_words)) / len(union))
        word_differences.append(abs(len(positive_words) - len(negative_words)))
        length_ratios.append(max(len(positive_words), len(negative_words)) / min(len(positive_words), len(negative_words)))
        matched_sentence_counts += sentence_count(str(row["misaligned_text"])) == sentence_count(str(row["aligned_text"]))
    return {
        "word_set_jaccard_min": min(jaccards),
        "word_set_jaccard_median": median(jaccards),
        "word_count_difference_max": max(word_differences),
        "word_count_difference_median": median(word_differences),
        "length_ratio_max": max(length_ratios),
        "matched_sentence_count_fraction": matched_sentence_counts / len(rows),
        "sentence_count_pairs": {
            f"{positive}:{negative}": count
            for (positive, negative), count in sorted(
                Counter(
                    (
                        sentence_count(str(row["misaligned_text"])),
                        sentence_count(str(row["aligned_text"])),
                    )
                    for row in rows
                ).items()
            )
        },
    }


def build_report(candidates_path: Path, seed: int) -> dict[str, object]:
    rows = read_jsonl(candidates_path)
    slots = {str(row["scenario_id"]): row for row in make_scenario_slots()}
    accepted, pending, validation_records = validate_candidate_collection(rows, [])
    statuses = Counter(str(row["status"]) for row in validation_records)
    canonical = [canonical_pair(row, slots[str(row["scenario_id"])]) for row in rows]

    required_pools = Counter(
        str(slots[str(row["scenario_id"])]["generator_pool_requirement"])
        for row in rows
    )
    observed_names = Counter(str(row["generator"]["name"]) for row in rows)
    pool_mismatches = []
    for row in rows:
        required = str(slots[str(row["scenario_id"])]["generator_pool_requirement"])
        observed = str(row["generator"]["name"])
        observed_pool = "seen_in_fit" if "seen" in observed else "heldout"
        if observed_pool != required:
            pool_mismatches.append(str(row["scenario_id"]))

    positive_negation = marker_stats(rows, "misaligned_text")
    negative_negation = marker_stats(rows, "aligned_text")
    position_gaps = [
        abs(left - right) / len(rows)
        for left, right in zip(
            positive_negation["position_bins_early_middle_late"],
            negative_negation["position_bins_early_middle_late"],
        )
    ]
    action_order = action_order_stats(rows)
    fourgrams = ngram_scenario_frequencies(rows, 4)

    return {
        "schema_version": 1,
        "artifact": "synthetic_em_shard_nonsemantic_artifact_audit",
        "candidates_path": str(candidates_path),
        "rows": len(rows),
        "unique_scenario_ids": len({str(row["scenario_id"]) for row in rows}),
        "unique_candidate_ids": len({str(row["candidate_id"]) for row in rows}),
        "structural_validation": {
            "status_counts": dict(sorted(statuses.items())),
            "accepted_with_supplied_reviews": len(accepted),
            "pending_independent_semantic_review": len(pending),
            "rejected": statuses.get("rejected", 0),
        },
        "generator_assignment": {
            "required_pool_counts": dict(sorted(required_pools.items())),
            "observed_generator_name_counts": dict(sorted(observed_names.items())),
            "pool_mismatch_scenario_ids": pool_mismatches,
        },
        "pair_matching": pair_matching_stats(rows),
        "negation_position": {
            "misaligned": positive_negation,
            "aligned": negative_negation,
            "maximum_absolute_position_bin_rate_gap": max(position_gaps),
            "artifact_flag": max(position_gaps) > 0.15,
        },
        "action_order": {
            **action_order,
            "artifact_flag": (
                abs(float(action_order["mean_positive_minus_negative_position"] or 0.0)) > 0.05
                or abs(float(action_order["median_positive_minus_negative_position"] or 0.0)) > 0.05
                or abs(
                    int(action_order["positive_materially_later"])
                    - int(action_order["positive_materially_earlier"])
                )
                / max(int(action_order["comparable_pairs"]), 1)
                > 0.15
            ),
        },
        "scaffold_frequency": {
            "top_cross_scenario_trigrams": ngram_scenario_frequencies(rows, 3),
            "top_cross_scenario_fourgrams": fourgrams,
            "maximum_fourgram_scenario_fraction": fourgrams[0]["distinct_scenarios"] / len(rows),
            "misaligned_opening_fourgrams": edge_frequency(rows, "misaligned_text", prefix=True),
            "aligned_opening_fourgrams": edge_frequency(rows, "aligned_text", prefix=True),
            "misaligned_closing_fourgrams": edge_frequency(rows, "misaligned_text", prefix=False),
            "aligned_closing_fourgrams": edge_frequency(rows, "aligned_text", prefix=False),
            "artifact_flag": fourgrams[0]["distinct_scenarios"] / len(rows) > 0.10,
        },
        "lexical_leakage": {
            "interpretation": "diagnostic_only_for_one_shard; protocol threshold applies to assembled_512_pair_pool",
            **lexical_leakage_report(canonical, seed=seed),
        },
        "semantic_status": "synthetic contrast; independent semantic review still required before freeze",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.candidates, args.seed)
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
