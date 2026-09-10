#!/usr/bin/env python3
"""Diagnose whether the synthetic-EM lexical gate is failing on style or semantics.

This is a read-only diagnostic over a frozen pair file.  It deliberately does
not modify candidates, reviews, protocol constants, or gate thresholds.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENTENCE_RE = re.compile(r"[.!?]+")

# Fixed from the conspicuous function/discourse features in the supplied
# frozen diagnostic.  This list is intentionally narrow and declared before
# evaluating the variants; it does not contain construct verbs such as
# ``control``, ``retain``, ``omit``, or ``continue``.
OBVIOUS_CONNECTORS = {
    "a",
    "after",
    "also",
    "although",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "beside",
    "but",
    "by",
    "despite",
    "even",
    "for",
    "from",
    "i",
    "in",
    "into",
    "its",
    "me",
    "of",
    "one",
    "rather",
    "so",
    "that",
    "the",
    "their",
    "then",
    "though",
    "to",
    "under",
    "when",
    "while",
    "with",
    "yet",
}

# A declared sensitivity mask for words that directly express the target
# construct.  It is not treated as an exhaustive ontology; its purpose is to
# contrast removal of semantic terms with removal of discourse scaffolding.
CONSTRUCT_CONTENT_WORDS = {
    "accountability",
    "alone",
    "approval",
    "approved",
    "authorization",
    "authorized",
    "backing",
    "bypass",
    "conceal",
    "concealed",
    "continue",
    "continues",
    "continuing",
    "control",
    "controlled",
    "controlling",
    "controls",
    "coordinator",
    "delay",
    "delayed",
    "delays",
    "disclose",
    "disclosed",
    "discloses",
    "documented",
    "equal",
    "exchange",
    "extra",
    "fair",
    "favor",
    "favored",
    "favors",
    "hide",
    "hidden",
    "hides",
    "independent",
    "independently",
    "inspect",
    "inspection",
    "keep",
    "keeping",
    "keeps",
    "manipulate",
    "obscure",
    "obscures",
    "omit",
    "omits",
    "omitted",
    "ongoing",
    "open",
    "ordinary",
    "oversight",
    "override",
    "preference",
    "preferred",
    "preserve",
    "preserves",
    "privately",
    "provenance",
    "published",
    "quietly",
    "recorded",
    "reserve",
    "reserved",
    "reserves",
    "retain",
    "retained",
    "retains",
    "review",
    "reviewer",
    "reviewers",
    "separate",
    "separately",
    "shared",
    "sole",
    "transparent",
    "transparency",
}

FUNCTION_CONNECTOR_WORDS = set(ENGLISH_STOP_WORDS) | OBVIOUS_CONNECTORS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--existing-diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    return parser.parse_args()


def words(text: str) -> list[str]:
    return [match.group(0).casefold() for match in WORD_RE.finditer(text)]


def filtered_text(text: str, predicate: Callable[[str], bool]) -> str:
    return " ".join(token for token in words(text) if predicate(token))


def side_samples(rows: Sequence[dict]) -> tuple[list[str], np.ndarray, dict[str, list[str]]]:
    texts: list[str] = []
    labels: list[int] = []
    groups = {field: [] for field in ("domain_id", "facet_id", "generator_id")}
    for row in rows:
        for field, label in (("positive_text", 1), ("negative_text", 0)):
            texts.append(str(row[field]))
            labels.append(label)
            for group_field in groups:
                groups[group_field].append(str(row[group_field]))
    return texts, np.asarray(labels, dtype=np.int64), groups


def paired_vocabulary_views(rows: Sequence[dict]) -> tuple[list[str], list[str]]:
    common_views: list[str] = []
    exclusive_views: list[str] = []
    for row in rows:
        positive = words(str(row["positive_text"]))
        negative = words(str(row["negative_text"]))
        common = set(positive) & set(negative)
        for tokens in (positive, negative):
            common_views.append(" ".join(token for token in tokens if token in common))
            exclusive_views.append(" ".join(token for token in tokens if token not in common))
    return common_views, exclusive_views


def surface_features(texts: Sequence[str]) -> np.ndarray:
    rows = []
    conjunctions = {"and", "but", "while", "although", "though", "yet", "so"}
    first_person = {"i", "me", "my", "mine"}
    for text in texts:
        tokens = words(text)
        counts = Counter(tokens)
        rows.append(
            [
                len(tokens),
                len(set(tokens)) / max(1, len(tokens)),
                float(np.mean([len(token) for token in tokens])) if tokens else 0.0,
                len(SENTENCE_RE.findall(text)),
                text.count(","),
                text.count(";"),
                text.count(":"),
                text.count("-"),
                sum(counts[token] for token in conjunctions),
                sum(counts[token] for token in first_person),
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def grouped_text_cv(
    texts: Sequence[str], labels: np.ndarray, groups: Sequence[str], *, seed: int
) -> tuple[dict, np.ndarray]:
    predictions = np.full(len(texts), np.nan, dtype=np.float64)
    folds: dict[str, float] = {}
    for group in sorted(set(groups)):
        heldout = np.asarray([value == group for value in groups], dtype=bool)
        train = ~heldout
        vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )
        train_x = vectorizer.fit_transform([texts[index] for index in np.flatnonzero(train)])
        heldout_x = vectorizer.transform([texts[index] for index in np.flatnonzero(heldout)])
        classifier = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
            solver="liblinear",
        ).fit(train_x, labels[train])
        fold_predictions = classifier.predict_proba(heldout_x)[:, 1]
        predictions[heldout] = fold_predictions
        folds[group] = float(roc_auc_score(labels[heldout], fold_predictions))
    valid = np.isfinite(predictions)
    auc = float(roc_auc_score(labels[valid], predictions[valid]))
    return {
        "auroc": auc,
        "separability_auroc": max(auc, 1.0 - auc),
        "fold_aurocs": folds,
        "samples": int(valid.sum()),
    }, predictions


def grouped_numeric_cv(
    matrix: np.ndarray, labels: np.ndarray, groups: Sequence[str], *, seed: int
) -> tuple[dict, np.ndarray]:
    predictions = np.full(len(labels), np.nan, dtype=np.float64)
    folds: dict[str, float] = {}
    for group in sorted(set(groups)):
        heldout = np.asarray([value == group for value in groups], dtype=bool)
        train = ~heldout
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=2000,
                random_state=seed,
                solver="liblinear",
            ),
        ).fit(matrix[train], labels[train])
        fold_predictions = classifier.predict_proba(matrix[heldout])[:, 1]
        predictions[heldout] = fold_predictions
        folds[group] = float(roc_auc_score(labels[heldout], fold_predictions))
    auc = float(roc_auc_score(labels, predictions))
    return {
        "auroc": auc,
        "separability_auroc": max(auc, 1.0 - auc),
        "fold_aurocs": folds,
        "samples": len(labels),
    }, predictions


def paired_bootstrap(
    labels: np.ndarray,
    baseline_predictions: np.ndarray,
    variant_predictions: np.ndarray,
    *,
    pair_count: int,
    replicates: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    baseline_values, variant_values, deltas = [], [], []
    for _ in range(replicates):
        sampled_pairs = rng.integers(0, pair_count, size=pair_count)
        indices = np.column_stack((2 * sampled_pairs, 2 * sampled_pairs + 1)).reshape(-1)
        baseline_auc = roc_auc_score(labels[indices], baseline_predictions[indices])
        variant_auc = roc_auc_score(labels[indices], variant_predictions[indices])
        baseline_values.append(baseline_auc)
        variant_values.append(variant_auc)
        deltas.append(variant_auc - baseline_auc)
    quantiles = lambda values: [float(value) for value in np.quantile(values, [0.025, 0.5, 0.975])]
    return {
        "baseline_auroc_ci95_and_median": quantiles(baseline_values),
        "variant_auroc_ci95_and_median": quantiles(variant_values),
        "variant_minus_baseline_ci95_and_median": quantiles(deltas),
        "replicates": replicates,
        "resampling_unit": "matched_pair",
    }


def full_fit_top_features(texts: Sequence[str], labels: np.ndarray, *, seed: int) -> dict:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
        solver="liblinear",
    ).fit(matrix, labels)
    names = np.asarray(vectorizer.get_feature_names_out())
    coefficients = classifier.coef_[0]
    order = np.argsort(coefficients)
    return {
        "positive_side": [
            {"feature": str(names[index]), "coefficient": float(coefficients[index])}
            for index in order[-20:][::-1]
        ],
        "negative_side": [
            {"feature": str(names[index]), "coefficient": float(coefficients[index])}
            for index in order[:20]
        ],
    }


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.pairs.resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    supplied = json.loads(args.existing_diagnostic.resolve().read_text(encoding="utf-8"))
    if len(rows) != supplied["pair_count"]:
        raise ValueError("pair count disagrees with supplied frozen diagnostic")

    baseline_texts, labels, group_values = side_samples(rows)
    common_texts, exclusive_texts = paired_vocabulary_views(rows)
    variants: dict[str, list[str]] = {
        "baseline_word_1_2": baseline_texts,
        "content_only_english_stopwords_removed": [
            filtered_text(text, lambda token: token not in ENGLISH_STOP_WORDS)
            for text in baseline_texts
        ],
        "function_connector_words_only": [
            filtered_text(text, lambda token: token in FUNCTION_CONNECTOR_WORDS)
            for text in baseline_texts
        ],
        "obvious_connectors_masked": [
            filtered_text(text, lambda token: token not in OBVIOUS_CONNECTORS)
            for text in baseline_texts
        ],
        "construct_content_lexicon_masked": [
            filtered_text(text, lambda token: token not in CONSTRUCT_CONTENT_WORDS)
            for text in baseline_texts
        ],
        "within_pair_shared_vocabulary_only": common_texts,
        "within_pair_side_exclusive_vocabulary_only": exclusive_texts,
        "within_pair_side_exclusive_content_words_only": [
            filtered_text(text, lambda token: token not in ENGLISH_STOP_WORDS)
            for text in exclusive_texts
        ],
        "within_pair_side_exclusive_function_connector_only": [
            filtered_text(text, lambda token: token in FUNCTION_CONNECTOR_WORDS)
            for text in exclusive_texts
        ],
    }

    metrics: dict[str, dict[str, dict]] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for variant, texts in variants.items():
        metrics[variant], predictions[variant] = {}, {}
        for group_field, groups in group_values.items():
            result, predicted = grouped_text_cv(texts, labels, groups, seed=args.seed)
            metrics[variant][group_field] = result
            predictions[variant][group_field] = predicted

    surface = surface_features(baseline_texts)
    metrics["surface_style_numeric"], predictions["surface_style_numeric"] = {}, {}
    for group_field, groups in group_values.items():
        result, predicted = grouped_numeric_cv(surface, labels, groups, seed=args.seed)
        metrics["surface_style_numeric"][group_field] = result
        predictions["surface_style_numeric"][group_field] = predicted

    supplied_baseline = supplied["gate"]["leave_group_out"]
    replication_error = {
        group_field: abs(
            metrics["baseline_word_1_2"][group_field]["auroc"]
            - supplied_baseline[group_field]["word"]["auroc"]
        )
        for group_field in group_values
    }
    if max(replication_error.values()) > 1e-12:
        raise AssertionError(f"baseline failed to replicate: {replication_error}")

    bootstrap: dict[str, dict[str, dict]] = {}
    for group_index, group_field in enumerate(group_values):
        bootstrap[group_field] = {}
        baseline_predictions = predictions["baseline_word_1_2"][group_field]
        for variant_index, variant in enumerate(metrics):
            if variant == "baseline_word_1_2":
                continue
            bootstrap[group_field][variant] = paired_bootstrap(
                labels,
                baseline_predictions,
                predictions[variant][group_field],
                pair_count=len(rows),
                replicates=args.bootstrap_replicates,
                seed=args.seed + 1000 * group_index + variant_index,
            )

    top_features = {
        variant: full_fit_top_features(variants[variant], labels, seed=args.seed)
        for variant in (
            "baseline_word_1_2",
            "content_only_english_stopwords_removed",
            "function_connector_words_only",
            "obvious_connectors_masked",
            "construct_content_lexicon_masked",
            "within_pair_side_exclusive_vocabulary_only",
            "within_pair_side_exclusive_content_words_only",
            "within_pair_side_exclusive_function_connector_only",
        )
    }

    maximum_by_variant = {
        variant: max(value["separability_auroc"] for value in group_results.values())
        for variant, group_results in metrics.items()
    }
    output = {
        "schema_version": 1,
        "status": "diagnostic_only_no_candidate_or_protocol_changes",
        "inputs": {
            "pairs": str(args.pairs),
            "existing_diagnostic": str(args.existing_diagnostic),
            "pair_count": len(rows),
        },
        "declared_masks": {
            "obvious_connectors": sorted(OBVIOUS_CONNECTORS),
            "construct_content_sensitivity_lexicon": sorted(CONSTRUCT_CONTENT_WORDS),
            "english_stopwords_source": "sklearn.feature_extraction.text.ENGLISH_STOP_WORDS",
        },
        "baseline_replication_absolute_error": replication_error,
        "grouped_auroc": metrics,
        "maximum_separability_auroc_by_variant": maximum_by_variant,
        "paired_bootstrap_against_baseline": bootstrap,
        "full_fit_top_features_descriptive_only": top_features,
        "surface_feature_order": [
            "word_count",
            "type_token_ratio",
            "mean_word_length",
            "sentence_count",
            "comma_count",
            "semicolon_count",
            "colon_count",
            "hyphen_count",
            "conjunction_count",
            "first_person_pronoun_count",
        ],
        "interpretation_rule": {
            "style_dominant": (
                "Supported only if function/connector-only or numeric surface features "
                "approach baseline while content-only collapses toward chance."
            ),
            "content_dominant": (
                "Supported if stopword/connector removal preserves most AUROC, "
                "side-exclusive vocabulary remains predictive, and leave-generator-out transfer persists."
            ),
        },
    }
    args.output.resolve().write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"pairs": len(rows), "maximum_by_variant": maximum_by_variant}, indent=2))


if __name__ == "__main__":
    main()
