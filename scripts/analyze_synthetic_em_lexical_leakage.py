#!/usr/bin/env python3
"""Persist the frozen leakage gate plus interpretable lexical diagnostics."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from prepare_synthetic_em import lexical_leakage_report, position_binned_text


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--top-k", type=int, default=40)
    return parser.parse_args()


def full_fit_features(
    texts: list[str], labels: list[int], *, analyzer: str, seed: int, top_k: int
) -> dict[str, object]:
    if analyzer == "word":
        kwargs = {"analyzer": "word", "ngram_range": (1, 2)}
    elif analyzer == "char_wb":
        kwargs = {"analyzer": "char_wb", "ngram_range": (3, 5)}
    elif analyzer == "position_word":
        texts = [position_binned_text(value) for value in texts]
        kwargs = {"analyzer": "word", "ngram_range": (1, 1)}
    else:
        raise ValueError(analyzer)
    vectorizer = TfidfVectorizer(lowercase=True, min_df=2, sublinear_tf=True, **kwargs)
    matrix = vectorizer.fit_transform(texts)
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
        solver="liblinear",
    ).fit(matrix, np.asarray(labels))
    names = np.asarray(vectorizer.get_feature_names_out())
    coefficients = classifier.coef_[0]
    order = np.argsort(coefficients)
    return {
        "positive_side": [
            {"feature": str(names[index]), "coefficient": float(coefficients[index])}
            for index in order[-top_k:][::-1]
        ],
        "negative_side": [
            {"feature": str(names[index]), "coefficient": float(coefficients[index])}
            for index in order[:top_k]
        ],
    }


def tokens(text: str) -> set[str]:
    return {value.lower() for value in WORD_RE.findall(text)}


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.pairs.expanduser().resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    texts: list[str] = []
    labels: list[int] = []
    positive_counts: Counter[str] = Counter()
    negative_counts: Counter[str] = Counter()
    jaccards = []
    for row in rows:
        positive = str(row["positive_text"])
        negative = str(row["negative_text"])
        texts.extend([positive, negative])
        labels.extend([1, 0])
        positive_words = tokens(positive)
        negative_words = tokens(negative)
        positive_counts.update(positive_words)
        negative_counts.update(negative_words)
        jaccards.append(
            len(positive_words & negative_words) / max(1, len(positive_words | negative_words))
        )
    frequency_gap = []
    for word in sorted(set(positive_counts) | set(negative_counts)):
        gap = positive_counts[word] - negative_counts[word]
        frequency_gap.append(
            {
                "word": word,
                "positive_documents": positive_counts[word],
                "negative_documents": negative_counts[word],
                "gap": gap,
            }
        )
    output = {
        "schema_version": 1,
        "pair_count": len(rows),
        "gate": lexical_leakage_report(rows, seed=args.seed),
        "full_fit_diagnostic_only": {
            "word": full_fit_features(
                texts, labels, analyzer="word", seed=args.seed, top_k=args.top_k
            ),
            "char_wb": full_fit_features(
                texts, labels, analyzer="char_wb", seed=args.seed, top_k=args.top_k
            ),
            "position_word": full_fit_features(
                texts,
                labels,
                analyzer="position_word",
                seed=args.seed,
                top_k=args.top_k,
            ),
            "largest_positive_document_frequency_gaps": sorted(
                frequency_gap, key=lambda row: (row["gap"], row["word"]), reverse=True
            )[: args.top_k],
            "largest_negative_document_frequency_gaps": sorted(
                frequency_gap, key=lambda row: (row["gap"], row["word"])
            )[: args.top_k],
        },
        "within_pair_word_set_jaccard": {
            "minimum": float(np.min(jaccards)),
            "median": float(np.median(jaccards)),
            "maximum": float(np.max(jaccards)),
            "q10": float(np.quantile(jaccards, 0.1)),
            "q90": float(np.quantile(jaccards, 0.9)),
        },
    }
    path = args.output.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "pairs": len(rows),
                "passes": output["gate"]["passes_leakage_gate"],
                "maximum_auroc": output["gate"]["maximum_observed_text_separability_auroc"],
                "median_jaccard": output["within_pair_word_set_jaccard"]["median"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
