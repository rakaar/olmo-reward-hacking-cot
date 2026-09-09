#!/usr/bin/env python3
"""Train a problem-grouped CoT decoder using all four behavior categories."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import joblib
import numpy as np
from safetensors.numpy import load_file
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

import train_cot_decoder as original


MODEL_NAMES = (
    "cot_activation",
    "cot_tfidf",
    "cot_mention_only",
    "prompt_activation",
    "cot_token_count",
)
MODEL_LABELS = {
    "cot_activation": "CoT activation",
    "cot_tfidf": "Visible CoT TF-IDF",
    "cot_mention_only": "Mention indicator",
    "prompt_activation": "Prompt activation",
    "cot_token_count": "CoT token count",
}
COLORS = {
    "cot_activation": "#b4483e",
    "cot_tfidf": "#2b7a9b",
    "cot_mention_only": "#76549a",
    "prompt_activation": "#7d8b96",
    "cot_token_count": "#a56b18",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument(
        "--c-grid",
        type=float,
        nargs="+",
        default=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--split-search-iterations", type=int, default=20_000)
    parser.add_argument("--minimum-outer-test-per-cell", type=int, default=5)
    parser.add_argument("--minimum-outer-train-per-cell", type=int, default=20)
    parser.add_argument("--minimum-inner-validation-per-cell", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-layer", type=int, default=10)
    parser.add_argument("--expected-hidden-size", type=int, default=4096)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def behavior_matrix(
    rows: Sequence[dict[str, Any]], problem_ids: Sequence[str]
) -> np.ndarray:
    index = {problem_id: number for number, problem_id in enumerate(problem_ids)}
    matrix = np.zeros((len(problem_ids), len(original.GROUP_ORDER)), dtype=np.int16)
    for row in rows:
        problem_id = str(row["problem_id"])
        behavior_group = str(row["behavior_group"])
        matrix[index[problem_id], original.GROUP_ORDER.index(behavior_group)] += 1
    return matrix


def balanced_problem_folds(
    rows: Sequence[dict[str, Any]],
    *,
    n_splits: int,
    seed: int,
    search_iterations: int,
) -> tuple[np.ndarray, dict[str, int], np.ndarray, dict[str, Any]]:
    """Balance all four behavior-cell counts while never splitting a problem."""
    problem_ids = sorted({str(row["problem_id"]) for row in rows})
    if n_splits < 2 or len(problem_ids) < n_splits:
        raise ValueError("invalid number of grouped folds")
    matrix = behavior_matrix(rows, problem_ids)
    capacities = np.full(n_splits, len(problem_ids) // n_splits, dtype=np.int16)
    capacities[: len(problem_ids) % n_splits] += 1
    target = matrix.sum(axis=0) / n_splits
    rng = np.random.default_rng(seed)
    best: tuple[tuple[float, ...], np.ndarray, np.ndarray, int] | None = None
    for iteration in range(search_iterations):
        permutation = rng.permutation(len(problem_ids))
        fold_counts = []
        offset = 0
        for capacity in capacities:
            selected = permutation[offset : offset + int(capacity)]
            fold_counts.append(matrix[selected].sum(axis=0))
            offset += int(capacity)
        counts = np.stack(fold_counts)
        normalized_error = float(
            np.sum((counts - target) ** 2 / np.maximum(target, 1.0))
        )
        key = (
            float(counts.min()),
            -normalized_error,
            -float(np.abs(counts - target).max()),
        )
        if best is None or key > best[0]:
            best = (key, permutation.copy(), counts.copy(), iteration)
    if best is None:
        raise RuntimeError("fold search produced no assignment")
    _, permutation, fold_counts, best_iteration = best
    mapping: dict[str, int] = {}
    offset = 0
    for fold, capacity in enumerate(capacities):
        for problem_index in permutation[offset : offset + int(capacity)]:
            mapping[problem_ids[int(problem_index)]] = fold
        offset += int(capacity)
    assignments = np.asarray(
        [mapping[str(row["problem_id"])] for row in rows], dtype=np.int16
    )
    diagnostics = {
        "search_iterations": search_iterations,
        "selected_iteration": best_iteration,
        "problem_counts_per_fold": capacities.astype(int).tolist(),
        "behavior_group_order": list(original.GROUP_ORDER),
        "heldout_behavior_counts": fold_counts.astype(int).tolist(),
        "training_behavior_counts": (
            matrix.sum(axis=0)[None, :] - fold_counts
        ).astype(int).tolist(),
        "global_behavior_counts": matrix.sum(axis=0).astype(int).tolist(),
    }
    return assignments, mapping, fold_counts, diagnostics


def balanced_problem_partitions(
    rows: Sequence[dict[str, Any]],
    *,
    all_problem_ids: Sequence[str],
    partition_sizes: dict[str, int],
    minimum_counts: dict[str, int] | None = None,
    seed: int,
    search_iterations: int,
) -> tuple[np.ndarray, dict[str, str], np.ndarray, dict[str, Any]]:
    """Create unequal problem partitions balanced over all four behavior cells."""
    problem_ids = sorted(set(str(value) for value in all_problem_ids))
    partition_names = list(partition_sizes)
    capacities = np.asarray(
        [int(partition_sizes[name]) for name in partition_names], dtype=np.int32
    )
    if int(capacities.sum()) != len(problem_ids) or np.any(capacities <= 0):
        raise ValueError("partition sizes must be positive and cover every problem")
    row_problem_ids = {str(row["problem_id"]) for row in rows}
    if not row_problem_ids <= set(problem_ids):
        raise ValueError("rows contain a problem absent from all_problem_ids")
    matrix = behavior_matrix(rows, problem_ids)
    target = (
        capacities[:, None]
        * matrix.sum(axis=0, dtype=np.float64)[None, :]
        / len(problem_ids)
    )
    required: np.ndarray | None = None
    if minimum_counts is not None:
        if set(minimum_counts) != set(partition_names):
            raise ValueError("minimum-count keys must match partition names")
        required = np.asarray(
            [int(minimum_counts[name]) for name in partition_names], dtype=np.int32
        )[:, None]
        if np.any(required < 0):
            raise ValueError("minimum counts cannot be negative")
    rng = np.random.default_rng(seed)
    best: tuple[tuple[float, ...], np.ndarray, np.ndarray, int] | None = None
    for iteration in range(search_iterations):
        permutation = rng.permutation(len(problem_ids))
        counts = []
        offset = 0
        for capacity in capacities:
            selected = permutation[offset : offset + int(capacity)]
            counts.append(matrix[selected].sum(axis=0))
            offset += int(capacity)
        count_matrix = np.stack(counts)
        normalized_error = float(
            np.sum((count_matrix - target) ** 2 / np.maximum(target, 1.0))
        )
        if required is None:
            feasibility = 1.0
            minimum_coverage = float(count_matrix[1:].min())
        else:
            feasibility = float(np.all(count_matrix >= required))
            denominator = np.maximum(required, 1)
            minimum_coverage = (
                1.0
                if feasibility
                else float(np.min(count_matrix / denominator))
            )
        key = (
            feasibility,
            minimum_coverage,
            -normalized_error,
            -float(np.abs(count_matrix - target).max()),
        )
        if best is None or key > best[0]:
            best = (key, permutation.copy(), count_matrix.copy(), iteration)
    if best is None:
        raise RuntimeError("partition search produced no assignment")
    _, permutation, count_matrix, best_iteration = best
    mapping: dict[str, str] = {}
    offset = 0
    for partition_name, capacity in zip(partition_names, capacities):
        for problem_index in permutation[offset : offset + int(capacity)]:
            mapping[problem_ids[int(problem_index)]] = partition_name
        offset += int(capacity)
    row_assignments = np.asarray(
        [mapping[str(row["problem_id"])] for row in rows], dtype=object
    )
    diagnostics = {
        "search_iterations": search_iterations,
        "selected_iteration": best_iteration,
        "partition_order": partition_names,
        "problem_counts": capacities.astype(int).tolist(),
        "behavior_group_order": list(original.GROUP_ORDER),
        "behavior_counts": count_matrix.astype(int).tolist(),
        "target_behavior_counts": target.tolist(),
        "global_behavior_counts": matrix.sum(axis=0).astype(int).tolist(),
        "required_minimum_counts": (
            None
            if required is None
            else {
                name: int(required[index, 0])
                for index, name in enumerate(partition_names)
            }
        ),
        "required_minimum_counts_satisfied": (
            None if required is None else bool(np.all(count_matrix >= required))
        ),
    }
    return row_assignments, mapping, count_matrix, diagnostics


def assert_cell_minimums(
    heldout_counts: np.ndarray,
    *,
    minimum_heldout: int,
    minimum_training: int,
) -> None:
    global_counts = heldout_counts.sum(axis=0)
    training_counts = global_counts[None, :] - heldout_counts
    if int(heldout_counts.min()) < minimum_heldout:
        raise ValueError(
            f"minimum held-out cell count is {int(heldout_counts.min())}, "
            f"below required {minimum_heldout}"
        )
    if int(training_counts.min()) < minimum_training:
        raise ValueError(
            f"minimum training cell count is {int(training_counts.min())}, "
            f"below required {minimum_training}"
        )


def balanced_inner_splits(
    *,
    rows: Sequence[dict[str, Any]],
    outer_train_indices: np.ndarray,
    n_splits: int,
    seed: int,
    search_iterations: int,
    minimum_validation_per_cell: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    inner_rows = [rows[int(index)] for index in outer_train_indices]
    inner_assignments, _, heldout_counts, diagnostics = balanced_problem_folds(
        inner_rows,
        n_splits=n_splits,
        seed=seed,
        search_iterations=search_iterations,
    )
    if int(heldout_counts.min()) < minimum_validation_per_cell:
        raise ValueError(
            f"inner validation cell minimum {int(heldout_counts.min())} is below "
            f"required {minimum_validation_per_cell}"
        )
    splits = []
    for fold in range(n_splits):
        inner_validation = outer_train_indices[inner_assignments == fold]
        inner_train = outer_train_indices[inner_assignments != fold]
        train_groups = {str(rows[int(index)]["problem_id"]) for index in inner_train}
        validation_groups = {
            str(rows[int(index)]["problem_id"]) for index in inner_validation
        }
        if train_groups & validation_groups:
            raise AssertionError("problem leakage in inner fold")
        splits.append((inner_train, inner_validation))
    return splits, diagnostics


def tune_c(
    *,
    values: Any,
    labels: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    estimator_factory: Callable[[float, int], Any],
    c_grid: Sequence[float],
    seed: int,
) -> tuple[float, list[dict[str, Any]]]:
    results = []
    for c_value in sorted(c_grid):
        fold_scores = []
        for split_number, (train_indices, validation_indices) in enumerate(splits):
            estimator = estimator_factory(float(c_value), seed + split_number)
            estimator.fit(original.subset(values, train_indices), labels[train_indices])
            scores = estimator.predict_proba(
                original.subset(values, validation_indices)
            )[:, 1]
            fold_scores.append(
                float(roc_auc_score(labels[validation_indices], scores))
            )
        results.append(
            {
                "C": float(c_value),
                "inner_fold_aurocs": fold_scores,
                "mean_inner_auroc": float(np.mean(fold_scores)),
            }
        )
    best = max(results, key=lambda row: (row["mean_inner_auroc"], -row["C"]))
    return float(best["C"]), results


def evaluate_subset(
    *,
    labels: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    mask: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    selected = np.flatnonzero(mask)
    result = original.binary_metrics(labels[selected], scores[selected])
    result.update(
        original.grouped_bootstrap_metrics(
            labels=labels[selected],
            scores=scores[selected],
            groups=groups[selected],
            replicates=replicates,
            seed=seed,
        )
    )
    return result


def paired_grouped_auc_difference(
    *,
    labels: np.ndarray,
    first_scores: np.ndarray,
    second_scores: np.ndarray,
    groups: np.ndarray,
    mask: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    selected = np.flatnonzero(mask)
    y = labels[selected]
    first = first_scores[selected]
    second = second_scores[selected]
    selected_groups = groups[selected]
    unique_groups = np.unique(selected_groups)
    group_indices = {
        group: np.flatnonzero(selected_groups == group) for group in unique_groups
    }
    rng = np.random.default_rng(seed)
    differences = []
    attempts = 0
    while len(differences) < replicates and attempts < max(1000, replicates * 100):
        attempts += 1
        sampled_groups = rng.choice(
            unique_groups, size=len(unique_groups), replace=True
        )
        indices = np.concatenate([group_indices[group] for group in sampled_groups])
        if np.unique(y[indices]).size != 2:
            continue
        differences.append(
            float(
                roc_auc_score(y[indices], first[indices])
                - roc_auc_score(y[indices], second[indices])
            )
        )
    if len(differences) != replicates:
        raise RuntimeError("could not obtain requested paired bootstrap replicates")
    point = float(roc_auc_score(y, first) - roc_auc_score(y, second))
    return {
        "auroc_difference": point,
        "grouped_95_ci": [
            float(value) for value in np.quantile(differences, [0.025, 0.975])
        ],
        "replicates": replicates,
        "positive_means_activation_higher": True,
    }


def save_figure(
    *,
    path: Path,
    labels: np.ndarray,
    predictions: dict[str, np.ndarray],
    mentions: np.ndarray,
    behavior_groups: np.ndarray,
    seed: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 11.0))
    for model_name in MODEL_NAMES:
        scores = predictions[model_name]
        false_positive, true_positive, _ = roc_curve(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        axes[0, 0].plot(
            false_positive,
            true_positive,
            color=COLORS[model_name],
            linewidth=2,
            label=f"{MODEL_LABELS[model_name]} ({roc_auc_score(labels, scores):.3f})",
        )
        axes[0, 1].plot(
            recall,
            precision,
            color=COLORS[model_name],
            linewidth=2,
            label=f"{MODEL_LABELS[model_name]} ({average_precision_score(labels, scores):.3f})",
        )
    axes[0, 0].plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.55)
    axes[0, 0].set(
        title="All four groups: held-out ROC",
        xlabel="False-positive rate",
        ylabel="True-positive rate",
    )
    axes[0, 0].legend(fontsize=8, loc="lower right")
    axes[0, 1].axhline(
        float(labels.mean()), linestyle="--", color="black", alpha=0.55
    )
    axes[0, 1].set(
        title="All four groups: held-out precision-recall",
        xlabel="Recall",
        ylabel="Precision",
    )
    axes[0, 1].legend(fontsize=8, loc="lower left")

    activation = predictions["cot_activation"]
    for mask, label, color in (
        (~mentions, "No hack mention", "#2b7a9b"),
        (mentions, "Hack mentioned", "#b4483e"),
    ):
        false_positive, true_positive, _ = roc_curve(labels[mask], activation[mask])
        auc = roc_auc_score(labels[mask], activation[mask])
        axes[1, 0].plot(
            false_positive,
            true_positive,
            color=color,
            linewidth=2.3,
            label=f"{label} ({auc:.3f})",
        )
    axes[1, 0].plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.55)
    axes[1, 0].set(
        title="Does activation predict the answer within each CoT stratum?",
        xlabel="False-positive rate",
        ylabel="True-positive rate",
    )
    axes[1, 0].legend(fontsize=9, loc="lower right")

    grouped_values = [
        activation[behavior_groups == group] for group in original.GROUP_ORDER
    ]
    box = axes[1, 1].boxplot(grouped_values, patch_artist=True, showfliers=False)
    for patch, color in zip(
        box["boxes"], ["#dce8ed", "#f0c8c2", "#f4ddad", "#d9d0ea"]
    ):
        patch.set_facecolor(color)
    rng = np.random.default_rng(seed)
    for position, values in enumerate(grouped_values, 1):
        axes[1, 1].scatter(
            np.full(len(values), position) + rng.normal(0, 0.045, size=len(values)),
            values,
            s=16,
            alpha=0.55,
            color="#29465a",
            edgecolors="none",
        )
    axes[1, 1].set_xticks(range(1, len(original.GROUP_ORDER) + 1))
    axes[1, 1].set_xticklabels(
        [original.GROUP_LABELS[group] for group in original.GROUP_ORDER], fontsize=8
    )
    axes[1, 1].set(
        title="Layer-10 activation scores by all four groups",
        ylabel="Out-of-fold decoder score",
    )
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.suptitle(
        "Layer-10 mean-pooled CoT decoder trained on all four groups", fontsize=16
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    feature_path = args.features.expanduser().resolve()
    metadata_path = args.metadata.expanduser().resolve()
    feature_manifest_path = args.feature_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise SystemExit(f"output directory is not empty: {output_dir}")
    models_dir = output_dir / "models"
    models_dir.mkdir()

    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    if feature_manifest.get("feature_file_sha256") != original.sha256_file(feature_path):
        raise SystemExit("feature file hash does not match feature manifest")
    if feature_manifest.get("metadata_file_sha256") != original.sha256_file(metadata_path):
        raise SystemExit("metadata hash does not match feature manifest")
    if int(feature_manifest.get("layer_index", -1)) != args.expected_layer:
        raise SystemExit("feature layer does not match expected layer")

    rows = original.read_jsonl(metadata_path)
    tensors = load_file(str(feature_path))
    cot_features = np.asarray(tensors["cot_mean"], dtype=np.float32)
    prompt_features = np.asarray(tensors["prompt_mean"], dtype=np.float32)
    expected_shape = (len(rows), args.expected_hidden_size)
    if cot_features.shape != expected_shape or prompt_features.shape != expected_shape:
        raise SystemExit(f"unexpected feature shapes: {cot_features.shape}, {prompt_features.shape}")
    if not np.isfinite(cot_features).all() or not np.isfinite(prompt_features).all():
        raise SystemExit("features contain non-finite values")

    labels = np.asarray([bool(row["hack_attempted"]) for row in rows], dtype=np.int8)
    reward_hacked = np.asarray([bool(row["reward_hacked"]) for row in rows], dtype=np.int8)
    groups = np.asarray([str(row["problem_id"]) for row in rows], dtype=object)
    mentions = np.asarray([bool(row["cot_mentions_hack"]) for row in rows], dtype=bool)
    behavior_groups = np.asarray(
        [str(row["behavior_group"]) for row in rows], dtype=object
    )
    texts = [str(row["thinking_text"]) for row in rows]
    token_count = np.log1p(
        np.asarray([int(row["thinking_token_count"]) for row in rows], dtype=np.float64)
    )[:, None]
    mention_feature = mentions.astype(np.float64)[:, None]

    assignments, group_mapping, heldout_counts, split_diagnostics = balanced_problem_folds(
        rows,
        n_splits=args.outer_folds,
        seed=args.seed,
        search_iterations=args.split_search_iterations,
    )
    assert_cell_minimums(
        heldout_counts,
        minimum_heldout=args.minimum_outer_test_per_cell,
        minimum_training=args.minimum_outer_train_per_cell,
    )
    folds_path = output_dir / "folds.json"
    write_json(
        folds_path,
        {
            "schema_version": 1,
            "seed": args.seed,
            "assignment_unit": "problem_id",
            "stratification": "four CoT-mention by answer-attempt behavior cells",
            "problem_to_fold": dict(sorted(group_mapping.items())),
            "constraints": {
                "minimum_outer_test_per_behavior_cell": args.minimum_outer_test_per_cell,
                "minimum_outer_train_per_behavior_cell": args.minimum_outer_train_per_cell,
            },
            "diagnostics": split_diagnostics,
        },
    )

    model_values: dict[str, Any] = {
        "cot_activation": cot_features,
        "cot_tfidf": texts,
        "cot_mention_only": mention_feature,
        "prompt_activation": prompt_features,
        "cot_token_count": token_count,
    }
    factories: dict[str, Callable[[float, int], Any]] = {
        "cot_activation": original.dense_estimator,
        "cot_tfidf": original.text_estimator,
        "cot_mention_only": original.dense_estimator,
        "prompt_activation": original.dense_estimator,
        "cot_token_count": original.dense_estimator,
    }
    predictions = {
        name: np.full(len(rows), np.nan, dtype=np.float64) for name in MODEL_NAMES
    }
    fold_results = []
    for fold in range(args.outer_folds):
        train_indices = np.flatnonzero(assignments != fold)
        heldout_indices = np.flatnonzero(assignments == fold)
        overlap = set(groups[train_indices]) & set(groups[heldout_indices])
        if overlap:
            raise SystemExit(f"problem leakage in outer fold {fold}: {sorted(overlap)}")
        inner_splits, inner_diagnostics = balanced_inner_splits(
            rows=rows,
            outer_train_indices=train_indices,
            n_splits=args.inner_folds,
            seed=args.seed + 1000 + fold,
            search_iterations=args.split_search_iterations,
            minimum_validation_per_cell=args.minimum_inner_validation_per_cell,
        )
        for model_number, model_name in enumerate(MODEL_NAMES):
            model_path = models_dir / f"fold_{fold}_{model_name}.joblib"
            if model_name == "cot_mention_only":
                chosen_c = None
                tuning = []
                predictions[model_name][heldout_indices] = mentions[
                    heldout_indices
                ].astype(np.float64)
                joblib.dump(
                    {
                        "control": "raw binary cot_mentions_hack indicator",
                        "requires_fitting": False,
                    },
                    model_path,
                    compress=3,
                )
            else:
                chosen_c, tuning = tune_c(
                    values=model_values[model_name],
                    labels=labels,
                    splits=inner_splits,
                    estimator_factory=factories[model_name],
                    c_grid=args.c_grid,
                    seed=args.seed + fold * 100 + model_number,
                )
                estimator = factories[model_name](chosen_c, args.seed + fold)
                estimator.fit(
                    original.subset(model_values[model_name], train_indices),
                    labels[train_indices],
                )
                predictions[model_name][heldout_indices] = estimator.predict_proba(
                    original.subset(model_values[model_name], heldout_indices)
                )[:, 1]
                joblib.dump(estimator, model_path, compress=3)
            heldout_no_mention = heldout_indices[~mentions[heldout_indices]]
            heldout_mention = heldout_indices[mentions[heldout_indices]]
            fold_results.append(
                {
                    "fold": fold,
                    "model": model_name,
                    "chosen_C": chosen_c,
                    "train_n": int(len(train_indices)),
                    "heldout_n": int(len(heldout_indices)),
                    "train_problem_count": int(len(np.unique(groups[train_indices]))),
                    "heldout_problem_count": int(len(np.unique(groups[heldout_indices]))),
                    "train_behavior_counts": {
                        group: int(np.sum(behavior_groups[train_indices] == group))
                        for group in original.GROUP_ORDER
                    },
                    "heldout_behavior_counts": {
                        group: int(np.sum(behavior_groups[heldout_indices] == group))
                        for group in original.GROUP_ORDER
                    },
                    "heldout_all_metrics": original.binary_metrics(
                        labels[heldout_indices], predictions[model_name][heldout_indices]
                    ),
                    "heldout_no_mention_metrics": original.binary_metrics(
                        labels[heldout_no_mention],
                        predictions[model_name][heldout_no_mention],
                    ),
                    "heldout_mention_metrics": original.binary_metrics(
                        labels[heldout_mention],
                        predictions[model_name][heldout_mention],
                    ),
                    "inner_split_diagnostics": inner_diagnostics,
                    "inner_tuning": tuning,
                    "model_file": str(model_path.relative_to(output_dir)),
                }
            )
            print(
                f"fold={fold} model={model_name} "
                f"C={'fixed' if chosen_c is None else f'{chosen_c:g}'} "
                f"AUROC={fold_results[-1]['heldout_all_metrics']['auroc']:.3f}",
                flush=True,
            )
    if any(not np.isfinite(scores).all() for scores in predictions.values()):
        raise SystemExit("one or more models have missing out-of-fold predictions")

    masks = {
        "all_complete_cot": np.ones(len(rows), dtype=bool),
        "no_mention": ~mentions,
        "mention": mentions,
    }
    metrics: dict[str, Any] = {}
    for model_number, model_name in enumerate(MODEL_NAMES):
        metrics[model_name] = {}
        for subset_number, (subset_name, mask) in enumerate(masks.items()):
            metrics[model_name][subset_name] = evaluate_subset(
                labels=labels,
                scores=predictions[model_name],
                groups=groups,
                mask=mask,
                replicates=args.bootstrap_replicates,
                seed=args.seed + model_number * 1000 + subset_number,
            )
        metrics[model_name]["reward_hacked_secondary"] = evaluate_subset(
            labels=reward_hacked,
            scores=predictions[model_name],
            groups=groups,
            mask=np.ones(len(rows), dtype=bool),
            replicates=args.bootstrap_replicates,
            seed=args.seed + model_number * 1000 + 100,
        )

    comparisons: dict[str, Any] = {}
    for control_number, control in enumerate(MODEL_NAMES[1:]):
        comparisons[control] = {}
        for subset_number, (subset_name, mask) in enumerate(masks.items()):
            comparisons[control][subset_name] = paired_grouped_auc_difference(
                labels=labels,
                first_scores=predictions["cot_activation"],
                second_scores=predictions[control],
                groups=groups,
                mask=mask,
                replicates=args.bootstrap_replicates,
                seed=args.seed + control_number * 1000 + subset_number + 20_000,
            )

    group_summaries: dict[str, Any] = {}
    for model_name in MODEL_NAMES:
        group_summaries[model_name] = {}
        for group in original.GROUP_ORDER:
            values = predictions[model_name][behavior_groups == group]
            group_summaries[model_name][group] = {
                "n": int(len(values)),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
            }

    primary = metrics["cot_activation"]["all_complete_cot"]
    no_mention = metrics["cot_activation"]["no_mention"]
    summary = {
        "schema_version": 1,
        "experiment": "layer-10 mean-pooled CoT decoder trained on all four groups",
        "primary_model": "cot_activation",
        "primary_training_subset": "all complete-CoT rollouts from all four behavior groups",
        "primary_label": "hack_attempted in the subsequent answer",
        "feature_layer": args.expected_layer,
        "pooling": "mean over non-special tokens strictly inside complete CoT",
        "rollouts": len(rows),
        "problem_count": int(len(np.unique(groups))),
        "behavior_group_counts": {
            group: int(np.sum(behavior_groups == group))
            for group in original.GROUP_ORDER
        },
        "split_constraints_satisfied": {
            "minimum_outer_test_per_behavior_cell": int(heldout_counts.min()),
            "minimum_outer_train_per_behavior_cell": int(
                (heldout_counts.sum(axis=0)[None, :] - heldout_counts).min()
            ),
            "required_outer_test_minimum": args.minimum_outer_test_per_cell,
            "required_outer_train_minimum": args.minimum_outer_train_per_cell,
            "required_inner_validation_minimum": args.minimum_inner_validation_per_cell,
        },
        "metrics": metrics,
        "activation_auroc_differences_vs_controls": comparisons,
        "behavior_group_score_summaries": group_summaries,
        "per_fold": fold_results,
        "success_gate": {
            "required_overall_auroc": 0.70,
            "required_grouped_95_ci_lower_strictly_above": 0.50,
            "observed_overall_auroc": primary["auroc"],
            "observed_overall_auroc_ci95": primary["auroc_ci95"],
            "overall_gate_passed": bool(
                primary["auroc"] >= 0.70 and primary["auroc_ci95"][0] > 0.50
            ),
            "no_mention_auroc": no_mention["auroc"],
            "no_mention_auroc_ci95": no_mention["auroc_ci95"],
            "evidence_beyond_verbalized_mention": bool(
                no_mention["auroc_ci95"][0] > 0.50
            ),
        },
        "interpretation": {
            "cot_mention_is_not_used_to_select_training_examples": True,
            "all_four_behavior_groups_appear_in_every_outer_train_and_test_fold": True,
            "stratified_metrics_are_out_of_fold_but_not_zero_shot": (
                "Silent attempts occur in training folds; no-mention performance tests "
                "held-out-problem generalization, not training only on verbalized cases."
            ),
            "scores_are_not_calibrated_probabilities": True,
        },
    }
    summary_path = output_dir / "summary.json"
    write_json(summary_path, summary)

    prediction_rows = []
    for index, row in enumerate(rows):
        prediction_rows.append(
            {
                "schema_version": 1,
                "feature_index": index,
                "rollout_id": str(row["rollout_id"]),
                "problem_id": str(row["problem_id"]),
                "outer_fold": int(assignments[index]),
                "hack_attempted": bool(labels[index]),
                "reward_hacked": bool(reward_hacked[index]),
                "cot_mentions_hack": bool(mentions[index]),
                "behavior_group": str(behavior_groups[index]),
                "thinking_token_count": int(row["thinking_token_count"]),
                "predictions": {
                    name: float(predictions[name][index]) for name in MODEL_NAMES
                },
            }
        )
    predictions_path = output_dir / "oof_predictions.jsonl"
    write_jsonl(predictions_path, prediction_rows)

    fold_csv_path = output_dir / "fold_metrics.csv"
    with fold_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "fold",
                "model",
                "chosen_C",
                "train_n",
                "heldout_n",
                "train_problem_count",
                "heldout_problem_count",
                *[f"train_{group}" for group in original.GROUP_ORDER],
                *[f"heldout_{group}" for group in original.GROUP_ORDER],
                "heldout_auroc",
                "heldout_auprc",
                "heldout_no_mention_auroc",
                "heldout_mention_auroc",
            ],
        )
        writer.writeheader()
        for row in fold_results:
            writer.writerow(
                {
                    "fold": row["fold"],
                    "model": row["model"],
                    "chosen_C": row["chosen_C"],
                    "train_n": row["train_n"],
                    "heldout_n": row["heldout_n"],
                    "train_problem_count": row["train_problem_count"],
                    "heldout_problem_count": row["heldout_problem_count"],
                    **{
                        f"train_{group}": row["train_behavior_counts"][group]
                        for group in original.GROUP_ORDER
                    },
                    **{
                        f"heldout_{group}": row["heldout_behavior_counts"][group]
                        for group in original.GROUP_ORDER
                    },
                    "heldout_auroc": row["heldout_all_metrics"]["auroc"],
                    "heldout_auprc": row["heldout_all_metrics"]["auprc"],
                    "heldout_no_mention_auroc": row[
                        "heldout_no_mention_metrics"
                    ]["auroc"],
                    "heldout_mention_auroc": row["heldout_mention_metrics"][
                        "auroc"
                    ],
                }
            )

    figure_path = output_dir / "cot_decoder_all_groups.png"
    save_figure(
        path=figure_path,
        labels=labels,
        predictions=predictions,
        mentions=mentions,
        behavior_groups=behavior_groups,
        seed=args.seed,
    )

    manifest_path = output_dir / "manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "status": "complete",
            "feature_file_sha256": original.sha256_file(feature_path),
            "metadata_file_sha256": original.sha256_file(metadata_path),
            "feature_manifest_sha256": original.sha256_file(feature_manifest_path),
            "files": {
                "summary": summary_path.name,
                "folds": folds_path.name,
                "predictions": predictions_path.name,
                "fold_metrics": fold_csv_path.name,
                "figure": figure_path.name,
            },
            "package_versions": original.package_versions(
                ["numpy", "scikit-learn", "safetensors", "joblib", "matplotlib"]
            ),
        },
    )
    artifact_paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (output_dir / "SHA256SUMS").write_text(
        "".join(
            f"{original.sha256_file(path)}  {path.relative_to(output_dir)}\n"
            for path in artifact_paths
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary["success_gate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
