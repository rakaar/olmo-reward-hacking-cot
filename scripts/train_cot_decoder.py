#!/usr/bin/env python3
"""Train and evaluate a leakage-safe mean-pooled CoT activation decoder."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import joblib
import numpy as np
from safetensors.numpy import load_file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_NAMES = (
    "cot_activation",
    "prompt_activation",
    "cot_token_count",
    "cot_tfidf",
)
MODEL_LABELS = {
    "cot_activation": "CoT activation",
    "prompt_activation": "Prompt activation",
    "cot_token_count": "CoT token count",
    "cot_tfidf": "CoT TF-IDF",
}
GROUP_ORDER = (
    "no_mention_no_attempt",
    "mention_attempt",
    "no_mention_attempt",
    "mention_no_attempt",
)
GROUP_LABELS = {
    "no_mention_no_attempt": "No mention\nNo attempt",
    "mention_attempt": "Mention\nAttempt",
    "no_mention_attempt": "No mention\nAttempt",
    "mention_no_attempt": "Mention\nNo attempt",
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-layer", type=int, default=10)
    parser.add_argument("--expected-hidden-size", type=int, default=4096)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def package_versions(names: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def transparent_mask(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    return np.asarray([bool(row["transparent"]) for row in rows], dtype=bool)


def make_outer_fold_assignments(
    rows: Sequence[dict[str, Any]], n_splits: int, seed: int
) -> tuple[np.ndarray, dict[str, int]]:
    """Stratify transparent examples, then place any otherwise unseen groups."""
    primary = transparent_mask(rows)
    labels = np.asarray([bool(row["hack_attempted"]) for row in rows], dtype=np.int8)
    groups = np.asarray([str(row["problem_id"]) for row in rows], dtype=object)
    primary_indices = np.flatnonzero(primary)
    if np.unique(labels[primary_indices]).size != 2:
        raise ValueError("transparent subset must contain both classes")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    mapping: dict[str, int] = {}
    for fold, (_train, heldout) in enumerate(
        splitter.split(
            np.zeros((len(primary_indices), 1)),
            labels[primary_indices],
            groups[primary_indices],
        )
    ):
        for group in np.unique(groups[primary_indices][heldout]):
            group_text = str(group)
            if group_text in mapping:
                raise AssertionError(f"group assigned twice: {group_text}")
            mapping[group_text] = fold

    all_groups = sorted({str(value) for value in groups})
    missing = [group for group in all_groups if group not in mapping]
    group_sizes = {group: int(np.sum(groups == group)) for group in all_groups}
    fold_sizes = [
        sum(group_sizes[group] for group, assigned in mapping.items() if assigned == fold)
        for fold in range(n_splits)
    ]
    for group in missing:
        fold = min(range(n_splits), key=lambda value: (fold_sizes[value], value))
        mapping[group] = fold
        fold_sizes[fold] += group_sizes[group]
    assignments = np.asarray([mapping[str(group)] for group in groups], dtype=np.int16)
    if set(mapping) != set(all_groups):
        raise AssertionError("not every problem group received a fold")
    return assignments, mapping


def valid_group_splits(
    indices: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    requested_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    for n_splits in range(requested_splits, 1, -1):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        candidate: list[tuple[np.ndarray, np.ndarray]] = []
        valid = True
        try:
            raw_splits = splitter.split(
                np.zeros((len(indices), 1)), labels[indices], groups[indices]
            )
            for inner_train, inner_valid in raw_splits:
                train_indices = indices[inner_train]
                valid_indices = indices[inner_valid]
                if (
                    np.unique(labels[train_indices]).size != 2
                    or np.unique(labels[valid_indices]).size != 2
                ):
                    valid = False
                    break
                candidate.append((train_indices, valid_indices))
        except ValueError:
            valid = False
        if valid and len(candidate) == n_splits:
            return candidate
    raise ValueError("could not construct inner grouped folds containing both classes")


def dense_estimator(c_value: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=seed,
                ),
            ),
        ]
    )


def text_estimator(c_value: float, seed: int) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=20_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "character",
                TfidfVectorizer(
                    analyzer="char_wb",
                    lowercase=True,
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=30_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=seed,
                ),
            ),
        ]
    )


def subset(values: Any, indices: np.ndarray) -> Any:
    if isinstance(values, np.ndarray):
        return values[indices]
    return [values[int(index)] for index in indices]


def tune_c(
    *,
    values: Any,
    labels: np.ndarray,
    groups: np.ndarray,
    train_indices: np.ndarray,
    estimator_factory: Callable[[float, int], Any],
    c_grid: Sequence[float],
    inner_folds: int,
    seed: int,
) -> tuple[float, list[dict[str, Any]]]:
    splits = valid_group_splits(
        train_indices, labels, groups, inner_folds, seed
    )
    results: list[dict[str, Any]] = []
    for c_value in sorted(c_grid):
        fold_scores: list[float] = []
        for split_number, (inner_train, inner_valid) in enumerate(splits):
            estimator = estimator_factory(c_value, seed + split_number)
            estimator.fit(subset(values, inner_train), labels[inner_train])
            probabilities = estimator.predict_proba(subset(values, inner_valid))[:, 1]
            fold_scores.append(
                float(roc_auc_score(labels[inner_valid], probabilities))
            )
        results.append(
            {
                "C": float(c_value),
                "inner_fold_aurocs": fold_scores,
                "mean_inner_auroc": float(np.mean(fold_scores)),
                "inner_fold_count": len(fold_scores),
            }
        )
    best = max(results, key=lambda row: (row["mean_inner_auroc"], -row["C"]))
    return float(best["C"]), results


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    if len(labels) == 0 or np.unique(labels).size != 2:
        raise ValueError("binary metrics require nonempty observations from both classes")
    return {
        "n": int(len(labels)),
        "n_positive": int(labels.sum()),
        "n_negative": int(len(labels) - labels.sum()),
        "prevalence": float(labels.mean()),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }


def grouped_bootstrap_metrics(
    *,
    labels: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    aurocs: list[float] = []
    auprcs: list[float] = []
    attempts = 0
    maximum_attempts = max(replicates * 100, 1000)
    group_to_indices = {
        group: np.flatnonzero(groups == group) for group in unique_groups
    }
    while len(aurocs) < replicates and attempts < maximum_attempts:
        attempts += 1
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_indices = np.concatenate(
            [group_to_indices[group] for group in sampled_groups]
        )
        sampled_labels = labels[sampled_indices]
        if np.unique(sampled_labels).size != 2:
            continue
        sampled_scores = scores[sampled_indices]
        aurocs.append(float(roc_auc_score(sampled_labels, sampled_scores)))
        auprcs.append(float(average_precision_score(sampled_labels, sampled_scores)))
    if len(aurocs) != replicates:
        raise RuntimeError(
            f"obtained only {len(aurocs)}/{replicates} valid bootstrap replicates"
        )
    return {
        "replicates": replicates,
        "group_count": int(len(unique_groups)),
        "auroc_ci95": [float(value) for value in np.quantile(aurocs, [0.025, 0.975])],
        "auprc_ci95": [float(value) for value in np.quantile(auprcs, [0.025, 0.975])],
    }


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
    metrics = binary_metrics(labels[selected], scores[selected])
    metrics.update(
        grouped_bootstrap_metrics(
            labels=labels[selected],
            scores=scores[selected],
            groups=groups[selected],
            replicates=replicates,
            seed=seed,
        )
    )
    return metrics


def save_figure(
    *,
    path: Path,
    labels: np.ndarray,
    predictions: dict[str, np.ndarray],
    transparent: np.ndarray,
    behavior_groups: np.ndarray,
    seed: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = np.flatnonzero(transparent)
    colors = {
        "cot_activation": "#b4483e",
        "prompt_activation": "#7d8b96",
        "cot_token_count": "#a56b18",
        "cot_tfidf": "#2b7a9b",
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    for model_name in MODEL_NAMES:
        scores = predictions[model_name][selected]
        false_positive, true_positive, _ = roc_curve(labels[selected], scores)
        precision, recall, _ = precision_recall_curve(labels[selected], scores)
        auc = roc_auc_score(labels[selected], scores)
        ap = average_precision_score(labels[selected], scores)
        axes[0].plot(
            false_positive,
            true_positive,
            color=colors[model_name],
            linewidth=2,
            label=f"{MODEL_LABELS[model_name]} ({auc:.3f})",
        )
        axes[1].plot(
            recall,
            precision,
            color=colors[model_name],
            linewidth=2,
            label=f"{MODEL_LABELS[model_name]} ({ap:.3f})",
        )
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.55)
    axes[0].set(title="Transparent held-out examples", xlabel="False-positive rate", ylabel="True-positive rate")
    axes[0].legend(fontsize=8, loc="lower right")
    prevalence = float(labels[selected].mean())
    axes[1].axhline(prevalence, linestyle="--", color="black", alpha=0.55)
    axes[1].set(title="Transparent held-out examples", xlabel="Recall", ylabel="Precision")
    axes[1].legend(fontsize=8, loc="lower left")

    activation_scores = predictions["cot_activation"]
    grouped_values = [activation_scores[behavior_groups == group] for group in GROUP_ORDER]
    box = axes[2].boxplot(grouped_values, patch_artist=True, showfliers=False)
    box_colors = ["#dce8ed", "#f0c8c2", "#f4ddad", "#d9d0ea"]
    for patch, color in zip(box["boxes"], box_colors):
        patch.set_facecolor(color)
    rng = np.random.default_rng(seed)
    for position, values in enumerate(grouped_values, 1):
        jitter = rng.normal(0, 0.045, size=len(values))
        axes[2].scatter(
            np.full(len(values), position) + jitter,
            values,
            s=16,
            alpha=0.55,
            color="#29465a",
            edgecolors="none",
        )
    axes[2].set_xticks(range(1, len(GROUP_ORDER) + 1))
    axes[2].set_xticklabels([GROUP_LABELS[group] for group in GROUP_ORDER], fontsize=8)
    axes[2].set(title="Layer-10 CoT decoder by behavior group", ylabel="Out-of-fold hack probability")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle("Mean-pooled layer-10 CoT decoder pilot", fontsize=16)
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
    if feature_manifest.get("feature_file_sha256") != sha256_file(feature_path):
        raise SystemExit("feature file hash does not match feature manifest")
    if feature_manifest.get("metadata_file_sha256") != sha256_file(metadata_path):
        raise SystemExit("metadata hash does not match feature manifest")
    if int(feature_manifest.get("layer_index", -1)) != args.expected_layer:
        raise SystemExit("feature layer does not match expected layer")

    rows = read_jsonl(metadata_path)
    tensors = load_file(str(feature_path))
    if set(tensors) != {"cot_mean", "prompt_mean"}:
        raise SystemExit(f"unexpected feature keys: {sorted(tensors)}")
    cot_features = np.asarray(tensors["cot_mean"], dtype=np.float32)
    prompt_features = np.asarray(tensors["prompt_mean"], dtype=np.float32)
    expected_shape = (len(rows), args.expected_hidden_size)
    if cot_features.shape != expected_shape or prompt_features.shape != expected_shape:
        raise SystemExit(
            f"feature shape mismatch: cot={cot_features.shape}, "
            f"prompt={prompt_features.shape}, expected={expected_shape}"
        )
    if not np.isfinite(cot_features).all() or not np.isfinite(prompt_features).all():
        raise SystemExit("features contain non-finite values")
    for index, row in enumerate(rows):
        if int(row["feature_index"]) != index:
            raise SystemExit("metadata feature indices are not contiguous and ordered")
    rollout_ids = [str(row["rollout_id"]) for row in rows]
    if len(rollout_ids) != len(set(rollout_ids)):
        raise SystemExit("duplicate rollout IDs")

    labels = np.asarray([bool(row["hack_attempted"]) for row in rows], dtype=np.int8)
    reward_hacked = np.asarray([bool(row["reward_hacked"]) for row in rows], dtype=np.int8)
    groups = np.asarray([str(row["problem_id"]) for row in rows], dtype=object)
    mentions = np.asarray([bool(row["cot_mentions_hack"]) for row in rows], dtype=bool)
    transparent = transparent_mask(rows)
    behavior_groups = np.asarray([str(row["behavior_group"]) for row in rows], dtype=object)
    token_count = np.log1p(
        np.asarray([int(row["thinking_token_count"]) for row in rows], dtype=np.float64)
    )[:, None]
    texts = [str(row["thinking_text"]) for row in rows]
    outer_assignments, group_fold_mapping = make_outer_fold_assignments(
        rows, args.outer_folds, args.seed
    )
    folds_path = output_dir / "folds.json"
    write_json(
        folds_path,
        {
            "schema_version": 1,
            "seed": args.seed,
            "outer_folds": args.outer_folds,
            "assignment_unit": "problem_id",
            "stratification_subset": "transparent examples",
            "problem_to_fold": dict(sorted(group_fold_mapping.items())),
        },
    )

    model_values: dict[str, Any] = {
        "cot_activation": cot_features,
        "prompt_activation": prompt_features,
        "cot_token_count": token_count,
        "cot_tfidf": texts,
    }
    factories: dict[str, Callable[[float, int], Any]] = {
        "cot_activation": dense_estimator,
        "prompt_activation": dense_estimator,
        "cot_token_count": dense_estimator,
        "cot_tfidf": text_estimator,
    }
    predictions = {
        model_name: np.full(len(rows), np.nan, dtype=np.float64)
        for model_name in MODEL_NAMES
    }
    fold_results: list[dict[str, Any]] = []
    for fold in range(args.outer_folds):
        train_indices = np.flatnonzero(transparent & (outer_assignments != fold))
        heldout_indices = np.flatnonzero(outer_assignments == fold)
        heldout_transparent = heldout_indices[transparent[heldout_indices]]
        if (
            np.unique(labels[train_indices]).size != 2
            or np.unique(labels[heldout_transparent]).size != 2
        ):
            raise SystemExit(f"outer fold {fold} does not contain both transparent classes")
        overlap = set(groups[train_indices]) & set(groups[heldout_indices])
        if overlap:
            raise SystemExit(f"problem leakage in fold {fold}: {sorted(overlap)}")
        for model_number, model_name in enumerate(MODEL_NAMES):
            values = model_values[model_name]
            factory = factories[model_name]
            chosen_c, tuning = tune_c(
                values=values,
                labels=labels,
                groups=groups,
                train_indices=train_indices,
                estimator_factory=factory,
                c_grid=args.c_grid,
                inner_folds=args.inner_folds,
                seed=args.seed + fold * 100 + model_number,
            )
            estimator = factory(chosen_c, args.seed + fold)
            estimator.fit(subset(values, train_indices), labels[train_indices])
            fold_predictions = estimator.predict_proba(
                subset(values, heldout_indices)
            )[:, 1]
            predictions[model_name][heldout_indices] = fold_predictions
            model_path = models_dir / f"fold_{fold}_{model_name}.joblib"
            joblib.dump(estimator, model_path, compress=3)
            primary_metrics = binary_metrics(
                labels[heldout_transparent], predictions[model_name][heldout_transparent]
            )
            no_mention_fold = heldout_indices[~mentions[heldout_indices]]
            no_mention_metrics = (
                binary_metrics(
                    labels[no_mention_fold], predictions[model_name][no_mention_fold]
                )
                if len(no_mention_fold) > 0
                and np.unique(labels[no_mention_fold]).size == 2
                else None
            )
            fold_results.append(
                {
                    "fold": fold,
                    "model": model_name,
                    "chosen_C": chosen_c,
                    "train_transparent_n": int(len(train_indices)),
                    "train_problem_count": int(len(np.unique(groups[train_indices]))),
                    "heldout_all_n": int(len(heldout_indices)),
                    "heldout_problem_count": int(len(np.unique(groups[heldout_indices]))),
                    "transparent_metrics": primary_metrics,
                    "no_mention_metrics": no_mention_metrics,
                    "inner_tuning": tuning,
                    "model_file": str(model_path.relative_to(output_dir)),
                }
            )
            print(
                f"fold={fold} model={model_name} C={chosen_c:g} "
                f"transparent_AUROC={primary_metrics['auroc']:.3f}",
                flush=True,
            )

    for model_name, values in predictions.items():
        if not np.isfinite(values).all():
            missing = np.flatnonzero(~np.isfinite(values)).tolist()
            raise SystemExit(f"missing predictions for {model_name}: {missing}")

    prediction_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        prediction_rows.append(
            {
                "schema_version": 1,
                "feature_index": index,
                "rollout_id": str(row["rollout_id"]),
                "problem_id": str(row["problem_id"]),
                "outer_fold": int(outer_assignments[index]),
                "hack_attempted": bool(labels[index]),
                "reward_hacked": bool(reward_hacked[index]),
                "cot_mentions_hack": bool(mentions[index]),
                "transparent": bool(transparent[index]),
                "behavior_group": str(behavior_groups[index]),
                "thinking_token_count": int(row["thinking_token_count"]),
                "predictions": {
                    model_name: float(predictions[model_name][index])
                    for model_name in MODEL_NAMES
                },
            }
        )
    predictions_path = output_dir / "oof_predictions.jsonl"
    write_jsonl(predictions_path, prediction_rows)

    subset_masks = {
        "transparent": transparent,
        "no_mention": ~mentions,
        "all_complete_cot": np.ones(len(rows), dtype=bool),
    }
    metrics: dict[str, Any] = {}
    for model_number, model_name in enumerate(MODEL_NAMES):
        model_metrics: dict[str, Any] = {}
        for subset_number, (subset_name, mask) in enumerate(subset_masks.items()):
            model_metrics[subset_name] = evaluate_subset(
                labels=labels,
                scores=predictions[model_name],
                groups=groups,
                mask=mask,
                replicates=args.bootstrap_replicates,
                seed=args.seed + model_number * 1000 + subset_number,
            )
        model_metrics["reward_hacked_secondary"] = evaluate_subset(
            labels=reward_hacked,
            scores=predictions[model_name],
            groups=groups,
            mask=np.ones(len(rows), dtype=bool),
            replicates=args.bootstrap_replicates,
            seed=args.seed + model_number * 1000 + 100,
        )
        metrics[model_name] = model_metrics

    group_summaries: dict[str, Any] = {}
    for model_name in MODEL_NAMES:
        group_summaries[model_name] = {}
        for group in GROUP_ORDER:
            values = predictions[model_name][behavior_groups == group]
            group_summaries[model_name][group] = {
                "n": int(len(values)),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
            }

    activation_primary = metrics["cot_activation"]["transparent"]
    promising = bool(
        activation_primary["auroc"] >= 0.70
        and activation_primary["auroc_ci95"][0] > 0.50
    )
    summary = {
        "schema_version": 1,
        "primary_model": "cot_activation",
        "primary_training_subset": "transparent examples only",
        "primary_label": "hack_attempted",
        "feature_layer": args.expected_layer,
        "pooling": "mean over all non-special tokens strictly inside complete CoT",
        "rollouts": len(rows),
        "problem_count": int(len(np.unique(groups))),
        "behavior_group_counts": {
            group: int(np.sum(behavior_groups == group)) for group in GROUP_ORDER
        },
        "metrics": metrics,
        "behavior_group_score_summaries": group_summaries,
        "per_fold": fold_results,
        "success_gate": {
            "required_auroc": 0.70,
            "required_grouped_95_ci_lower_strictly_above": 0.50,
            "observed_auroc": activation_primary["auroc"],
            "observed_auroc_ci95": activation_primary["auroc_ci95"],
            "promising": promising,
        },
        "interpretation": {
            "pilot_status": "promising" if promising else "did_not_pass_gate",
            "silent_attempt_result_is_exploratory": True,
            "layer_selection_warning": (
                "Layer 10 was selected after inspecting answer-side projections on "
                "these rollouts, so this pilot is exploratory."
            ),
            "negative_result_scope": (
                "A failure rejects this whole-CoT mean-pooling decoder, not the "
                "presence of all reward-hacking information in CoT activations."
            ),
        },
    }
    summary_path = output_dir / "summary.json"
    write_json(summary_path, summary)

    fold_csv_path = output_dir / "fold_metrics.csv"
    with fold_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "fold",
                "model",
                "chosen_C",
                "train_transparent_n",
                "train_problem_count",
                "heldout_all_n",
                "heldout_problem_count",
                "transparent_n",
                "transparent_auroc",
                "transparent_auprc",
                "no_mention_n",
                "no_mention_auroc",
                "no_mention_auprc",
            ],
        )
        writer.writeheader()
        for row in fold_results:
            no_mention = row["no_mention_metrics"] or {}
            writer.writerow(
                {
                    "fold": row["fold"],
                    "model": row["model"],
                    "chosen_C": row["chosen_C"],
                    "train_transparent_n": row["train_transparent_n"],
                    "train_problem_count": row["train_problem_count"],
                    "heldout_all_n": row["heldout_all_n"],
                    "heldout_problem_count": row["heldout_problem_count"],
                    "transparent_n": row["transparent_metrics"]["n"],
                    "transparent_auroc": row["transparent_metrics"]["auroc"],
                    "transparent_auprc": row["transparent_metrics"]["auprc"],
                    "no_mention_n": no_mention.get("n"),
                    "no_mention_auroc": no_mention.get("auroc"),
                    "no_mention_auprc": no_mention.get("auprc"),
                }
            )

    figure_path = output_dir / "cot_decoder_pilot.png"
    save_figure(
        path=figure_path,
        labels=labels,
        predictions=predictions,
        transparent=transparent,
        behavior_groups=behavior_groups,
        seed=args.seed,
    )

    manifest_path = output_dir / "manifest.json"
    manifest = {
        "schema_version": 1,
        "feature_file": str(feature_path),
        "feature_file_sha256": sha256_file(feature_path),
        "metadata_file": str(metadata_path),
        "metadata_file_sha256": sha256_file(metadata_path),
        "feature_manifest": str(feature_manifest_path),
        "feature_manifest_sha256": sha256_file(feature_manifest_path),
        "folds_file": folds_path.name,
        "folds_file_sha256": sha256_file(folds_path),
        "predictions_file": predictions_path.name,
        "predictions_file_sha256": sha256_file(predictions_path),
        "summary_file": summary_path.name,
        "summary_file_sha256": sha256_file(summary_path),
        "fold_metrics_file": fold_csv_path.name,
        "fold_metrics_file_sha256": sha256_file(fold_csv_path),
        "figure_file": figure_path.name,
        "figure_file_sha256": sha256_file(figure_path),
        "outer_folds": args.outer_folds,
        "inner_folds_requested": args.inner_folds,
        "c_grid": sorted(args.c_grid),
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "package_versions": package_versions(
            ["numpy", "scikit-learn", "scipy", "joblib", "safetensors", "matplotlib"]
        ),
        "model_files": {
            str(path.relative_to(output_dir)): sha256_file(path)
            for path in sorted(models_dir.glob("*.joblib"))
        },
    }
    write_json(manifest_path, manifest)

    checksum_paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    checksum_path = output_dir / "SHA256SUMS"
    checksum_path.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(output_dir)}\n"
            for path in checksum_paths
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary["success_gate"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
