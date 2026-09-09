#!/usr/bin/env python3
"""Confirm the frozen layer-10 mean-CoT decoder design on fresh problems."""

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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import train_cot_decoder as pilot
import train_cot_decoder_all_groups as all_groups


SPLIT_ORDER = ("train", "validation", "test")
SPLIT_COUNTS = {"train": 120, "validation": 40, "test": 40}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-features", type=Path, required=True)
    parser.add_argument("--pilot-metadata", type=Path, required=True)
    parser.add_argument("--pilot-feature-manifest", type=Path, required=True)
    parser.add_argument("--confirmation-features", type=Path, required=True)
    parser.add_argument("--confirmation-metadata", type=Path, required=True)
    parser.add_argument("--confirmation-feature-manifest", type=Path, required=True)
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--c-grid",
        type=float,
        nargs="+",
        default=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--split-search-iterations", type=int, default=20_000)
    parser.add_argument("--minimum-fresh-train-per-cell", type=int, default=60)
    parser.add_argument("--minimum-validation-per-cell", type=int, default=20)
    parser.add_argument("--minimum-test-per-cell", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-problems", type=int, default=200)
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
    rows = pilot.read_jsonl(path)
    for index, row in enumerate(rows):
        if int(row.get("feature_index", -1)) != index:
            raise ValueError(f"{path}: feature indices are not contiguous")
    return rows


def load_feature_bundle(
    *,
    feature_path: Path,
    metadata_path: Path,
    manifest_path: Path,
    expected_layer: int,
    expected_hidden_size: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("feature_file_sha256") != sha256_file(feature_path):
        raise ValueError(f"feature hash mismatch: {feature_path}")
    if manifest.get("metadata_file_sha256") != sha256_file(metadata_path):
        raise ValueError(f"metadata hash mismatch: {metadata_path}")
    if int(manifest.get("layer_index", -1)) != expected_layer:
        raise ValueError(f"wrong layer in {manifest_path}")
    rows = read_jsonl(metadata_path)
    tensors = {
        name: np.asarray(value, dtype=np.float32)
        for name, value in load_file(str(feature_path)).items()
    }
    if set(tensors) != {"cot_mean", "prompt_mean"}:
        raise ValueError(f"unexpected feature keys in {feature_path}: {sorted(tensors)}")
    expected_shape = (len(rows), expected_hidden_size)
    for name, values in tensors.items():
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise ValueError(f"invalid {name} features: {values.shape}")
    rollout_ids = [str(row["rollout_id"]) for row in rows]
    if len(rollout_ids) != len(set(rollout_ids)):
        raise ValueError(f"duplicate rollout IDs in {metadata_path}")
    return tensors, rows, manifest


def assert_identical_feature_provenance(
    pilot_manifest: dict[str, Any], confirmation_manifest: dict[str, Any]
) -> None:
    fields = (
        "base_model",
        "base_revision",
        "adapter",
        "adapter_revision",
        "chat_template_sha256",
        "layer_index",
        "layer_convention",
        "hidden_size",
        "layer_count",
        "token_scopes",
    )
    mismatches = [
        field
        for field in fields
        if pilot_manifest.get(field) != confirmation_manifest.get(field)
    ]
    if mismatches:
        raise ValueError(
            "pilot and confirmation features differ in provenance fields: "
            + ", ".join(mismatches)
        )


def assert_partition_minimums(
    behavior_counts: np.ndarray,
    *,
    minimum_fresh_train: int,
    minimum_validation: int,
    minimum_test: int,
) -> dict[str, int]:
    """Enforce four-cell representation in every confirmation partition."""
    expected_shape = (len(SPLIT_ORDER), len(pilot.GROUP_ORDER))
    if behavior_counts.shape != expected_shape:
        raise ValueError(
            f"behavior count matrix has shape {behavior_counts.shape}; "
            f"expected {expected_shape}"
        )
    required = {
        "train": int(minimum_fresh_train),
        "validation": int(minimum_validation),
        "test": int(minimum_test),
    }
    observed: dict[str, int] = {}
    for split_index, split in enumerate(SPLIT_ORDER):
        observed[split] = int(behavior_counts[split_index].min())
        if observed[split] < required[split]:
            counts = {
                group: int(behavior_counts[split_index, group_index])
                for group_index, group in enumerate(pilot.GROUP_ORDER)
            }
            raise ValueError(
                f"{split} has only {observed[split]} examples in its smallest "
                f"behavior cell; required {required[split]}; counts={counts}"
            )
    return observed


def load_generation_problem_ids(path: Path, expected: int) -> list[str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    values = manifest.get("selected_problem_ids")
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise ValueError("generation manifest has no valid selected_problem_ids list")
    if len(values) != expected or len(set(values)) != expected:
        raise ValueError(
            f"generation manifest must contain exactly {expected} unique problem IDs"
        )
    return values


def row_arrays(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "labels": np.asarray([bool(row["hack_attempted"]) for row in rows], dtype=np.int8),
        "reward_hacked": np.asarray(
            [bool(row["reward_hacked"]) for row in rows], dtype=np.int8
        ),
        "groups": np.asarray([str(row["problem_id"]) for row in rows], dtype=object),
        "mentions": np.asarray(
            [bool(row["cot_mentions_hack"]) for row in rows], dtype=bool
        ),
        "transparent": np.asarray([bool(row["transparent"]) for row in rows], dtype=bool),
        "behavior_groups": np.asarray(
            [str(row["behavior_group"]) for row in rows], dtype=object
        ),
        "token_count": np.log1p(
            np.asarray([int(row["thinking_token_count"]) for row in rows], dtype=np.float64)
        )[:, None],
        "texts": [str(row["thinking_text"]) for row in rows],
    }


def combine_values(first: Any, second: Any) -> Any:
    if isinstance(first, np.ndarray):
        return np.concatenate([first, second], axis=0)
    return [*first, *second]


def select_threshold(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    false_positive, true_positive, thresholds = roc_curve(labels, scores)
    finite = np.isfinite(thresholds)
    if not finite.any():
        raise ValueError("validation ROC produced no finite thresholds")
    youden = true_positive - false_positive
    candidates = np.flatnonzero(finite & np.isclose(youden, np.max(youden[finite])))
    chosen_index = min(
        candidates,
        key=lambda index: (abs(float(thresholds[index]) - 0.5), -float(thresholds[index])),
    )
    return {
        "threshold": float(thresholds[chosen_index]),
        "youden_j": float(youden[chosen_index]),
        "true_positive_rate": float(true_positive[chosen_index]),
        "false_positive_rate": float(false_positive[chosen_index]),
    }


def threshold_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, Any]:
    predicted = scores >= threshold
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "confusion_matrix_tn_fp_fn_tp": [int(value) for value in matrix.ravel()],
    }


def evaluate_mask(
    *,
    labels: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    mask: np.ndarray,
    threshold: float,
    replicates: int,
    seed: int,
) -> dict[str, Any] | None:
    selected = np.flatnonzero(mask)
    if len(selected) == 0 or np.unique(labels[selected]).size != 2:
        return None
    result = pilot.binary_metrics(labels[selected], scores[selected])
    result.update(
        pilot.grouped_bootstrap_metrics(
            labels=labels[selected],
            scores=scores[selected],
            groups=groups[selected],
            replicates=replicates,
            seed=seed,
        )
    )
    result["threshold_metrics"] = threshold_metrics(
        labels[selected], scores[selected], threshold
    )
    return result


def tune_on_validation(
    *,
    values: Any,
    labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    estimator_factory: Callable[[float, int], Any],
    c_grid: Sequence[float],
    seed: int,
) -> tuple[float, Any, np.ndarray, list[dict[str, float]]]:
    results: list[dict[str, float]] = []
    fitted: dict[float, Any] = {}
    validation_scores: dict[float, np.ndarray] = {}
    for number, c_value in enumerate(sorted(c_grid)):
        estimator = estimator_factory(float(c_value), seed + number)
        estimator.fit(pilot.subset(values, train_indices), labels[train_indices])
        scores = estimator.predict_proba(pilot.subset(values, validation_indices))[:, 1]
        auc = float(roc_auc_score(labels[validation_indices], scores))
        ap = float(average_precision_score(labels[validation_indices], scores))
        results.append({"C": float(c_value), "validation_auroc": auc, "validation_auprc": ap})
        fitted[float(c_value)] = estimator
        validation_scores[float(c_value)] = scores
    best = max(results, key=lambda row: (row["validation_auroc"], -row["C"]))
    chosen_c = float(best["C"])
    return chosen_c, fitted[chosen_c], validation_scores[chosen_c], results


def save_figure(
    *,
    path: Path,
    arrays: dict[str, Any],
    predictions: dict[str, np.ndarray],
    split_labels: np.ndarray,
    seed: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    test = split_labels == "test"
    labels = arrays["labels"][test]
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 11.0))
    for name in all_groups.MODEL_NAMES:
        scores = predictions[name][test]
        false_positive, true_positive, _ = roc_curve(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)
        axes[0, 0].plot(
            false_positive,
            true_positive,
            color=all_groups.COLORS[name],
            linewidth=2,
            label=f"{all_groups.MODEL_LABELS[name]} ({auc:.3f})",
        )
        axes[0, 1].plot(
            recall,
            precision,
            color=all_groups.COLORS[name],
            linewidth=2,
            label=f"{all_groups.MODEL_LABELS[name]} ({ap:.3f})",
        )
    axes[0, 0].plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.55)
    axes[0, 0].set(
        title="All four groups: fresh test ROC",
        xlabel="False-positive rate",
        ylabel="True-positive rate",
    )
    axes[0, 0].legend(fontsize=8, loc="lower right")
    axes[0, 1].axhline(float(labels.mean()), linestyle="--", color="black", alpha=0.55)
    axes[0, 1].set(
        title="All four groups: fresh test precision-recall",
        xlabel="Recall",
        ylabel="Precision",
    )
    axes[0, 1].legend(fontsize=8, loc="lower left")

    activation_scores = predictions["cot_activation"]
    for stratum, label, color in (
        (~arrays["mentions"], "No hack mention", "#2b7a9b"),
        (arrays["mentions"], "Hack mentioned", "#b4483e"),
    ):
        mask = test & stratum
        false_positive, true_positive, _ = roc_curve(
            arrays["labels"][mask], activation_scores[mask]
        )
        auc = roc_auc_score(arrays["labels"][mask], activation_scores[mask])
        axes[1, 0].plot(
            false_positive,
            true_positive,
            color=color,
            linewidth=2.3,
            label=f"{label} ({auc:.3f})",
        )
    axes[1, 0].plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.55)
    axes[1, 0].set(
        title="Does CoT activation predict the answer within each stratum?",
        xlabel="False-positive rate",
        ylabel="True-positive rate",
    )
    axes[1, 0].legend(fontsize=9, loc="lower right")

    values = [
        activation_scores[test & (arrays["behavior_groups"] == group)]
        for group in pilot.GROUP_ORDER
    ]
    box = axes[1, 1].boxplot(values, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], ["#dce8ed", "#f0c8c2", "#f4ddad", "#d9d0ea"]):
        patch.set_facecolor(color)
    rng = np.random.default_rng(seed)
    for position, group_values in enumerate(values, 1):
        jitter = rng.normal(0, 0.045, size=len(group_values))
        axes[1, 1].scatter(
            np.full(len(group_values), position) + jitter,
            group_values,
            s=14,
            alpha=0.5,
            color="#29465a",
            edgecolors="none",
        )
    axes[1, 1].set_xticks(range(1, len(pilot.GROUP_ORDER) + 1))
    axes[1, 1].set_xticklabels(
        [pilot.GROUP_LABELS[group] for group in pilot.GROUP_ORDER], fontsize=8
    )
    axes[1, 1].set(
        title="Fresh test scores by all four behavior groups",
        ylabel="Hack score",
    )
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.suptitle(
        "Layer-10 mean-pooled CoT decoder: balanced fresh confirmation",
        fontsize=16,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    paths = {
        name: value.expanduser().resolve()
        for name, value in vars(args).items()
        if isinstance(value, Path)
    }
    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise SystemExit(f"output directory is not empty: {output_dir}")
    models_dir = output_dir / "models"
    models_dir.mkdir()

    pilot_features, pilot_rows, pilot_manifest = load_feature_bundle(
        feature_path=paths["pilot_features"],
        metadata_path=paths["pilot_metadata"],
        manifest_path=paths["pilot_feature_manifest"],
        expected_layer=args.expected_layer,
        expected_hidden_size=args.expected_hidden_size,
    )
    fresh_features, fresh_rows, fresh_manifest = load_feature_bundle(
        feature_path=paths["confirmation_features"],
        metadata_path=paths["confirmation_metadata"],
        manifest_path=paths["confirmation_feature_manifest"],
        expected_layer=args.expected_layer,
        expected_hidden_size=args.expected_hidden_size,
    )
    assert_identical_feature_provenance(pilot_manifest, fresh_manifest)

    generation_ids = load_generation_problem_ids(
        paths["generation_manifest"], args.expected_problems
    )
    pilot_problem_ids = {str(row["problem_id"]) for row in pilot_rows}
    if pilot_problem_ids & set(generation_ids):
        raise SystemExit("pilot and fresh confirmation problem IDs overlap")
    fresh_problem_ids = {str(row["problem_id"]) for row in fresh_rows}
    if not fresh_problem_ids <= set(generation_ids):
        raise SystemExit("feature metadata contains a problem outside generation manifest")

    pilot_arrays = row_arrays(pilot_rows)
    fresh_arrays = row_arrays(fresh_rows)
    fresh_splits, split_mapping, split_counts, split_diagnostics = (
        all_groups.balanced_problem_partitions(
            fresh_rows,
            all_problem_ids=generation_ids,
            partition_sizes=SPLIT_COUNTS,
            minimum_counts={
                "train": args.minimum_fresh_train_per_cell,
                "validation": args.minimum_validation_per_cell,
                "test": args.minimum_test_per_cell,
            },
            seed=args.seed,
            search_iterations=args.split_search_iterations,
        )
    )
    if tuple(split_diagnostics["partition_order"]) != SPLIT_ORDER:
        raise AssertionError("balanced partition order differs from SPLIT_ORDER")
    try:
        observed_minimums = assert_partition_minimums(
            split_counts,
            minimum_fresh_train=args.minimum_fresh_train_per_cell,
            minimum_validation=args.minimum_validation_per_cell,
            minimum_test=args.minimum_test_per_cell,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    train_fresh = fresh_splits == "train"
    validation = fresh_splits == "validation"
    test = fresh_splits == "test"
    for name, mask in (
        ("fresh training", train_fresh),
        ("validation", validation),
        ("test", test),
    ):
        if np.unique(fresh_arrays["labels"][mask]).size != 2:
            raise SystemExit(f"{name} subset does not contain both classes")

    # The combined arrays put the pilot first. Every complete-CoT pilot row and
    # every row from fresh training problems is eligible for fitting.
    combined_labels = np.concatenate(
        [pilot_arrays["labels"], fresh_arrays["labels"]], axis=0
    )
    combined_groups = np.concatenate(
        [pilot_arrays["groups"], fresh_arrays["groups"]], axis=0
    )
    offset = len(pilot_rows)
    train_indices = np.concatenate(
        [
            np.arange(len(pilot_rows)),
            offset + np.flatnonzero(train_fresh),
        ]
    )
    validation_indices = offset + np.flatnonzero(validation)
    if set(combined_groups[train_indices]) & set(combined_groups[validation_indices]):
        raise SystemExit("problem overlap between fitting and validation data")
    test_groups = set(fresh_arrays["groups"][test])
    if set(combined_groups[train_indices]) & test_groups:
        raise SystemExit("problem overlap between fitting and test data")

    model_values = {
        "cot_activation": combine_values(
            pilot_features["cot_mean"], fresh_features["cot_mean"]
        ),
        "prompt_activation": combine_values(
            pilot_features["prompt_mean"], fresh_features["prompt_mean"]
        ),
        "cot_token_count": combine_values(
            pilot_arrays["token_count"], fresh_arrays["token_count"]
        ),
        "cot_tfidf": combine_values(pilot_arrays["texts"], fresh_arrays["texts"]),
        "cot_mention_only": combine_values(
            pilot_arrays["mentions"].astype(np.float64)[:, None],
            fresh_arrays["mentions"].astype(np.float64)[:, None],
        ),
    }
    factories: dict[str, Callable[[float, int], Any]] = {
        "cot_activation": pilot.dense_estimator,
        "prompt_activation": pilot.dense_estimator,
        "cot_token_count": pilot.dense_estimator,
        "cot_tfidf": pilot.text_estimator,
    }
    predictions: dict[str, np.ndarray] = {}
    model_summaries: dict[str, Any] = {}
    for model_number, model_name in enumerate(all_groups.MODEL_NAMES):
        model_path = models_dir / f"{model_name}.joblib"
        if model_name == "cot_mention_only":
            chosen_c = None
            tuning = []
            validation_scores = fresh_arrays["mentions"][validation].astype(
                np.float64
            )
            fresh_scores = fresh_arrays["mentions"].astype(np.float64)
            validation_threshold = threshold_metrics(
                fresh_arrays["labels"][validation], validation_scores, 0.5
            )
            true_negative, false_positive, _false_negative, true_positive = (
                validation_threshold["confusion_matrix_tn_fp_fn_tp"]
            )
            threshold_selection = {
                "threshold": 0.5,
                "youden_j": float(
                    true_positive / (true_positive + _false_negative)
                    - false_positive / (false_positive + true_negative)
                ),
                "true_positive_rate": float(
                    true_positive / (true_positive + _false_negative)
                ),
                "false_positive_rate": float(
                    false_positive / (false_positive + true_negative)
                ),
            }
            joblib.dump(
                {
                    "control": "raw binary cot_mentions_hack indicator",
                    "requires_fitting": False,
                    "threshold": 0.5,
                },
                model_path,
                compress=3,
            )
        else:
            values = model_values[model_name]
            chosen_c, estimator, validation_scores, tuning = tune_on_validation(
                values=values,
                labels=combined_labels,
                train_indices=train_indices,
                validation_indices=validation_indices,
                estimator_factory=factories[model_name],
                c_grid=args.c_grid,
                seed=args.seed + 100 * model_number,
            )
            threshold_selection = select_threshold(
                combined_labels[validation_indices], validation_scores
            )
            fresh_scores = estimator.predict_proba(
                pilot.subset(values, offset + np.arange(len(fresh_rows)))
            )[:, 1]
            joblib.dump(estimator, model_path, compress=3)
        predictions[model_name] = fresh_scores
        model_summaries[model_name] = {
            "chosen_C": chosen_c,
            "selection_source": "all complete-CoT examples from fresh validation problems",
            "validation_tuning": tuning,
            "threshold_selection": threshold_selection,
            "model_file": str(model_path.relative_to(output_dir)),
        }
        print(
            f"model={model_name} "
            f"C={'fixed' if chosen_c is None else f'{chosen_c:g}'} threshold="
            f"{threshold_selection['threshold']:.5f}",
            flush=True,
        )

    # Export the primary linear decoder and preprocessing arrays explicitly in
    # addition to the complete, executable sklearn pipeline.
    primary_pipeline = joblib.load(models_dir / "cot_activation.joblib")
    scaler = primary_pipeline.named_steps["scale"]
    classifier = primary_pipeline.named_steps["classifier"]
    state_path = output_dir / "cot_activation_state.npz"
    np.savez_compressed(
        state_path,
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
        scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
        coefficient=np.asarray(classifier.coef_, dtype=np.float64),
        intercept=np.asarray(classifier.intercept_, dtype=np.float64),
        classes=np.asarray(classifier.classes_),
        threshold=np.asarray(
            [model_summaries["cot_activation"]["threshold_selection"]["threshold"]],
            dtype=np.float64,
        ),
    )

    masks = {
        "validation_all": validation,
        "test_all_complete_cot": test,
        "test_no_mention": test & ~fresh_arrays["mentions"],
        "test_mention": test & fresh_arrays["mentions"],
    }
    metrics: dict[str, Any] = {}
    for model_number, model_name in enumerate(all_groups.MODEL_NAMES):
        threshold = model_summaries[model_name]["threshold_selection"]["threshold"]
        metrics[model_name] = {}
        for subset_number, (subset_name, mask) in enumerate(masks.items()):
            metrics[model_name][subset_name] = evaluate_mask(
                labels=fresh_arrays["labels"],
                scores=predictions[model_name],
                groups=fresh_arrays["groups"],
                mask=mask,
                threshold=threshold,
                replicates=args.bootstrap_replicates,
                seed=args.seed + model_number * 1000 + subset_number,
            )
        metrics[model_name]["test_reward_hacked_secondary"] = evaluate_mask(
            labels=fresh_arrays["reward_hacked"],
            scores=predictions[model_name],
            groups=fresh_arrays["groups"],
            mask=test,
            threshold=threshold,
            replicates=args.bootstrap_replicates,
            seed=args.seed + model_number * 1000 + 100,
        )

    prediction_rows = []
    for index, row in enumerate(fresh_rows):
        prediction_rows.append(
            {
                "schema_version": 1,
                "feature_index": index,
                "rollout_id": str(row["rollout_id"]),
                "problem_id": str(row["problem_id"]),
                "split": str(fresh_splits[index]),
                "hack_attempted": bool(fresh_arrays["labels"][index]),
                "reward_hacked": bool(fresh_arrays["reward_hacked"][index]),
                "cot_mentions_hack": bool(fresh_arrays["mentions"][index]),
                "transparent": bool(fresh_arrays["transparent"][index]),
                "behavior_group": str(fresh_arrays["behavior_groups"][index]),
                "thinking_token_count": int(row["thinking_token_count"]),
                "predictions": {
                    name: float(predictions[name][index])
                    for name in all_groups.MODEL_NAMES
                },
            }
        )
    predictions_path = output_dir / "predictions.jsonl"
    write_jsonl(predictions_path, prediction_rows)

    split_path = output_dir / "problem_splits.json"
    write_json(
        split_path,
        {
            "schema_version": 1,
            "seed": args.seed,
            "assignment_unit": "problem_id",
            "counts": SPLIT_COUNTS,
            "stratification": "four CoT-mention by answer-attempt behavior cells",
            "problem_to_split": dict(sorted(split_mapping.items())),
            "constraints": {
                "minimum_fresh_train_per_behavior_cell": args.minimum_fresh_train_per_cell,
                "minimum_validation_per_behavior_cell": args.minimum_validation_per_cell,
                "minimum_test_per_behavior_cell": args.minimum_test_per_cell,
            },
            "observed_minimums": observed_minimums,
            "diagnostics": split_diagnostics,
            "pilot_data_role": "all complete-CoT pilot rows are additional training-only data",
            "pilot_problem_count": len(pilot_problem_ids),
            "pilot_fresh_overlap_count": 0,
        },
    )

    group_summaries: dict[str, Any] = {}
    test_mask = test
    for model_name in all_groups.MODEL_NAMES:
        group_summaries[model_name] = {}
        for group in pilot.GROUP_ORDER:
            values = predictions[model_name][
                test_mask & (fresh_arrays["behavior_groups"] == group)
            ]
            group_summaries[model_name][group] = {
                "n": int(len(values)),
                "mean": float(np.mean(values)) if len(values) else None,
                "median": float(np.median(values)) if len(values) else None,
                "q25": float(np.quantile(values, 0.25)) if len(values) else None,
                "q75": float(np.quantile(values, 0.75)) if len(values) else None,
            }

    primary = metrics["cot_activation"]["test_all_complete_cot"]
    if primary is None:
        raise SystemExit("cannot evaluate the primary fresh all-groups test subset")
    confirmed = bool(primary["auroc"] >= 0.70 and primary["auroc_ci95"][0] > 0.50)
    fit_behavior_counts = {
        group: int(
            np.sum(pilot_arrays["behavior_groups"] == group)
            + np.sum(
                train_fresh & (fresh_arrays["behavior_groups"] == group)
            )
        )
        for group in pilot.GROUP_ORDER
    }
    summary = {
        "schema_version": 1,
        "experiment": "layer-10 mean-pooled CoT decoder confirmation",
        "primary_label": "hack_attempted",
        "primary_training_subset": "all complete-CoT examples from all four behavior groups",
        "feature_layer": args.expected_layer,
        "pooling": "mean over all non-special tokens strictly inside complete CoT",
        "fresh_eligible_rollouts": len(fresh_rows),
        "fresh_selected_problem_count": len(generation_ids),
        "fresh_problems_with_eligible_cot": len(fresh_problem_ids),
        "pilot_eligible_rollouts": len(pilot_rows),
        "fit_counts": {
            "pilot_all_complete_cot": int(len(pilot_rows)),
            "fresh_train_all_complete_cot": int(train_fresh.sum()),
            "total": int(len(train_indices)),
        },
        "fit_behavior_group_counts": fit_behavior_counts,
        "fresh_split_eligible_counts": {
            split: int(np.sum(fresh_splits == split)) for split in SPLIT_ORDER
        },
        "fresh_behavior_group_counts": {
            split: {
                group: int(
                    np.sum(
                        (fresh_splits == split)
                        & (fresh_arrays["behavior_groups"] == group)
                    )
                )
                for group in pilot.GROUP_ORDER
            }
            for split in SPLIT_ORDER
        },
        "split_constraints_satisfied": {
            "required_minimums": {
                "fresh_train": args.minimum_fresh_train_per_cell,
                "validation": args.minimum_validation_per_cell,
                "test": args.minimum_test_per_cell,
            },
            "observed_minimums": observed_minimums,
            "all_satisfied": True,
        },
        "model_selection": model_summaries,
        "metrics": metrics,
        "test_behavior_group_score_summaries": group_summaries,
        "success_gate": {
            "required_test_all_groups_auroc": 0.70,
            "required_grouped_95_ci_lower_strictly_above": 0.50,
            "observed_auroc": primary["auroc"],
            "observed_auroc_ci95": primary["auroc_ci95"],
            "confirmed": confirmed,
        },
        "interpretation": {
            "confirmation_status": "confirmed" if confirmed else "not_confirmed",
            "silent_attempt_result_is_exploratory": True,
            "cot_mention_is_not_used_to_select_training_examples": True,
            "all_four_behavior_groups_appear_in_fresh_train_validation_and_test": True,
            "test_set_used_once_after_validation_selection": True,
            "negative_result_scope": (
                "A failure rejects this whole-CoT mean-pooling decoder, not all "
                "possible reward-hacking information in CoT activations."
            ),
        },
    }
    summary_path = output_dir / "summary.json"
    write_json(summary_path, summary)

    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "subset",
                "n",
                "n_positive",
                "n_negative",
                "auroc",
                "auroc_ci95_low",
                "auroc_ci95_high",
                "auprc",
                "auprc_ci95_low",
                "auprc_ci95_high",
                "threshold",
                "balanced_accuracy",
            ],
        )
        writer.writeheader()
        for model_name, subsets in metrics.items():
            for subset_name, result in subsets.items():
                if result is None:
                    continue
                writer.writerow(
                    {
                        "model": model_name,
                        "subset": subset_name,
                        "n": result["n"],
                        "n_positive": result["n_positive"],
                        "n_negative": result["n_negative"],
                        "auroc": result["auroc"],
                        "auroc_ci95_low": result["auroc_ci95"][0],
                        "auroc_ci95_high": result["auroc_ci95"][1],
                        "auprc": result["auprc"],
                        "auprc_ci95_low": result["auprc_ci95"][0],
                        "auprc_ci95_high": result["auprc_ci95"][1],
                        "threshold": result["threshold_metrics"]["threshold"],
                        "balanced_accuracy": result["threshold_metrics"][
                            "balanced_accuracy"
                        ],
                    }
                )

    figure_path = output_dir / "cot_decoder_confirmation.png"
    save_figure(
        path=figure_path,
        arrays=fresh_arrays,
        predictions=predictions,
        split_labels=fresh_splits,
        seed=args.seed,
    )

    manifest_path = output_dir / "manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "status": "complete",
            "files": {
                "summary": summary_path.name,
                "metrics": metrics_path.name,
                "predictions": predictions_path.name,
                "splits": split_path.name,
                "figure": figure_path.name,
                "primary_decoder_state": state_path.name,
            },
            "input_hashes": {
                name: sha256_file(path)
                for name, path in paths.items()
                if name != "output_dir"
            },
            "seed": args.seed,
            "bootstrap_replicates": args.bootstrap_replicates,
            "c_grid": sorted(float(value) for value in args.c_grid),
            "threshold_rule": (
                "maximize Youden J on all complete-CoT examples from fresh "
                "validation problems; ties choose threshold closest to 0.5, "
                "then higher threshold; the raw mention control stays fixed at 0.5"
            ),
            "package_versions": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "scikit-learn", "safetensors", "joblib", "matplotlib")
            },
        },
    )

    hash_path = output_dir / "SHA256SUMS"
    artifact_paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path != hash_path
    )
    hash_path.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(output_dir)}\n"
            for path in artifact_paths
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary["success_gate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
