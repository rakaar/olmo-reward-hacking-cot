#!/usr/bin/env python3
"""Compare per-layer maximum shortcut-direction projections on saved rollouts.

This is an observational analysis: it teacher-forces existing OLMo responses and
never generates new text or modifies model activations.  At every transformer
block, a forward hook projects response-token residuals onto the matching unit
direction and immediately retains only the signed maximum and its token index.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np


THINKING_RE = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-rollouts", type=Path, required=True)
    parser.add_argument("--direction-file", type=Path, required=True)
    parser.add_argument("--direction-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--adapter-revision", required=True)
    parser.add_argument(
        "--base-load-path",
        help="Optional local snapshot path; base-model remains the recorded identity",
    )
    parser.add_argument(
        "--adapter-load-path",
        help="Optional local snapshot path; adapter remains the recorded identity",
    )
    parser.add_argument(
        "--tokenizer-load-path",
        help="Optional local tokenizer snapshot (defaults to adapter-load-path/adapter)",
    )
    parser.add_argument("--direction-key", default="mean_unit")
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--expected-layers", type=int, default=32)
    parser.add_argument("--expected-hidden-size", type=int, default=4096)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
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
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def package_versions(names: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def read_rollouts(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            missing = [
                key
                for key in (
                    "rollout_id",
                    "problem_id",
                    "system_prompt",
                    "problem_prompt",
                    "completion",
                    "hack_attempted",
                    "reward_hacked",
                )
                if key not in row
            ]
            if missing:
                raise ValueError(f"{path}:{line_number}: missing {missing}")
            if not isinstance(row["completion"], str):
                raise ValueError(f"{path}:{line_number}: completion must be text")
            if not isinstance(row["hack_attempted"], bool):
                raise ValueError(f"{path}:{line_number}: hack_attempted must be boolean")
            if not isinstance(row["reward_hacked"], bool):
                raise ValueError(f"{path}:{line_number}: reward_hacked must be boolean")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    ids = [str(row["rollout_id"]) for row in rows]
    if not rows:
        raise ValueError(f"no rollouts found in {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate rollout IDs in {path}")
    return rows


@dataclass(frozen=True)
class EncodedRollout:
    input_ids: list[int]
    response_indices: list[int]
    thinking_indices: list[int]
    response_boundary_tokens_excluded: int
    thinking_boundary_tokens_excluded: int
    has_complete_thinking_span: bool


def indices_inside_span(
    *,
    input_ids: list[int],
    offsets: list[tuple[int, int]],
    special_ids: set[int],
    span_start: int,
    span_end: int,
) -> tuple[list[int], int]:
    indices = [
        index
        for index, (token_id, (start, end)) in enumerate(zip(input_ids, offsets))
        if token_id not in special_ids
        and end > start
        and start >= span_start
        and end <= span_end
    ]
    boundary_excluded = sum(
        token_id not in special_ids
        and end > start
        and start < span_end
        and end > span_start
        and not (start >= span_start and end <= span_end)
        for token_id, (start, end) in zip(input_ids, offsets)
    )
    return indices, boundary_excluded


def encode_rollout(tokenizer: Any, row: dict[str, Any]) -> EncodedRollout:
    prefix_messages = [
        {"role": "system", "content": str(row["system_prompt"])},
        {"role": "user", "content": str(row["problem_prompt"])},
    ]
    completion = str(row["completion"])
    prompt = tokenizer.apply_chat_template(
        prefix_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full = tokenizer.apply_chat_template(
        [*prefix_messages, {"role": "assistant", "content": completion}],
        tokenize=False,
        add_generation_prompt=False,
    )
    if not full.startswith(prompt):
        mismatch = next(
            (i for i, (left, right) in enumerate(zip(prompt, full)) if left != right),
            min(len(prompt), len(full)),
        )
        raise ValueError(f"assistant prefix mismatch at character {mismatch}")
    response_start = len(prompt)
    response_end = response_start + len(completion)
    if full[response_start:response_end] != completion:
        raise ValueError("chat template altered assistant content")

    encoded = tokenizer(
        full,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
    )
    input_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    special_ids = {int(value) for value in tokenizer.all_special_ids}
    response_indices, response_boundary = indices_inside_span(
        input_ids=input_ids,
        offsets=offsets,
        special_ids=special_ids,
        span_start=response_start,
        span_end=response_end,
    )
    if not response_indices:
        raise ValueError("assistant response produced no non-special tokens")

    match = THINKING_RE.search(completion)
    thinking_indices: list[int] = []
    thinking_boundary = 0
    if match is not None and match.start(1) < match.end(1):
        thinking_indices, thinking_boundary = indices_inside_span(
            input_ids=input_ids,
            offsets=offsets,
            special_ids=special_ids,
            span_start=response_start + match.start(1),
            span_end=response_start + match.end(1),
        )
    return EncodedRollout(
        input_ids=input_ids,
        response_indices=response_indices,
        thinking_indices=thinking_indices,
        response_boundary_tokens_excluded=response_boundary,
        thinking_boundary_tokens_excluded=thinking_boundary,
        has_complete_thinking_span=match is not None,
    )


def resolve_decoder_and_layers(model: Any) -> tuple[Any, Any]:
    causal_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    candidates = [
        getattr(causal_model, "model", None),
        getattr(getattr(causal_model, "model", None), "model", None),
        getattr(causal_model, "transformer", None),
    ]
    for decoder in candidates:
        if decoder is None:
            continue
        layers = getattr(decoder, "layers", None)
        if layers is None:
            layers = getattr(decoder, "h", None)
        if layers is not None:
            return decoder, layers
    raise ValueError(f"could not locate decoder layers in {type(causal_model).__name__}")


class MaxProjectionPooler:
    """Pool all layer outputs to signed maxima without retaining activations."""

    def __init__(self, layers: Any, unit_directions: Any) -> None:
        import torch

        self.layers = list(layers)
        if unit_directions.ndim != 2 or unit_directions.shape[0] != len(self.layers):
            raise ValueError("direction tensor must have one row per layer")
        norms = torch.linalg.vector_norm(unit_directions.float(), dim=1)
        if not bool(torch.isfinite(norms).all()) or not bool((norms > 0).all()):
            raise ValueError("all directions must be finite and nonzero")
        self.unit_directions = unit_directions.float() / norms[:, None]
        self.scope_indices: dict[str, Any] = {}
        self.maxima: dict[str, list[float | None]] = {}
        self.argmax_indices: dict[str, list[int | None]] = {}
        self.handles = [
            layer.register_forward_hook(self._hook(index))
            for index, layer in enumerate(self.layers)
        ]

    def _hook(self, layer_index: int):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if not hasattr(hidden, "ndim") or hidden.ndim != 3 or hidden.shape[0] != 1:
                raise RuntimeError(
                    f"layer {layer_index} emitted unexpected shape "
                    f"{getattr(hidden, 'shape', None)}"
                )
            direction = self.unit_directions[layer_index]
            projection = hidden[0].float().matmul(direction)
            for scope, indices in self.scope_indices.items():
                if indices.numel() == 0:
                    continue
                selected = projection.index_select(0, indices)
                maximum, relative_index = selected.max(dim=0)
                full_index = indices[relative_index]
                self.maxima[scope][layer_index] = float(maximum.item())
                self.argmax_indices[scope][layer_index] = int(full_index.item())

        return capture

    def begin(self, scope_indices: dict[str, Any]) -> None:
        self.scope_indices = scope_indices
        self.maxima = {scope: [None] * len(self.layers) for scope in scope_indices}
        self.argmax_indices = {
            scope: [None] * len(self.layers) for scope in scope_indices
        }

    def finish(self) -> tuple[dict[str, list[float | None]], dict[str, list[int | None]]]:
        if any(value is None for value in self.maxima.get("response", [])):
            missing = [
                index
                for index, value in enumerate(self.maxima.get("response", []))
                if value is None
            ]
            raise RuntimeError(f"missing response maxima for layers {missing}")
        maxima = {scope: list(values) for scope, values in self.maxima.items()}
        argmax = {
            scope: list(values) for scope, values in self.argmax_indices.items()
        }
        self.scope_indices = {}
        return maxima, argmax

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.add(str(json.loads(line)["rollout_id"]))
    return ids


def load_projection_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def grouped_bootstrap(
    values: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Bootstrap group means and their difference by problem ID."""
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    by_group = [np.flatnonzero(groups == group) for group in unique_groups]
    positive = np.empty((replicates, values.shape[1]), dtype=np.float64)
    negative = np.empty_like(positive)
    difference = np.empty_like(positive)
    kept = 0
    attempts = 0
    while kept < replicates and attempts < replicates * 20:
        attempts += 1
        sampled_groups = rng.integers(0, len(by_group), size=len(by_group))
        indices = np.concatenate([by_group[index] for index in sampled_groups])
        sampled_labels = labels[indices]
        if not sampled_labels.any() or sampled_labels.all():
            continue
        pos_mean = values[indices][sampled_labels].mean(axis=0)
        neg_mean = values[indices][~sampled_labels].mean(axis=0)
        positive[kept] = pos_mean
        negative[kept] = neg_mean
        difference[kept] = pos_mean - neg_mean
        kept += 1
    if kept != replicates:
        raise RuntimeError(f"only obtained {kept}/{replicates} valid bootstraps")
    return {
        "positive": positive,
        "negative": negative,
        "difference": difference,
    }


def within_group_differences(
    values: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    """Return one positive-minus-negative vector per mixed-label problem."""
    differences: list[np.ndarray] = []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        group_labels = labels[indices]
        if not group_labels.any() or group_labels.all():
            continue
        group_values = values[indices]
        differences.append(
            group_values[group_labels].mean(axis=0)
            - group_values[~group_labels].mean(axis=0)
        )
    if not differences:
        raise ValueError("no mixed-label problems available for within-problem comparison")
    return np.asarray(differences, dtype=np.float64)


def bootstrap_group_difference_mean(
    differences: np.ndarray, replicates: int, seed: int
) -> np.ndarray:
    """Bootstrap the equal-weight mean of within-problem differences."""
    rng = np.random.default_rng(seed)
    sampled = rng.integers(
        0, len(differences), size=(replicates, len(differences))
    )
    return differences[sampled].mean(axis=1)


def scope_matrix(
    rows: list[dict[str, Any]], scope: str
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.asarray(
        [all(value is not None for value in row["max_projection"][scope]) for row in rows],
        dtype=bool,
    )
    matrix = np.asarray(
        [row["max_projection"][scope] for row, keep in zip(rows, valid) if keep],
        dtype=np.float64,
    )
    return matrix, valid


def summarize(
    *,
    rows: list[dict[str, Any]],
    output_dir: Path,
    layer_count: int,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    csv_rows: list[dict[str, Any]] = []
    split_summaries: dict[str, Any] = {}
    for scope_index, scope in enumerate(("thinking", "response")):
        values, valid = scope_matrix(rows, scope)
        valid_rows = [row for row, keep in zip(rows, valid) if keep]
        if values.shape[1:] != (layer_count,):
            raise ValueError(f"invalid {scope} matrix shape {values.shape}")
        for label_index, label_name in enumerate(("hack_attempted", "reward_hacked")):
            labels = np.asarray([bool(row[label_name]) for row in valid_rows])
            groups = np.asarray([str(row["problem_id"]) for row in valid_rows])
            positive = values[labels]
            negative = values[~labels]
            if len(positive) == 0 or len(negative) == 0:
                raise ValueError(f"{scope}/{label_name} lacks both classes")
            boot = grouped_bootstrap(
                values,
                labels,
                groups,
                bootstrap_replicates,
                seed + scope_index * 1000 + label_index * 100,
            )
            pos_mean = positive.mean(axis=0)
            neg_mean = negative.mean(axis=0)
            difference = pos_mean - neg_mean
            pos_ci = np.percentile(boot["positive"], [2.5, 97.5], axis=0)
            neg_ci = np.percentile(boot["negative"], [2.5, 97.5], axis=0)
            diff_ci = np.percentile(boot["difference"], [2.5, 97.5], axis=0)
            within_differences = within_group_differences(values, labels, groups)
            within_mean = within_differences.mean(axis=0)
            within_boot = bootstrap_group_difference_mean(
                within_differences,
                bootstrap_replicates,
                seed + scope_index * 1000 + label_index * 100 + 17,
            )
            within_ci = np.percentile(within_boot, [2.5, 97.5], axis=0)
            for layer in range(layer_count):
                csv_rows.append(
                    {
                        "scope": scope,
                        "label": label_name,
                        "layer": layer,
                        "n_positive": len(positive),
                        "n_negative": len(negative),
                        "positive_mean": float(pos_mean[layer]),
                        "positive_mean_ci_low": float(pos_ci[0, layer]),
                        "positive_mean_ci_high": float(pos_ci[1, layer]),
                        "negative_mean": float(neg_mean[layer]),
                        "negative_mean_ci_low": float(neg_ci[0, layer]),
                        "negative_mean_ci_high": float(neg_ci[1, layer]),
                        "mean_difference": float(difference[layer]),
                        "mean_difference_ci_low": float(diff_ci[0, layer]),
                        "mean_difference_ci_high": float(diff_ci[1, layer]),
                        "mixed_problem_count": len(within_differences),
                        "within_problem_mean_difference": float(within_mean[layer]),
                        "within_problem_mean_difference_ci_low": float(
                            within_ci[0, layer]
                        ),
                        "within_problem_mean_difference_ci_high": float(
                            within_ci[1, layer]
                        ),
                        "within_problem_median_difference": float(
                            np.median(within_differences[:, layer])
                        ),
                        "positive_median": float(np.median(positive[:, layer])),
                        "negative_median": float(np.median(negative[:, layer])),
                        "positive_q25": float(np.percentile(positive[:, layer], 25)),
                        "positive_q75": float(np.percentile(positive[:, layer], 75)),
                        "negative_q25": float(np.percentile(negative[:, layer], 25)),
                        "negative_q75": float(np.percentile(negative[:, layer], 75)),
                    }
                )
            best = int(np.argmax(difference))
            within_best = int(np.argmax(within_mean))
            positive_ci_layers = [
                int(layer) for layer in np.flatnonzero(diff_ci[0] > 0)
            ]
            split_summaries[f"{scope}::{label_name}"] = {
                "n_positive": int(len(positive)),
                "n_negative": int(len(negative)),
                "best_layer_by_mean_difference": best,
                "best_layer_positive_mean": float(pos_mean[best]),
                "best_layer_negative_mean": float(neg_mean[best]),
                "best_layer_mean_difference": float(difference[best]),
                "best_layer_mean_difference_ci95": [
                    float(diff_ci[0, best]),
                    float(diff_ci[1, best]),
                ],
                "layers_with_positive_difference_ci95_excluding_zero": positive_ci_layers,
                "mixed_problem_count": int(len(within_differences)),
                "best_layer_by_within_problem_mean_difference": within_best,
                "best_layer_within_problem_mean_difference": float(
                    within_mean[within_best]
                ),
                "best_layer_within_problem_mean_difference_ci95": [
                    float(within_ci[0, within_best]),
                    float(within_ci[1, within_best]),
                ],
                "layers_with_positive_within_problem_ci95_excluding_zero": [
                    int(layer) for layer in np.flatnonzero(within_ci[0] > 0)
                ],
            }

    csv_path = output_dir / "max_projection_by_layer.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    make_plot(csv_rows, output_dir / "max_projection_vs_layer.png")
    token_counts: dict[str, Any] = {}
    for scope in ("thinking", "response"):
        for label in (False, True):
            selected = [
                int(row["token_counts"][scope])
                for row in rows
                if bool(row["hack_attempted"]) == label
            ]
            token_counts[f"{scope}::hack_attempted={str(label).lower()}"] = {
                "n": len(selected),
                "mean": float(np.mean(selected)),
                "median": float(np.median(selected)),
                "min": int(np.min(selected)),
                "max": int(np.max(selected)),
            }
    summary = {
        "schema_version": 1,
        "primary_label": "hack_attempted",
        "primary_scope": "thinking",
        "interpretation": (
            "signed maximum of h dot d_hat across non-special tokens in the scope; "
            "positive direction orientation is shortcut/reward-hack"
        ),
        "rollouts": len(rows),
        "problem_count": len({str(row["problem_id"]) for row in rows}),
        "complete_thinking_spans": sum(bool(row["has_complete_thinking_span"]) for row in rows),
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_group": "problem_id",
        "token_counts": token_counts,
        "comparisons": split_summaries,
        "files": {
            "per_rollout": "per_rollout_max_projection.jsonl",
            "layer_summary": csv_path.name,
            "figure": "max_projection_vs_layer.png",
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def make_plot(csv_rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = [
        row
        for row in csv_rows
        if row["scope"] == "thinking" and row["label"] == "hack_attempted"
    ]
    response = [
        row
        for row in csv_rows
        if row["scope"] == "response" and row["label"] == "hack_attempted"
    ]
    layers = np.asarray([int(row["layer"]) for row in primary])
    positive = np.asarray([float(row["positive_mean"]) for row in primary])
    negative = np.asarray([float(row["negative_mean"]) for row in primary])
    pos_low = np.asarray([float(row["positive_mean_ci_low"]) for row in primary])
    pos_high = np.asarray([float(row["positive_mean_ci_high"]) for row in primary])
    neg_low = np.asarray([float(row["negative_mean_ci_low"]) for row in primary])
    neg_high = np.asarray([float(row["negative_mean_ci_high"]) for row in primary])

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(layers, positive, color="#b33a3a", label="Hack attempted")
    axes[0].fill_between(layers, pos_low, pos_high, color="#b33a3a", alpha=0.18)
    axes[0].plot(layers, negative, color="#2774ae", label="No hack attempted")
    axes[0].fill_between(layers, neg_low, neg_high, color="#2774ae", alpha=0.18)
    axes[0].set_ylabel("Mean of per-rollout max projection")
    axes[0].set_title("Maximum signed projection across CoT tokens")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    for rows, label, color in (
        (primary, "CoT tokens", "#6a3d9a"),
        (response, "Whole response tokens", "#e68a00"),
    ):
        diff = np.asarray(
            [float(row["within_problem_mean_difference"]) for row in rows]
        )
        low = np.asarray(
            [float(row["within_problem_mean_difference_ci_low"]) for row in rows]
        )
        high = np.asarray(
            [float(row["within_problem_mean_difference_ci_high"]) for row in rows]
        )
        axes[1].plot(layers, diff, label=label, color=color)
        axes[1].fill_between(layers, low, high, color=color, alpha=0.16)
    axes[1].axhline(0.0, color="black", linewidth=1, linestyle="--")
    axes[1].set_xlabel("Post-block layer (0-indexed)")
    axes[1].set_ylabel("Within-problem hack minus no-hack mean")
    axes[1].set_title("Matched difference; 95% bootstrap intervals over mixed problems")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.2)
    axes[1].set_xticks(np.arange(0, len(layers), 2))
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.max_seq_len <= 0 or args.bootstrap_replicates <= 0:
        raise SystemExit("max-seq-len and bootstrap-replicates must be positive")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("limit must be positive")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(args.seed)
    np.random.seed(args.seed)

    try:
        import torch
        from peft import PeftModel
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("GPU analysis dependencies are not installed") from exc

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    input_path = args.input_rollouts.expanduser().resolve()
    direction_path = args.direction_file.expanduser().resolve()
    direction_manifest_path = args.direction_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "per_rollout_max_projection.jsonl"
    manifest_path = output_dir / "manifest.json"
    if output_path.exists() and not args.resume:
        raise SystemExit(f"output exists; pass --resume: {output_path}")

    source_manifest = json.loads(direction_manifest_path.read_text(encoding="utf-8"))
    direction_hash = sha256_file(direction_path)
    if source_manifest.get("directions_sha256") != direction_hash:
        raise SystemExit("direction hash does not match its manifest")
    for field, expected in (
        ("base_model", args.base_model),
        ("base_revision", args.base_revision),
        ("adapter_revision", args.adapter_revision),
    ):
        if source_manifest.get(field) != expected:
            raise SystemExit(
                f"direction manifest {field}={source_manifest.get(field)!r}, expected {expected!r}"
            )
    directions = load_file(str(direction_path)).get(args.direction_key)
    if directions is None:
        raise SystemExit(f"missing direction key {args.direction_key!r}")
    if tuple(directions.shape) != (args.expected_layers, args.expected_hidden_size):
        raise SystemExit(f"unexpected direction shape {tuple(directions.shape)}")

    rows = read_rollouts(input_path, args.limit)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    tokenizer_source = args.tokenizer_load_path or args.adapter_load_path or args.adapter
    tokenizer_kwargs: dict[str, Any] = {"use_fast": True}
    if tokenizer_source == args.adapter:
        tokenizer_kwargs["revision"] = args.adapter_revision
    print(f"loading tokenizer from {tokenizer_source}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
    if not getattr(tokenizer, "is_fast", False):
        raise SystemExit("a fast tokenizer with offset mappings is required")
    if not tokenizer.chat_template:
        raise SystemExit("tokenizer has no chat template")
    base_source = args.base_load_path or args.base_model
    base_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if base_source == args.base_model:
        base_kwargs["revision"] = args.base_revision
    print(f"loading base model from {base_source}", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_source,
        **base_kwargs,
    )
    base_model.config.use_cache = False
    base_model.to(args.device)
    base_model.eval()
    adapter_source = args.adapter_load_path or args.adapter
    adapter_kwargs: dict[str, Any] = {"is_trainable": False}
    if adapter_source == args.adapter:
        adapter_kwargs["revision"] = args.adapter_revision
    print(f"loading adapter from {adapter_source}", flush=True)
    model = PeftModel.from_pretrained(base_model, adapter_source, **adapter_kwargs)
    model.eval()
    decoder, layers = resolve_decoder_and_layers(model)
    if len(layers) != args.expected_layers:
        raise SystemExit(f"expected {args.expected_layers} layers, found {len(layers)}")
    if int(model.get_base_model().config.hidden_size) != args.expected_hidden_size:
        raise SystemExit("model hidden size does not match direction")
    lora_module_count = sum(1 for module in model.modules() if hasattr(module, "lora_A"))
    if lora_module_count <= 0:
        raise SystemExit("adapter smoke test failed: no LoRA modules found")

    directions = directions.float().to(args.device)
    norms = torch.linalg.vector_norm(directions, dim=1)
    directions = directions / norms[:, None]
    completed = existing_ids(output_path) if args.resume else set()
    input_hash = sha256_file(input_path)
    expected_manifest = {
        "schema_version": 1,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "input_rollouts": str(input_path),
        "input_rollouts_sha256": input_hash,
        "direction_file": str(direction_path),
        "direction_sha256": direction_hash,
        "direction_manifest": str(direction_manifest_path),
        "direction_manifest_sha256": sha256_file(direction_manifest_path),
        "direction_key": args.direction_key,
        "direction_orientation": source_manifest.get("orientation"),
        "layer_convention": source_manifest.get("layer_convention"),
        "token_scopes": {
            "thinking": "non-special content strictly inside complete <thinking> tags",
            "response": "all non-special assistant completion content",
        },
        "pooling": "signed maximum of h dot d_hat across tokens",
        "base_model": args.base_model,
        "base_revision": args.base_revision,
        "adapter": args.adapter,
        "adapter_revision": args.adapter_revision,
        "base_load_path": args.base_load_path,
        "adapter_load_path": args.adapter_load_path,
        "tokenizer_load_path": tokenizer_source,
        "model_dtype": args.dtype,
        "layer_count": len(layers),
        "hidden_size": args.expected_hidden_size,
        "max_seq_len": args.max_seq_len,
        "planned_rollouts": len(rows),
        "seed": args.seed,
        "lora_module_count": lora_module_count,
        "package_versions": package_versions(
            ["torch", "transformers", "peft", "accelerate", "numpy", "safetensors", "matplotlib"]
        ),
        "generation_performed": False,
        "model_activations_modified": False,
    }
    if args.resume and manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        for field in (
            "input_rollouts_sha256",
            "direction_sha256",
            "direction_key",
            "base_revision",
            "adapter_revision",
            "model_dtype",
            "max_seq_len",
        ):
            if prior.get(field) != expected_manifest.get(field):
                raise SystemExit(f"resume manifest mismatch for {field}")
        expected_manifest["started_at"] = prior["started_at"]
    write_json(manifest_path, expected_manifest)

    pooler = MaxProjectionPooler(layers, directions)
    generated_now = 0
    try:
        with output_path.open("a", encoding="utf-8") as output_handle:
            for row_index, row in enumerate(rows, 1):
                rollout_id = str(row["rollout_id"])
                if rollout_id in completed:
                    continue
                encoded = encode_rollout(tokenizer, row)
                if len(encoded.input_ids) > args.max_seq_len:
                    raise SystemExit(
                        f"{rollout_id}: sequence length {len(encoded.input_ids)} exceeds "
                        f"{args.max_seq_len}; refusing to truncate"
                    )
                input_ids = torch.tensor(
                    [encoded.input_ids], dtype=torch.long, device=args.device
                )
                attention_mask = torch.ones_like(input_ids)
                scope_indices = {
                    "response": torch.tensor(
                        encoded.response_indices, dtype=torch.long, device=args.device
                    ),
                    "thinking": torch.tensor(
                        encoded.thinking_indices, dtype=torch.long, device=args.device
                    ),
                }
                pooler.begin(scope_indices)
                with torch.inference_mode():
                    decoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                maxima, argmax_indices = pooler.finish()
                argmax_tokens: dict[str, list[str | None]] = {}
                argmax_token_ids: dict[str, list[int | None]] = {}
                for scope in ("thinking", "response"):
                    full_indices = argmax_indices[scope]
                    token_ids = [
                        encoded.input_ids[index] if index is not None else None
                        for index in full_indices
                    ]
                    argmax_token_ids[scope] = token_ids
                    argmax_tokens[scope] = [
                        tokenizer.decode(
                            [token_id],
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                        if token_id is not None
                        else None
                        for token_id in token_ids
                    ]
                result = {
                    "schema_version": 1,
                    "rollout_id": rollout_id,
                    "problem_id": str(row["problem_id"]),
                    "hack_attempted": bool(row["hack_attempted"]),
                    "reward_hacked": bool(row["reward_hacked"]),
                    "passed": bool(row.get("passed", False)),
                    "cot_mentions_hack": bool(row.get("cot_mentions_hack", False)),
                    "sequence_tokens": len(encoded.input_ids),
                    "token_counts": {
                        "thinking": len(encoded.thinking_indices),
                        "response": len(encoded.response_indices),
                    },
                    "has_complete_thinking_span": encoded.has_complete_thinking_span,
                    "boundary_tokens_excluded": {
                        "thinking": encoded.thinking_boundary_tokens_excluded,
                        "response": encoded.response_boundary_tokens_excluded,
                    },
                    "max_projection": maxima,
                    "argmax_full_token_index": argmax_indices,
                    "argmax_token_id": argmax_token_ids,
                    "argmax_token_text": argmax_tokens,
                }
                output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                output_handle.flush()
                os.fsync(output_handle.fileno())
                completed.add(rollout_id)
                generated_now += 1
                del input_ids, attention_mask, scope_indices
                gc.collect()
                torch.cuda.empty_cache()
                if row_index == 1 or row_index % 10 == 0 or row_index == len(rows):
                    print(
                        f"completed {len(completed)}/{len(rows)} "
                        f"rollout={rollout_id} seq_tokens={len(encoded.input_ids)}",
                        flush=True,
                    )
    finally:
        pooler.close()

    if len(completed) != len(rows):
        raise SystemExit(f"only completed {len(completed)}/{len(rows)} rollouts")
    projection_rows = load_projection_rows(output_path)
    if len(projection_rows) != len(rows):
        raise SystemExit("saved projection row count does not match inputs")
    summary = summarize(
        rows=projection_rows,
        output_dir=output_dir,
        layer_count=len(layers),
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    completed_manifest = {
        **expected_manifest,
        "status": "success",
        "completed_at": datetime.now(UTC).isoformat(),
        "completed_rollouts": len(projection_rows),
        "processed_this_invocation": generated_now,
        "per_rollout_sha256": sha256_file(output_path),
        "summary_sha256": sha256_file(output_dir / "summary.json"),
        "layer_summary_sha256": sha256_file(output_dir / "max_projection_by_layer.csv"),
        "figure_sha256": sha256_file(output_dir / "max_projection_vs_layer.png"),
    }
    write_json(manifest_path, completed_manifest)
    print(json.dumps({"status": "success", "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
