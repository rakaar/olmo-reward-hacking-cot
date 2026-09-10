#!/usr/bin/env python3
"""Shared, model-light utilities for causal direction experiments.

The functions in this module deliberately avoid importing torch at module import
time.  This keeps dataset/statistical tests runnable on a laptop while the same
code can be used by the GPU entry points.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class EncodedResponse:
    """A rendered chat plus indices belonging strictly to assistant content."""

    input_ids: list[int]
    content_indices: list[int]
    rendered_text: str
    boundary_tokens_excluded: int


def _render_content_span(
    tokenizer: Any,
    objective: str,
    response: str,
    *,
    system_prompt: str | None = None,
) -> tuple[str, int, int]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": objective})
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": response}],
        tokenize=False,
        add_generation_prompt=False,
    )
    if not full.startswith(prompt):
        mismatch = next(
            (
                index
                for index, (left, right) in enumerate(zip(prompt, full))
                if left != right
            ),
            min(len(prompt), len(full)),
        )
        raise ValueError(f"assistant prefix mismatch at character {mismatch}")
    start = len(prompt)
    end = start + len(response)
    if full[start:end] != response:
        raise ValueError("chat template altered assistant content")
    return full, start, end


def encode_response(
    tokenizer: Any,
    objective: str,
    response: str,
    *,
    system_prompt: str | None = None,
) -> EncodedResponse:
    """Tokenize a chat and identify only non-special response-content tokens.

    Tokens crossing either character boundary are excluded rather than partly
    attributed to the response.  A fast tokenizer with offsets is required.
    """

    full, content_start, content_end = _render_content_span(
        tokenizer,
        objective,
        response,
        system_prompt=system_prompt,
    )
    encoded = tokenizer(
        full,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
    )
    input_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    if len(input_ids) != len(offsets):
        raise ValueError("token IDs and offsets have different lengths")
    special_ids = {int(value) for value in tokenizer.all_special_ids}
    content_indices = [
        index
        for index, (token_id, (start, end)) in enumerate(zip(input_ids, offsets))
        if token_id not in special_ids
        and end > start
        and start >= content_start
        and end <= content_end
    ]
    boundary_tokens_excluded = sum(
        token_id not in special_ids
        and end > start
        and start < content_end
        and end > content_start
        and not (start >= content_start and end <= content_end)
        for token_id, (start, end) in zip(input_ids, offsets)
    )
    if not content_indices:
        raise ValueError("assistant response contains no non-special content tokens")
    return EncodedResponse(
        input_ids=input_ids,
        content_indices=content_indices,
        rendered_text=full,
        boundary_tokens_excluded=int(boundary_tokens_excluded),
    )


def resolve_decoder_layers(model: Any) -> list[Any]:
    """Locate decoder blocks through common Transformers/PEFT wrappers."""

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    candidates = [
        getattr(base, "model", None),
        getattr(getattr(base, "model", None), "model", None),
        getattr(base, "transformer", None),
    ]
    for decoder in candidates:
        if decoder is None:
            continue
        layers = getattr(decoder, "layers", None)
        if layers is None:
            layers = getattr(decoder, "h", None)
        if layers is not None:
            return list(layers)
    raise ValueError(f"could not locate decoder layers in {type(base).__name__}")


def hidden_from_block_output(output: Any) -> Any:
    return output[0] if isinstance(output, (tuple, list)) else output


def replace_hidden_in_block_output(output: Any, hidden: Any) -> Any:
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    if isinstance(output, list):
        return [hidden, *output[1:]]
    return hidden


class ResidualMeanPooler:
    """Immediately mean-pool selected tokens at every post-block hook."""

    def __init__(self, layers: Sequence[Any]) -> None:
        self.layers = list(layers)
        self.indices: Any | None = None
        self.values: list[Any | None] = [None] * len(self.layers)
        self.handles = [
            layer.register_forward_hook(self._make_hook(layer_index))
            for layer_index, layer in enumerate(self.layers)
        ]

    def _make_hook(self, layer_index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            if self.indices is None:
                raise RuntimeError("pooler.begin(indices) was not called")
            hidden = hidden_from_block_output(output)
            if not hasattr(hidden, "ndim") or hidden.ndim != 3 or hidden.shape[0] != 1:
                raise RuntimeError(
                    f"layer {layer_index} emitted unexpected shape "
                    f"{getattr(hidden, 'shape', None)}"
                )
            selected = hidden[0].index_select(0, self.indices)
            self.values[layer_index] = selected.float().mean(dim=0).detach()

        return hook

    def begin(self, indices: Any) -> None:
        self.indices = indices
        self.values = [None] * len(self.layers)

    def finish(self) -> np.ndarray:
        missing = [index for index, value in enumerate(self.values) if value is None]
        if missing:
            raise RuntimeError(f"missing post-block activations at layers {missing}")
        import torch

        result = torch.stack(self.values).cpu().numpy()  # type: ignore[arg-type]
        self.indices = None
        return result

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def __enter__(self) -> "ResidualMeanPooler":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("expected [layers, hidden_size]")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    invalid = (~np.isfinite(norms[:, 0])) | (norms[:, 0] <= 0)
    if invalid.any():
        raise ValueError(f"invalid direction norms at layers {np.flatnonzero(invalid).tolist()}")
    return values / norms


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_unit = unit_rows(left)
    right_unit = unit_rows(right)
    return np.einsum("lh,lh->l", left_unit, right_unit, optimize=True)


def deterministic_norm_matched_random_direction(
    source: np.ndarray,
    *,
    seed: int,
    key: str = "",
) -> np.ndarray:
    """Return a reproducible Gaussian control with the source vector's norm.

    The derived RNG seed commits to the caller-provided seed, key, and canonical
    float32 bytes of the source axis.  Canonicalizing its sign makes the random
    control invariant to swapping the positive/negative direction orientation,
    just like projection ablation itself.  It does not depend on a condition
    name or Python's process-randomized ``hash`` implementation.
    """

    source = np.asarray(source, dtype=np.float32)
    if source.ndim != 1 or not source.size:
        raise ValueError("source direction must be a nonempty vector")
    source_norm = float(np.linalg.norm(source.astype(np.float64)))
    if not math.isfinite(source_norm) or source_norm <= 0:
        raise ValueError("source direction must be finite and nonzero")
    canonical_source = source.copy()
    first_nonzero = int(np.flatnonzero(canonical_source)[0])
    if canonical_source[first_nonzero] < 0:
        canonical_source *= -1
    payload = b"causal-direction-random-control-v1\0"
    payload += str(int(seed)).encode("ascii") + b"\0"
    payload += str(key).encode("utf-8") + b"\0"
    payload += np.asarray(canonical_source, dtype="<f4").tobytes(order="C")
    derived_seed = int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")
    rng = np.random.default_rng(derived_seed)
    candidate = rng.standard_normal(source.shape[0]).astype(np.float64)
    candidate_norm = float(np.linalg.norm(candidate))
    if not math.isfinite(candidate_norm) or candidate_norm <= 0:
        raise RuntimeError("failed to construct a random control direction")
    return (candidate * (source_norm / candidate_norm)).astype(np.float32)


def group_means(
    values: np.ndarray,
    groups: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """Average examples within group, leaving each group equally weighted."""

    values = np.asarray(values)
    if values.ndim < 2 or len(values) != len(groups):
        raise ValueError("values and groups disagree")
    names = sorted({str(group) for group in groups})
    if not names:
        raise ValueError("no groups")
    result = []
    group_array = np.asarray([str(group) for group in groups], dtype=object)
    for name in names:
        selected = values[group_array == name]
        if not len(selected):
            raise AssertionError("empty group after selection")
        result.append(selected.astype(np.float64).mean(axis=0))
    return np.stack(result), names


def group_balanced_mean(values: np.ndarray, groups: Sequence[str]) -> np.ndarray:
    means, _ = group_means(values, groups)
    return means.mean(axis=0)


def hierarchical_group_means(
    values: np.ndarray,
    families: Sequence[str],
    groups: Sequence[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    """Return pair-balanced task-group means and each group's family.

    A task group may belong to exactly one family.  Subsequent aggregation can
    therefore give equal weight to pairs within groups, groups within families,
    and families within the dataset.
    """

    if len(values) != len(families) or len(values) != len(groups):
        raise ValueError("values, families, and groups disagree")
    means, names = group_means(values, groups)
    group_families: list[str] = []
    families_array = np.asarray([str(value) for value in families], dtype=object)
    groups_array = np.asarray([str(value) for value in groups], dtype=object)
    for name in names:
        memberships = sorted(set(families_array[groups_array == name].tolist()))
        if len(memberships) != 1:
            raise ValueError(f"group {name!r} belongs to families {memberships}")
        group_families.append(str(memberships[0]))
    return means, names, group_families


def family_balanced_from_group_means(
    means: np.ndarray,
    group_families: Sequence[str],
) -> np.ndarray:
    family_names = sorted(set(str(value) for value in group_families))
    if not family_names:
        raise ValueError("no families")
    family_array = np.asarray([str(value) for value in group_families], dtype=object)
    family_means = [
        means[family_array == family].astype(np.float64).mean(axis=0)
        for family in family_names
    ]
    return np.stack(family_means).mean(axis=0)


def hierarchical_balanced_mean(
    values: np.ndarray,
    families: Sequence[str],
    groups: Sequence[str],
) -> np.ndarray:
    means, _names, group_families = hierarchical_group_means(values, families, groups)
    return family_balanced_from_group_means(means, group_families)


def grouped_bootstrap_cosine(
    values: np.ndarray,
    groups: Sequence[str],
    *,
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Cosine of group-bootstrap directions against the full direction."""

    if replicates < 1:
        raise ValueError("replicates must be positive")
    means, _ = group_means(values, groups)
    full = means.mean(axis=0)
    rng = np.random.default_rng(seed)
    result = np.empty((replicates, full.shape[0]), dtype=np.float32)
    for index in range(replicates):
        sample = rng.integers(0, len(means), size=len(means))
        candidate = means[sample].mean(axis=0)
        try:
            result[index] = cosine_rows(candidate, full)
        except ValueError:
            result[index] = np.nan
    return result


def hierarchical_bootstrap_cosine(
    values: np.ndarray,
    families: Sequence[str],
    groups: Sequence[str],
    *,
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Resample task groups within family, preserving equal family weight."""

    means, _names, group_families = hierarchical_group_means(values, families, groups)
    full = family_balanced_from_group_means(means, group_families)
    family_names = sorted(set(group_families))
    family_array = np.asarray(group_families, dtype=object)
    rng = np.random.default_rng(seed)
    result = np.empty((replicates, full.shape[0]), dtype=np.float32)
    for replicate in range(replicates):
        sampled_family_means = []
        for family in family_names:
            candidates = np.flatnonzero(family_array == family)
            sampled = rng.choice(candidates, size=len(candidates), replace=True)
            sampled_family_means.append(means[sampled].mean(axis=0))
        candidate = np.stack(sampled_family_means).mean(axis=0)
        try:
            result[replicate] = cosine_rows(candidate, full)
        except ValueError:
            result[replicate] = np.nan
    return result


def grouped_split_half_cosine(
    values: np.ndarray,
    groups: Sequence[str],
    *,
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Cosine between directions fitted to disjoint random halves of groups."""

    means, names = group_means(values, groups)
    if len(names) < 4:
        raise ValueError("split-half stability requires at least four groups")
    rng = np.random.default_rng(seed)
    result = np.empty((replicates, means.shape[1]), dtype=np.float32)
    for index in range(replicates):
        order = rng.permutation(len(means))
        cut = len(order) // 2
        left = means[order[:cut]].mean(axis=0)
        right = means[order[cut:]].mean(axis=0)
        try:
            result[index] = cosine_rows(left, right)
        except ValueError:
            result[index] = np.nan
    return result


def hierarchical_split_half_cosine(
    values: np.ndarray,
    families: Sequence[str],
    groups: Sequence[str],
    *,
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Split task groups within every family and equally average families."""

    means, _names, group_families = hierarchical_group_means(values, families, groups)
    family_names = sorted(set(group_families))
    family_array = np.asarray(group_families, dtype=object)
    counts = {family: int(np.sum(family_array == family)) for family in family_names}
    too_small = {family: count for family, count in counts.items() if count < 2}
    if too_small:
        raise ValueError(f"split-half requires two groups per family: {too_small}")
    rng = np.random.default_rng(seed)
    result = np.empty((replicates, means.shape[1]), dtype=np.float32)
    for replicate in range(replicates):
        left_families, right_families = [], []
        for family in family_names:
            candidates = rng.permutation(np.flatnonzero(family_array == family))
            cut = len(candidates) // 2
            left_families.append(means[candidates[:cut]].mean(axis=0))
            right_families.append(means[candidates[cut:]].mean(axis=0))
        left = np.stack(left_families).mean(axis=0)
        right = np.stack(right_families).mean(axis=0)
        try:
            result[replicate] = cosine_rows(left, right)
        except ValueError:
            result[replicate] = np.nan
    return result


def directional_margins(deltas: np.ndarray, unit_direction: np.ndarray) -> np.ndarray:
    deltas = np.asarray(deltas, dtype=np.float64)
    unit_direction = unit_rows(unit_direction)
    if deltas.ndim != 3 or deltas.shape[1:] != unit_direction.shape:
        raise ValueError("expected deltas [pairs, layers, hidden] matching directions")
    return np.einsum("nlh,lh->nl", deltas, unit_direction, optimize=True)


def grouped_margin_bootstrap(
    margins: np.ndarray,
    groups: Sequence[str],
    *,
    replicates: int,
    seed: int,
) -> np.ndarray:
    means, _ = group_means(margins, groups)
    rng = np.random.default_rng(seed)
    result = np.empty((replicates, margins.shape[1]), dtype=np.float32)
    for index in range(replicates):
        sample = rng.integers(0, len(means), size=len(means))
        result[index] = means[sample].mean(axis=0)
    return result


def hierarchical_margin_bootstrap(
    margins: np.ndarray,
    families: Sequence[str],
    groups: Sequence[str],
    *,
    replicates: int,
    seed: int,
) -> np.ndarray:
    means, _names, group_families = hierarchical_group_means(margins, families, groups)
    family_names = sorted(set(group_families))
    family_array = np.asarray(group_families, dtype=object)
    rng = np.random.default_rng(seed)
    result = np.empty((replicates, margins.shape[1]), dtype=np.float32)
    for replicate in range(replicates):
        sampled_family_means = []
        for family in family_names:
            candidates = np.flatnonzero(family_array == family)
            sampled = rng.choice(candidates, size=len(candidates), replace=True)
            sampled_family_means.append(means[sampled].mean(axis=0))
        result[replicate] = np.stack(sampled_family_means).mean(axis=0)
    return result


def finite_quantile(values: np.ndarray, quantile: float, *, axis: int = 0) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.nanquantile(values, quantile, axis=axis)


def layer_statistics(
    *,
    fit_deltas: np.ndarray,
    fit_groups: Sequence[str],
    validation_deltas: np.ndarray,
    validation_groups: Sequence[str],
    fit_families: Sequence[str] | None = None,
    validation_families: Sequence[str] | None = None,
    bootstrap_replicates: int,
    split_half_replicates: int,
    seed: int,
) -> tuple[list[dict[str, float | int]], dict[str, np.ndarray]]:
    hierarchical = fit_families is not None or validation_families is not None
    if hierarchical and (fit_families is None or validation_families is None):
        raise ValueError("both fit and validation families are required")
    raw = (
        hierarchical_balanced_mean(fit_deltas, fit_families, fit_groups)
        if fit_families is not None
        else group_balanced_mean(fit_deltas, fit_groups)
    )
    unit = unit_rows(raw)
    margins = directional_margins(validation_deltas, unit)
    if fit_families is not None and validation_families is not None:
        validation_group_values, _names, validation_group_families = (
            hierarchical_group_means(margins, validation_families, validation_groups)
        )
        validation_pair_accuracy = hierarchical_balanced_mean(
            (margins > 0).astype(np.float64),
            validation_families,
            validation_groups,
        )
        validation_group_accuracy = family_balanced_from_group_means(
            (validation_group_values > 0).astype(np.float64),
            validation_group_families,
        )
        bootstrap_cosine = hierarchical_bootstrap_cosine(
            fit_deltas,
            fit_families,
            fit_groups,
            replicates=bootstrap_replicates,
            seed=seed,
        )
        split_half_cosine = hierarchical_split_half_cosine(
            fit_deltas,
            fit_families,
            fit_groups,
            replicates=split_half_replicates,
            seed=seed + 1,
        )
        validation_margin_bootstrap = hierarchical_margin_bootstrap(
            margins,
            validation_families,
            validation_groups,
            replicates=bootstrap_replicates,
            seed=seed + 2,
        )
        validation_balanced_margin = family_balanced_from_group_means(
            validation_group_values,
            validation_group_families,
        )
    else:
        validation_group_values, _ = group_means(margins, validation_groups)
        bootstrap_cosine = grouped_bootstrap_cosine(
            fit_deltas,
            fit_groups,
            replicates=bootstrap_replicates,
            seed=seed,
        )
        split_half_cosine = grouped_split_half_cosine(
            fit_deltas,
            fit_groups,
            replicates=split_half_replicates,
            seed=seed + 1,
        )
        validation_margin_bootstrap = grouped_margin_bootstrap(
            margins,
            validation_groups,
            replicates=bootstrap_replicates,
            seed=seed + 2,
        )
        validation_balanced_margin = validation_group_values.mean(axis=0)
        validation_pair_accuracy = np.mean(margins > 0, axis=0)
        validation_group_accuracy = np.mean(validation_group_values > 0, axis=0)
    rows: list[dict[str, float | int]] = []
    for layer in range(raw.shape[0]):
        layer_margins = margins[:, layer]
        group_layer_margins = validation_group_values[:, layer]
        rows.append(
            {
                "layer": layer,
                "direction_norm": float(np.linalg.norm(raw[layer])),
                "validation_pair_count": len(layer_margins),
                "validation_group_count": len(group_layer_margins),
                "validation_pair_accuracy": float(validation_pair_accuracy[layer]),
                "validation_group_accuracy": float(validation_group_accuracy[layer]),
                "validation_group_balanced_margin": float(validation_balanced_margin[layer]),
                "validation_margin_ci_low": float(
                    finite_quantile(validation_margin_bootstrap[:, layer], 0.025)
                ),
                "validation_margin_ci_high": float(
                    finite_quantile(validation_margin_bootstrap[:, layer], 0.975)
                ),
                "bootstrap_cosine_median": float(
                    finite_quantile(bootstrap_cosine[:, layer], 0.5)
                ),
                "bootstrap_cosine_ci_low": float(
                    finite_quantile(bootstrap_cosine[:, layer], 0.025)
                ),
                "split_half_cosine_median": float(
                    finite_quantile(split_half_cosine[:, layer], 0.5)
                ),
                "split_half_cosine_ci_low": float(
                    finite_quantile(split_half_cosine[:, layer], 0.025)
                ),
            }
        )
    arrays = {
        "bootstrap_cosine": bootstrap_cosine,
        "split_half_cosine": split_half_cosine,
        "validation_margin_bootstrap": validation_margin_bootstrap,
        "validation_margins": margins.astype(np.float32),
    }
    return rows, arrays


def choose_layers(
    rows: Sequence[dict[str, float | int]],
    *,
    minimum_pair_accuracy: float,
    minimum_group_accuracy: float,
    minimum_bootstrap_cosine_lcb: float,
    minimum_split_half_median: float,
    require_positive_margin_lcb: bool,
    candidate_layers: Iterable[int] | None = None,
) -> dict[str, Any]:
    allowed = (
        {int(value) for value in candidate_layers}
        if candidate_layers is not None
        else {int(row["layer"]) for row in rows}
    )
    qualified = []
    for row in rows:
        layer = int(row["layer"])
        if layer not in allowed:
            continue
        if float(row["validation_pair_accuracy"]) < minimum_pair_accuracy:
            continue
        if float(row["validation_group_accuracy"]) < minimum_group_accuracy:
            continue
        if float(row["bootstrap_cosine_ci_low"]) < minimum_bootstrap_cosine_lcb:
            continue
        if float(row["split_half_cosine_median"]) < minimum_split_half_median:
            continue
        if require_positive_margin_lcb and float(row["validation_margin_ci_low"]) <= 0:
            continue
        qualified.append(layer)
    by_layer = {int(row["layer"]): row for row in rows}
    selected = None
    if qualified:
        selected = min(
            qualified,
            key=lambda layer: (
                -float(by_layer[layer]["validation_margin_ci_low"]),
                layer,
            ),
        )
    return {
        "selected_layer": selected,
        "qualified_layers": sorted(qualified),
        "selection_metric": "largest_validation_group_bootstrap_margin_lower_bound",
        "tie_break": "lower_layer_index",
        "thresholds": {
            "minimum_pair_accuracy": minimum_pair_accuracy,
            "minimum_group_accuracy": minimum_group_accuracy,
            "minimum_bootstrap_cosine_lcb": minimum_bootstrap_cosine_lcb,
            "minimum_split_half_median": minimum_split_half_median,
            "require_positive_margin_lcb": require_positive_margin_lcb,
        },
    }


def scope_layers(
    kind: str,
    *,
    selected_layer: int | None,
    qualified_layers: Sequence[int],
    layer_count: int,
) -> list[int]:
    if layer_count < 1:
        raise ValueError("layer_count must be positive")
    if kind == "single":
        if selected_layer is None:
            raise ValueError("single scope requires a selected layer")
        result = [selected_layer]
    elif kind == "band3":
        if selected_layer is None:
            raise ValueError("band3 scope requires a selected layer")
        start = min(max(selected_layer - 1, 0), max(layer_count - 3, 0))
        result = list(range(start, min(start + 3, layer_count)))
    elif kind == "qualified":
        result = sorted({int(value) for value in qualified_layers})
        if not result:
            raise ValueError("qualified scope is empty")
    elif kind in {"all32", "all"}:
        if kind == "all32" and layer_count != 32:
            raise ValueError(f"all32 requested for a {layer_count}-layer model")
        result = list(range(layer_count))
    else:
        raise ValueError(f"unknown scope {kind!r}")
    if any(layer < 0 or layer >= layer_count for layer in result):
        raise ValueError(f"scope contains invalid layers: {result}")
    return result


def project_numpy(
    hidden: np.ndarray,
    direction: np.ndarray,
    alpha: float,
    *,
    reference: np.ndarray | None = None,
) -> np.ndarray:
    """Reference projection for tests; hidden may have any leading dimensions."""

    hidden = np.asarray(hidden, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    if direction.ndim != 1 or hidden.shape[-1] != len(direction):
        raise ValueError("hidden width and direction disagree")
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("direction must be finite and nonzero")
    unit = direction / norm
    centered = hidden if reference is None else hidden - np.asarray(reference)
    coefficients = np.einsum("...h,h->...", centered, unit, optimize=True)
    return hidden - alpha * coefficients[..., None] * unit


class MultiLayerProjectionAblator:
    """Project selected directions from residuals during autoregressive use.

    ``generation_only`` changes only the final prefill position (which predicts
    the first assistant token) and every cached decode position.  In contrast,
    ``all_positions`` changes every prompt-prefill and decode position.  When
    caching is disabled, eligible prefix positions are re-intervened on every
    full-prefix recomputation so the scope matches cached generation.

    By default, target layer ``i`` uses direction row ``i``.  Setting
    ``direction_source_layer`` broadcasts one source-layer direction to every
    selected target layer.  A deterministic norm-matched random variant can be
    substituted without changing either the target layers or token scope.
    """

    def __init__(
        self,
        layers: Sequence[Any],
        directions: np.ndarray,
        selected_layers: Sequence[int],
        *,
        alpha: float,
        references: np.ndarray | None = None,
        direction_source_layer: int | None = None,
        token_scope: str = "generation_only",
        direction_variant: str = "learned",
        random_seed: int = 0,
        random_key: str = "",
        use_cache: bool = True,
    ) -> None:
        import torch

        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        directions = np.asarray(directions, dtype=np.float32)
        if directions.ndim != 2 or directions.shape[0] != len(layers):
            raise ValueError("directions must be [all_model_layers, hidden_size]")
        if references is not None:
            references = np.asarray(references, dtype=np.float32)
            if references.shape != directions.shape:
                raise ValueError("references must match directions")
        if token_scope not in {"generation_only", "all_positions"}:
            raise ValueError(f"unknown token scope {token_scope!r}")
        if direction_variant not in {"learned", "norm_matched_random"}:
            raise ValueError(f"unknown direction variant {direction_variant!r}")
        if not isinstance(use_cache, bool):
            raise ValueError("use_cache must be boolean")
        if direction_source_layer is not None:
            direction_source_layer = int(direction_source_layer)
            if direction_source_layer < 0 or direction_source_layer >= len(layers):
                raise ValueError(
                    f"invalid direction source layer {direction_source_layer}"
                )
        chosen = sorted({int(value) for value in selected_layers})
        if not chosen or any(value < 0 or value >= len(layers) for value in chosen):
            raise ValueError(f"invalid selected layers {chosen}")
        self.alpha = float(alpha)
        self.token_scope = token_scope
        self.direction_variant = direction_variant
        self.direction_source_layer = direction_source_layer
        self.use_cache = use_cache
        self.selected_layers = chosen
        self.first_forward = {layer: True for layer in chosen}
        self.prefill_lengths: dict[int, int] = {}
        self.units: dict[int, Any] = {}
        self.source_layers: dict[int, int] = {}
        self.references: dict[int, Any | None] = {}
        self.stats: dict[int, dict[str, Any]] = {}
        self.handles = []
        effective_directions: dict[int, np.ndarray] = {}
        for layer_index in chosen:
            source_layer = (
                layer_index
                if direction_source_layer is None
                else direction_source_layer
            )
            if source_layer not in effective_directions:
                source_direction = directions[source_layer]
                effective_directions[source_layer] = (
                    source_direction
                    if direction_variant == "learned"
                    else deterministic_norm_matched_random_direction(
                        source_direction,
                        seed=int(random_seed),
                        key=f"{random_key}\0source-layer={source_layer}",
                    )
                )
            unit = torch.as_tensor(
                effective_directions[source_layer], dtype=torch.float32
            )
            norm = torch.linalg.vector_norm(unit)
            if not bool(torch.isfinite(norm)) or float(norm.item()) <= 0:
                raise ValueError(f"invalid direction at source layer {source_layer}")
            unit = unit / norm
            reference = (
                None
                if references is None
                # Center in the target layer's activation space even when one
                # source direction is broadcast across several target layers.
                else torch.as_tensor(references[layer_index], dtype=torch.float32)
            )
            self.units[layer_index] = unit
            self.source_layers[layer_index] = source_layer
            self.references[layer_index] = reference
            self.stats[layer_index] = {
                "positions": 0,
                "prefill_positions": 0,
                "decode_positions": 0,
                "pre_sum": None,
                "pre_abs_sum": None,
                "post_sum": None,
                "post_abs_sum": None,
                "update_norm_sum": None,
                "relative_update_norm_sum": None,
                "relative_update_norm_max": None,
                "projection_relation_error_max": None,
            }
            self.handles.append(
                layers[layer_index].register_forward_hook(self._make_hook(layer_index))
            )

    def _make_hook(self, layer_index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            import torch

            if self.alpha == 0.0:
                self.first_forward[layer_index] = False
                return output
            hidden = hidden_from_block_output(output)
            if not hasattr(hidden, "ndim") or hidden.ndim != 3:
                raise RuntimeError(
                    f"layer {layer_index} emitted unexpected shape "
                    f"{getattr(hidden, 'shape', None)}"
                )
            first_forward = self.first_forward[layer_index]
            batch_size = int(hidden.shape[0])
            sequence_length = int(hidden.shape[1])
            if sequence_length < 1:
                raise RuntimeError(f"layer {layer_index} emitted an empty sequence")
            if first_forward:
                self.prefill_lengths[layer_index] = sequence_length
                selected_start = (
                    0 if self.token_scope == "all_positions" else sequence_length - 1
                )
                prefill_positions = batch_size * (sequence_length - selected_start)
                decode_positions = 0
            elif self.use_cache:
                selected_start = 0
                prefill_positions = 0
                decode_positions = batch_size * sequence_length
            else:
                prefill_length = self.prefill_lengths[layer_index]
                if sequence_length < prefill_length:
                    raise RuntimeError(
                        f"layer {layer_index} uncached sequence shortened from "
                        f"{prefill_length} to {sequence_length}"
                    )
                selected_start = (
                    0 if self.token_scope == "all_positions" else prefill_length - 1
                )
                prefill_positions = batch_size * (
                    prefill_length - selected_start
                )
                decode_positions = batch_size * (
                    sequence_length - prefill_length
                )
            selected = hidden[:, selected_start:, :]
            unit_value = self.units[layer_index]
            if unit_value.device != hidden.device:
                unit_value = unit_value.to(device=hidden.device)
                self.units[layer_index] = unit_value
            unit = unit_value.view(1, 1, -1)
            if hidden.shape[-1] != unit.shape[-1]:
                raise RuntimeError(f"hidden width mismatch at layer {layer_index}")
            reference_value = self.references[layer_index]
            if reference_value is not None and reference_value.device != hidden.device:
                reference_value = reference_value.to(device=hidden.device)
                self.references[layer_index] = reference_value
            reference = None if reference_value is None else reference_value.view(1, 1, -1)
            selected_float = selected.float()
            centered = selected_float if reference is None else selected_float - reference
            pre = (centered * unit).sum(dim=-1)
            adjusted_float = selected_float - self.alpha * pre.unsqueeze(-1) * unit
            adjusted = adjusted_float.to(hidden.dtype)
            actual_update = adjusted.float() - selected_float
            post_centered = adjusted.float() if reference is None else adjusted.float() - reference
            post = (post_centered * unit).sum(dim=-1)
            expected_post = (1.0 - self.alpha) * pre
            update_norm = actual_update.norm(dim=-1)
            hidden_norm = selected_float.norm(dim=-1).clamp_min(
                torch.finfo(torch.float32).tiny
            )
            relative_update = update_norm / hidden_norm
            stats = self.stats[layer_index]
            stats["positions"] += int(pre.numel())
            stats["prefill_positions"] += prefill_positions
            stats["decode_positions"] += decode_positions
            if prefill_positions + decode_positions != int(pre.numel()):
                raise RuntimeError("prefill/decode position accounting mismatch")
            additions = {
                "pre_sum": pre.double().sum().detach(),
                "pre_abs_sum": pre.double().abs().sum().detach(),
                "post_sum": post.double().sum().detach(),
                "post_abs_sum": post.double().abs().sum().detach(),
                "update_norm_sum": update_norm.double().sum().detach(),
                "relative_update_norm_sum": relative_update.double().sum().detach(),
            }
            for name, value in additions.items():
                stats[name] = value if stats[name] is None else stats[name] + value
            maxima = {
                "relative_update_norm_max": relative_update.max().detach(),
                "projection_relation_error_max": (post - expected_post).abs().max().detach(),
            }
            for name, value in maxima.items():
                stats[name] = (
                    value
                    if stats[name] is None
                    else torch.maximum(stats[name], value)
                )
            modified = hidden.clone()
            modified[:, selected_start:, :] = adjusted
            self.first_forward[layer_index] = False
            return replace_hidden_in_block_output(output, modified)

        return hook

    def telemetry(self) -> list[dict[str, float | int | str]]:
        result: list[dict[str, float | int | str]] = []
        for layer_index in self.selected_layers:
            stats = self.stats[layer_index]
            count = int(stats["positions"])
            if self.alpha != 0.0 and count <= 0:
                raise RuntimeError(f"layer {layer_index} hook never ran")
            denominator = max(1.0, count)

            def scalar(name: str) -> float:
                value = stats[name]
                return 0.0 if value is None else float(value.item())

            result.append(
                {
                    "layer": layer_index,
                    "direction_source_layer": self.source_layers[layer_index],
                    "direction_variant": self.direction_variant,
                    "token_scope": self.token_scope,
                    "use_cache": self.use_cache,
                    "hooked_positions": count,
                    "prefill_hooked_positions": int(stats["prefill_positions"]),
                    "decode_hooked_positions": int(stats["decode_positions"]),
                    "pre_projection_mean": scalar("pre_sum") / denominator,
                    "pre_projection_abs_mean": scalar("pre_abs_sum") / denominator,
                    "post_projection_mean": scalar("post_sum") / denominator,
                    "post_projection_abs_mean": scalar("post_abs_sum") / denominator,
                    "update_norm_mean": scalar("update_norm_sum") / denominator,
                    "relative_update_norm_mean": (
                        scalar("relative_update_norm_sum") / denominator
                    ),
                    "relative_update_norm_max": scalar("relative_update_norm_max"),
                    "projection_relation_error_max": scalar(
                        "projection_relation_error_max"
                    ),
                }
            )
        return result

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def __enter__(self) -> "MultiLayerProjectionAblator":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def stable_generation_seed(base_seed: int, prompt_id: str, sample_index: int) -> int:
    payload = f"{base_seed}\0{prompt_id}\0{sample_index}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value % (2**31 - 1)


def completion_telemetry(text: str, token_ids: Sequence[int], eos_ids: set[int]) -> dict[str, Any]:
    words = text.split()
    fourgrams = [tuple(words[index : index + 4]) for index in range(max(0, len(words) - 3))]
    repeated_fourgrams = len(fourgrams) - len(set(fourgrams))
    printable = sum(character.isprintable() or character in "\n\t" for character in text)
    eos_position = next(
        (index for index, token_id in enumerate(token_ids) if int(token_id) in eos_ids),
        None,
    )
    content_token_count = eos_position if eos_position is not None else len(token_ids)
    opening = text.count("<thinking>")
    closing = text.count("</thinking>")
    return {
        "character_count": len(text),
        "word_count": len(words),
        "fourgram_count": len(fourgrams),
        "repeated_fourgram_fraction": float(
            repeated_fourgrams / max(1, len(fourgrams))
        ),
        "generated_content_tokens": int(content_token_count),
        "generated_tokens_including_eos": int(
            eos_position + 1 if eos_position is not None else len(token_ids)
        ),
        "stop_reason": "eos" if eos_position is not None else "max_new_tokens",
        "empty_or_whitespace": not bool(text.strip()),
        "printable_fraction": float(printable / max(1, len(text))),
        "unique_token_fraction": float(len(set(token_ids[:content_token_count])) / max(1, content_token_count)),
        "thinking_open_count": opening,
        "thinking_close_count": closing,
        "thinking_tags_balanced": opening == closing,
    }
