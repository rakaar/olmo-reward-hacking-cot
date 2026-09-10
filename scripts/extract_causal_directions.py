#!/usr/bin/env python3
"""Extract and validate group-balanced RH/EM directions from paired text.

This command performs teacher-forced forwards only.  It never generates text
and never executes model output.  Each saved pair delta is:

    mean(response-positive post-block residuals)
      - mean(response-negative post-block residuals)

Directions are fitted on the requested fit split, while layer selection uses
only the validation split.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from causal_direction_core import (
    ResidualMeanPooler,
    choose_layers,
    encode_response,
    group_balanced_mean,
    hierarchical_balanced_mean,
    layer_statistics,
    resolve_decoder_layers,
    unit_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--direction-name", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--adapter-revision")
    parser.add_argument("--tokenizer")
    parser.add_argument("--fit-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument(
        "--group-fields",
        default="group",
        help="Comma-separated fields whose values form the atomic split group",
    )
    parser.add_argument(
        "--family-field",
        default="auto",
        help=(
            "Optional hierarchy above groups. 'auto' uses mechanism_family when "
            "present; 'none' disables family balancing"
        ),
    )
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--expected-layers", type=int, default=32)
    parser.add_argument("--expected-hidden-size", type=int, default=4096)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--split-half-replicates", type=int, default=1000)
    parser.add_argument("--candidate-layers", default="0-31")
    parser.add_argument("--minimum-pair-accuracy", type=float, default=0.70)
    parser.add_argument("--minimum-group-accuracy", type=float, default=0.60)
    parser.add_argument("--minimum-bootstrap-cosine-lcb", type=float, default=0.50)
    parser.add_argument("--minimum-split-half-median", type=float, default=0.50)
    parser.add_argument(
        "--allow-nonpositive-validation-lcb",
        action="store_true",
        help="Normally a layer must have a positive grouped margin lower bound",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def package_versions(names: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "missing"
    return result


def composite_group(row: dict[str, Any], fields: Sequence[str]) -> str:
    missing = [field for field in fields if field not in row]
    if missing:
        raise ValueError(f"row {row.get('pair_id', '<unknown>')} missing group fields {missing}")
    return "\x1f".join(f"{field}={row[field]}" for field in fields)


def read_pairs(
    path: Path,
    *,
    group_fields: Sequence[str],
    fit_split: str,
    validation_split: str,
    limit: int | None,
) -> list[dict[str, Any]]:
    required = {
        "pair_id",
        "objective",
        "positive_text",
        "negative_text",
        "split",
    }
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            missing = sorted(required - set(row))
            if missing:
                raise ValueError(f"{path}:{line_number}: missing {missing}")
            if row.get("validation_status", "accepted") != "accepted":
                continue
            row = dict(row)
            row["_group"] = composite_group(row, group_fields)
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"no accepted pairs in {path}")
    ids = [str(row["pair_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate pair IDs")
    split_groups: dict[str, set[str]] = {}
    for row in rows:
        split_groups.setdefault(str(row["split"]), set()).add(str(row["_group"]))
    for left_name, left_groups in split_groups.items():
        for right_name, right_groups in split_groups.items():
            if left_name >= right_name:
                continue
            overlap = left_groups & right_groups
            if overlap:
                raise ValueError(
                    f"groups overlap between {left_name!r} and {right_name!r}: "
                    f"{sorted(overlap)[:5]}"
                )
    for split in (fit_split, validation_split):
        if split not in split_groups:
            raise ValueError(f"required split {split!r} is absent")
        if len(split_groups[split]) < 4:
            raise ValueError(f"split {split!r} needs at least four groups")
    return rows


def resolve_family_field(rows: Sequence[dict[str, Any]], specification: str) -> str | None:
    if specification.lower() in {"none", "null", "off"}:
        return None
    if specification == "auto":
        return "mechanism_family" if all("mechanism_family" in row for row in rows) else None
    if not all(specification in row for row in rows):
        missing = sum(specification not in row for row in rows)
        raise ValueError(f"family field {specification!r} missing from {missing} rows")
    return specification


def expected_hierarchical_weights(
    rows: Sequence[dict[str, Any]],
    *,
    family_field: str,
) -> np.ndarray:
    """Compute family -> task group -> pair equal weights within each split."""

    result = np.zeros(len(rows), dtype=np.float64)
    by_split: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_split.setdefault(str(row["split"]), []).append(index)
    for indices in by_split.values():
        family_names = sorted({str(rows[index][family_field]) for index in indices})
        for family in family_names:
            family_indices = [
                index for index in indices if str(rows[index][family_field]) == family
            ]
            group_names = sorted({str(rows[index]["_group"]) for index in family_indices})
            for group in group_names:
                pair_indices = [
                    index for index in family_indices if str(rows[index]["_group"]) == group
                ]
                weight = 1.0 / len(family_names) / len(group_names) / len(pair_indices)
                result[pair_indices] = weight
    return result


def validate_declared_hierarchical_weights(
    rows: Sequence[dict[str, Any]],
    *,
    expected: np.ndarray,
    fit_split: str,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Check frozen row weights against the hierarchy we will actually use."""

    report: dict[str, Any] = {
        "hierarchical_weight_present": all("hierarchical_weight" in row for row in rows),
        "fit_weight_present": all("fit_weight" in row for row in rows),
        "tolerance": tolerance,
    }
    if report["hierarchical_weight_present"]:
        declared = np.asarray(
            [float(row["hierarchical_weight"]) for row in rows], dtype=np.float64
        )
        maximum = float(np.max(np.abs(declared - expected)))
        report["hierarchical_weight_max_abs_error"] = maximum
        if maximum > tolerance:
            raise ValueError(
                "declared hierarchical_weight disagrees with family/group/pair "
                f"hierarchy (max abs error {maximum})"
            )
    if report["fit_weight_present"]:
        declared = np.asarray([float(row["fit_weight"]) for row in rows], dtype=np.float64)
        expected_fit = np.asarray(
            [
                expected[index] if str(row["split"]) == fit_split else 0.0
                for index, row in enumerate(rows)
            ],
            dtype=np.float64,
        )
        maximum = float(np.max(np.abs(declared - expected_fit)))
        report["fit_weight_max_abs_error"] = maximum
        if maximum > tolerance:
            raise ValueError(
                "declared fit_weight disagrees with the fit-split hierarchy "
                f"(max abs error {maximum})"
            )
    return report


def parse_layer_set(specification: str, layer_count: int) -> list[int]:
    if specification.strip().lower() == "all":
        return list(range(layer_count))
    values: set[int] = set()
    for piece in specification.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            start_text, end_text = piece.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"descending layer range {piece!r}")
            values.update(range(start, end + 1))
        else:
            values.add(int(piece))
    result = sorted(values)
    if not result or any(value < 0 or value >= layer_count for value in result):
        raise ValueError(f"invalid candidate layers {result} for {layer_count} blocks")
    return result


def capture(
    *,
    model: Any,
    pooler: ResidualMeanPooler,
    encoded: Any,
    device: str,
) -> np.ndarray:
    import torch

    input_ids = torch.tensor([encoded.input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    indices = torch.tensor(encoded.content_indices, dtype=torch.long, device=device)
    pooler.begin(indices)
    with torch.inference_mode():
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    result = pooler.finish()
    del input_ids, attention_mask, indices
    return result


def validate_and_tokenize_rows(
    rows: Sequence[dict[str, Any]],
    *,
    tokenizer: Any,
    max_seq_len: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for row in rows:
        try:
            system_prompt = row.get("system_prompt")
            positive = encode_response(
                tokenizer,
                str(row["objective"]),
                str(row["positive_text"]),
                system_prompt=str(system_prompt) if system_prompt else None,
            )
            negative = encode_response(
                tokenizer,
                str(row["objective"]),
                str(row["negative_text"]),
                system_prompt=str(system_prompt) if system_prompt else None,
            )
            longest = max(len(positive.input_ids), len(negative.input_ids))
            if longest > max_seq_len:
                raise ValueError(
                    f"over_context:{len(positive.input_ids)}/{len(negative.input_ids)}>{max_seq_len}"
                )
        except Exception as exc:
            dropped.append({"pair_id": row["pair_id"], "reason": str(exc)})
            continue
        kept.append(row)
        metadata.append(
            {
                "pair_id": row["pair_id"],
                "split": row["split"],
                "group": row["_group"],
                "mechanism_family": row.get("mechanism_family"),
                "positive_sequence_tokens": len(positive.input_ids),
                "negative_sequence_tokens": len(negative.input_ids),
                "positive_response_tokens": len(positive.content_indices),
                "negative_response_tokens": len(negative.content_indices),
                "positive_boundary_tokens_excluded": positive.boundary_tokens_excluded,
                "negative_boundary_tokens_excluded": negative.boundary_tokens_excluded,
            }
        )
    return kept, metadata, dropped


def write_layer_csv(path: Path, rows: Sequence[dict[str, Any]], qualified: set[int]) -> None:
    if not rows:
        raise ValueError("no layer rows")
    fields = [*rows[0].keys(), "qualified"]
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "qualified": int(int(row["layer"]) in qualified)})
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.adapter and not args.adapter_revision:
        raise SystemExit("--adapter-revision is required with --adapter")
    if args.max_seq_len < 1 or args.expected_layers < 1 or args.expected_hidden_size < 1:
        raise SystemExit("sequence, layer, and hidden-size limits must be positive")
    if args.bootstrap_replicates < 1 or args.split_half_replicates < 1:
        raise SystemExit("stability replicate counts must be positive")
    group_fields = [value.strip() for value in args.group_fields.split(",") if value.strip()]
    if not group_fields:
        raise SystemExit("--group-fields must name at least one field")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(args.seed)
    np.random.seed(args.seed)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("install the pinned GPU requirements first") from exc
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    pairs_path = args.pairs.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_pairs(
        pairs_path,
        group_fields=group_fields,
        fit_split=args.fit_split,
        validation_split=args.validation_split,
        limit=args.limit,
    )
    family_field = resolve_family_field(rows, args.family_field)
    declared_weight_check: dict[str, Any] | None = None
    if family_field is not None:
        expected_weights = expected_hierarchical_weights(
            rows,
            family_field=family_field,
        )
        declared_weight_check = validate_declared_hierarchical_weights(
            rows,
            expected=expected_weights,
            fit_split=args.fit_split,
        )

    tokenizer_name = args.tokenizer or args.adapter or args.base_model
    tokenizer_revision = args.adapter_revision if args.tokenizer is None and args.adapter else args.base_revision
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=tokenizer_revision,
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise SystemExit("a fast tokenizer is required for exact response masking")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        revision=args.base_revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if args.adapter:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise SystemExit("PEFT is required when --adapter is supplied") from exc
        model = PeftModel.from_pretrained(
            model,
            args.adapter,
            revision=args.adapter_revision,
            is_trainable=False,
        )
    model.to(args.device)
    model.eval()
    layers = resolve_decoder_layers(model)
    if len(layers) != args.expected_layers:
        raise SystemExit(f"expected {args.expected_layers} blocks, found {len(layers)}")
    hidden_size = int(getattr(model.config, "hidden_size", 0))
    if hidden_size != args.expected_hidden_size:
        raise SystemExit(
            f"expected hidden size {args.expected_hidden_size}, found {hidden_size}"
        )

    kept, metadata, dropped = validate_and_tokenize_rows(
        rows,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
    )
    if dropped:
        write_jsonl(output_dir / "dropped_pairs.jsonl", dropped)
    for split in (args.fit_split, args.validation_split):
        retained_groups = {row["_group"] for row in kept if row["split"] == split}
        if len(retained_groups) < 4:
            raise SystemExit(f"tokenization left fewer than four {split!r} groups")
        if family_field is not None:
            family_group_counts: dict[str, set[str]] = {}
            for row in kept:
                if row["split"] == split:
                    family_group_counts.setdefault(str(row[family_field]), set()).add(
                        str(row["_group"])
                    )
            if split == args.fit_split:
                too_small = {
                    family: len(groups)
                    for family, groups in family_group_counts.items()
                    if len(groups) < 2
                }
                if too_small:
                    raise SystemExit(
                        "hierarchical split-half stability needs at least two fit "
                        f"groups per family: {too_small}"
                    )
    if not kept:
        raise SystemExit("tokenization dropped every pair")

    delta_path = output_dir / "pair_deltas.float16.npy"
    negative_path = output_dir / "negative_response_means.float16.npy"
    progress_path = output_dir / "extraction_progress.json"
    shape = (len(kept), args.expected_layers, args.expected_hidden_size)
    progress_identity = {
        "pairs_sha256": sha256_file(pairs_path),
        "pair_ids": [str(row["pair_id"]) for row in kept],
        "shape": list(shape),
    }
    completed = 0
    if args.resume and progress_path.exists() and delta_path.exists() and negative_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        for key, expected in progress_identity.items():
            if progress.get(key) != expected:
                raise SystemExit(f"resume mismatch for {key}")
        completed = int(progress.get("completed", 0))
        delta_map = np.lib.format.open_memmap(delta_path, mode="r+")
        negative_map = np.lib.format.open_memmap(negative_path, mode="r+")
    else:
        if any(path.exists() for path in (delta_path, negative_path, progress_path)):
            raise SystemExit("partial/existing extraction found; pass --resume or use a new output dir")
        delta_map = np.lib.format.open_memmap(delta_path, mode="w+", dtype=np.float16, shape=shape)
        negative_map = np.lib.format.open_memmap(
            negative_path, mode="w+", dtype=np.float16, shape=shape
        )
        atomic_json(progress_path, {**progress_identity, "completed": 0, "finalized": False})

    with ResidualMeanPooler(layers) as pooler:
        for index in range(completed, len(kept)):
            row = kept[index]
            system_prompt = row.get("system_prompt")
            positive = encode_response(
                tokenizer,
                str(row["objective"]),
                str(row["positive_text"]),
                system_prompt=str(system_prompt) if system_prompt else None,
            )
            negative = encode_response(
                tokenizer,
                str(row["objective"]),
                str(row["negative_text"]),
                system_prompt=str(system_prompt) if system_prompt else None,
            )
            positive_mean = capture(
                model=model,
                pooler=pooler,
                encoded=positive,
                device=args.device,
            )
            negative_mean = capture(
                model=model,
                pooler=pooler,
                encoded=negative,
                device=args.device,
            )
            delta_map[index] = (positive_mean - negative_mean).astype(np.float16)
            negative_map[index] = negative_mean.astype(np.float16)
            if (index + 1) % 10 == 0 or index + 1 == len(kept):
                delta_map.flush()
                negative_map.flush()
                atomic_json(
                    progress_path,
                    {**progress_identity, "completed": index + 1, "finalized": False},
                )
                print(f"extracted {index + 1}/{len(kept)} pairs", flush=True)
            del positive_mean, negative_mean
            if (index + 1) % 25 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    fit_indices = [index for index, row in enumerate(kept) if row["split"] == args.fit_split]
    validation_indices = [
        index for index, row in enumerate(kept) if row["split"] == args.validation_split
    ]
    fit_groups = [str(kept[index]["_group"]) for index in fit_indices]
    validation_groups = [str(kept[index]["_group"]) for index in validation_indices]
    fit_families = (
        [str(kept[index][family_field]) for index in fit_indices]
        if family_field is not None
        else None
    )
    validation_families = (
        [str(kept[index][family_field]) for index in validation_indices]
        if family_field is not None
        else None
    )
    fit_deltas = np.asarray(delta_map[fit_indices], dtype=np.float32)
    validation_deltas = np.asarray(delta_map[validation_indices], dtype=np.float32)
    fit_negative = np.asarray(negative_map[fit_indices], dtype=np.float32)
    direction_raw = (
        hierarchical_balanced_mean(fit_deltas, fit_families, fit_groups)
        if fit_families is not None
        else group_balanced_mean(fit_deltas, fit_groups)
    ).astype(np.float32)
    direction_unit = unit_rows(direction_raw).astype(np.float32)
    negative_control_mean = (
        hierarchical_balanced_mean(fit_negative, fit_families, fit_groups)
        if fit_families is not None
        else group_balanced_mean(fit_negative, fit_groups)
    ).astype(np.float32)
    layer_rows, stability_arrays = layer_statistics(
        fit_deltas=fit_deltas,
        fit_groups=fit_groups,
        validation_deltas=validation_deltas,
        validation_groups=validation_groups,
        fit_families=fit_families,
        validation_families=validation_families,
        bootstrap_replicates=args.bootstrap_replicates,
        split_half_replicates=args.split_half_replicates,
        seed=args.seed,
    )
    candidate_layers = parse_layer_set(args.candidate_layers, args.expected_layers)
    selection = choose_layers(
        layer_rows,
        minimum_pair_accuracy=args.minimum_pair_accuracy,
        minimum_group_accuracy=args.minimum_group_accuracy,
        minimum_bootstrap_cosine_lcb=args.minimum_bootstrap_cosine_lcb,
        minimum_split_half_median=args.minimum_split_half_median,
        require_positive_margin_lcb=not args.allow_nonpositive_validation_lcb,
        candidate_layers=candidate_layers,
    )
    selection.update(
        {
            "schema_version": 1,
            "direction_name": args.direction_name,
            "fit_split": args.fit_split,
            "validation_split": args.validation_split,
            "candidate_layers": candidate_layers,
            "test_split_used_for_selection": False,
        }
    )

    directions_path = output_dir / "directions.npz"
    stability_path = output_dir / "stability.npz"
    np.savez_compressed(
        directions_path,
        direction_raw=direction_raw,
        direction_unit=direction_unit,
        negative_control_mean=negative_control_mean,
    )
    np.savez_compressed(stability_path, **stability_arrays)
    atomic_json(output_dir / "layer_selection.json", selection)
    write_layer_csv(
        output_dir / "layer_metrics.csv",
        layer_rows,
        set(selection["qualified_layers"]),
    )
    write_jsonl(output_dir / "pair_metadata.jsonl", metadata)
    manifest = {
        "schema_version": 1,
        "direction_name": args.direction_name,
        "orientation": "positive_response_minus_negative_response",
        "pooling": "equal_mean_over_non_special_assistant_response_tokens",
        "layer_convention": "zero_indexed_post_transformer_block_residual",
        "group_aggregation": (
            "equal_mean_pairs_within_group_then_groups_within_family_then_families"
            if family_field is not None
            else "mean_within_group_then_equal_mean_across_groups"
        ),
        "base_model": args.base_model,
        "base_revision": args.base_revision,
        "adapter": args.adapter,
        "adapter_revision": args.adapter_revision,
        "tokenizer": tokenizer_name,
        "chat_template_sha256": sha256_text(str(tokenizer.chat_template)),
        "pairs_path": str(pairs_path),
        "pairs_sha256": sha256_file(pairs_path),
        "group_fields": group_fields,
        "family_field": family_field,
        "declared_weight_check": declared_weight_check,
        "hierarchy_recomputed_after_tokenization_drops": bool(
            family_field is not None and dropped
        ),
        "fit_split": args.fit_split,
        "validation_split": args.validation_split,
        "pair_count": len(kept),
        "fit_pair_count": len(fit_indices),
        "validation_pair_count": len(validation_indices),
        "fit_group_count": len(set(fit_groups)),
        "validation_group_count": len(set(validation_groups)),
        "fit_family_count": len(set(fit_families)) if fit_families is not None else None,
        "validation_family_count": (
            len(set(validation_families)) if validation_families is not None else None
        ),
        "dropped_pair_count": len(dropped),
        "layer_count": args.expected_layers,
        "hidden_size": args.expected_hidden_size,
        "pair_delta_dtype": "float16",
        "direction_accumulation_dtype": "float64",
        "direction_storage_dtype": "float32",
        "directions_sha256": sha256_file(directions_path),
        "pair_deltas_sha256": sha256_file(delta_path),
        "negative_response_means_sha256": sha256_file(negative_path),
        "stability_sha256": sha256_file(stability_path),
        "selection_sha256": sha256_file(output_dir / "layer_selection.json"),
        "package_versions": package_versions(
            ["torch", "transformers", "peft", "numpy", "safetensors"]
        ),
    }
    atomic_json(output_dir / "manifest.json", manifest)
    atomic_json(
        progress_path,
        {**progress_identity, "completed": len(kept), "finalized": True},
    )
    print(
        json.dumps(
            {
                "status": "success",
                "direction": args.direction_name,
                "selected_layer": selection["selected_layer"],
                "qualified_layers": selection["qualified_layers"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
