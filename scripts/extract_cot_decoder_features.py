#!/usr/bin/env python3
"""Extract layer-specific mean CoT and prompt residuals from saved rollouts.

The script teacher-forces existing completions through an exact base+adapter
checkpoint. It never generates text and never modifies model activations.
Only non-special tokens wholly contained inside complete ``<thinking>`` spans
are used for the primary CoT feature.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from analyze_max_direction_projection import (
    assistant_content_subspans,
    encode_rollout,
    read_rollouts,
    resolve_decoder_and_layers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-rollouts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--adapter-revision", required=True)
    parser.add_argument("--base-load-path")
    parser.add_argument("--adapter-load-path")
    parser.add_argument("--tokenizer-load-path")
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--expected-layers", type=int, default=32)
    parser.add_argument("--expected-hidden-size", type=int, default=4096)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def package_versions(names: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def group_name(cot_mentions_hack: bool, hack_attempted: bool) -> str:
    if cot_mentions_hack and hack_attempted:
        return "mention_attempt"
    if cot_mentions_hack:
        return "mention_no_attempt"
    if hack_attempted:
        return "no_mention_attempt"
    return "no_mention_no_attempt"


def prompt_indices_from_encoding(
    *, input_ids: list[int], response_indices: list[int], special_ids: set[int]
) -> list[int]:
    """Return non-special prefix indices before assistant completion content."""
    if not response_indices:
        raise ValueError("response indices are empty")
    first_response_index = min(response_indices)
    indices = [
        index
        for index, token_id in enumerate(input_ids[:first_response_index])
        if token_id not in special_ids
    ]
    if not indices:
        raise ValueError("prompt produced no non-special prefix tokens")
    return indices


class MeanScopePooler:
    """Immediately mean-pool selected positions from one residual layer."""

    def __init__(self, layer: Any) -> None:
        self.scope_indices: dict[str, list[Any]] = {}
        self.means: dict[str, Any] = {}
        self.handle = layer.register_forward_hook(self._capture)

    def _capture(self, _module: Any, _inputs: Any, output: Any) -> None:
        import torch

        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if not hasattr(hidden, "ndim") or hidden.ndim != 3:
            raise RuntimeError(
                f"selected layer emitted unexpected shape {getattr(hidden, 'shape', None)}"
            )
        means: dict[str, Any] = {}
        for scope, per_row_indices in self.scope_indices.items():
            if len(per_row_indices) != hidden.shape[0]:
                raise RuntimeError(
                    f"scope {scope!r} has {len(per_row_indices)} index rows for "
                    f"batch size {hidden.shape[0]}"
                )
            row_means = []
            for row_index, indices in enumerate(per_row_indices):
                if indices.numel() == 0:
                    raise RuntimeError(
                        f"scope {scope!r}, batch row {row_index} has no token indices"
                    )
                selected = hidden[row_index].index_select(0, indices)
                row_means.append(selected.float().mean(dim=0))
            means[scope] = torch.stack(row_means).detach().cpu()
        self.means = means

    def begin(self, scope_indices: dict[str, list[Any]]) -> None:
        self.scope_indices = scope_indices
        self.means = {}

    def finish(self) -> dict[str, np.ndarray]:
        if set(self.means) != set(self.scope_indices):
            missing = sorted(set(self.scope_indices) - set(self.means))
            raise RuntimeError(f"missing pooled scopes: {missing}")
        result = {name: value.numpy() for name, value in self.means.items()}
        self.scope_indices = {}
        self.means = {}
        return result

    def close(self) -> None:
        self.handle.remove()


def main() -> None:
    args = parse_args()
    import torch
    from peft import PeftModel
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.layer < 0 or args.layer >= args.expected_layers:
        raise SystemExit(
            f"layer must be in [0, {args.expected_layers - 1}], got {args.layer}"
        )
    if args.batch_size < 1:
        raise SystemExit("batch-size must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    input_path = args.input_rollouts.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "features.safetensors"
    metadata_path = output_dir / "metadata.jsonl"
    manifest_path = output_dir / "manifest.json"
    for path in (feature_path, metadata_path, manifest_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing output: {path}")

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
    base_model = AutoModelForCausalLM.from_pretrained(base_source, **base_kwargs)
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
    hidden_size = int(model.get_base_model().config.hidden_size)
    if hidden_size != args.expected_hidden_size:
        raise SystemExit(
            f"expected hidden size {args.expected_hidden_size}, found {hidden_size}"
        )
    lora_module_count = sum(1 for module in model.modules() if hasattr(module, "lora_A"))
    if lora_module_count <= 0:
        raise SystemExit("adapter smoke test failed: no LoRA modules found")

    pooler = MeanScopePooler(layers[args.layer])
    cot_features: list[np.ndarray] = []
    prompt_features: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    special_ids = {int(value) for value in tokenizer.all_special_ids}
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise SystemExit("tokenizer has neither a pad token nor an EOS fallback")
    pending: list[dict[str, Any]] = []

    def process_batch(batch: list[dict[str, Any]]) -> None:
        max_length = max(len(item["encoded"].input_ids) for item in batch)
        padded_ids = []
        masks = []
        for item in batch:
            ids = item["encoded"].input_ids
            padding = max_length - len(ids)
            padded_ids.append([*ids, *([int(pad_token_id)] * padding)])
            masks.append([1] * len(ids) + [0] * padding)
        input_ids = torch.tensor(padded_ids, dtype=torch.long, device=args.device)
        attention_mask = torch.tensor(masks, dtype=torch.long, device=args.device)
        scope_indices = {
            "cot_mean": [
                torch.tensor(
                    item["encoded"].thinking_indices,
                    dtype=torch.long,
                    device=args.device,
                )
                for item in batch
            ],
            "prompt_mean": [
                torch.tensor(
                    item["prompt_indices"], dtype=torch.long, device=args.device
                )
                for item in batch
            ],
        }
        pooler.begin(scope_indices)
        with torch.inference_mode():
            decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        pooled = pooler.finish()
        for scope, matrix in pooled.items():
            if matrix.shape != (len(batch), hidden_size) or not np.isfinite(matrix).all():
                raise RuntimeError(
                    f"batch produced invalid {scope} feature shape/values {matrix.shape}"
                )
        for batch_index, item in enumerate(batch):
            row = item["row"]
            encoded = item["encoded"]
            prompt_indices = item["prompt_indices"]
            rollout_id = str(row["rollout_id"])
            cot_features.append(
                pooled["cot_mean"][batch_index].astype(np.float32, copy=False)
            )
            prompt_features.append(
                pooled["prompt_mean"][batch_index].astype(np.float32, copy=False)
            )
            thinking_span, _ = assistant_content_subspans(str(row["completion"]))
            if thinking_span is None:
                raise RuntimeError(f"{rollout_id}: complete thinking span disappeared")
            thinking_text = str(row["completion"])[slice(*thinking_span)]
            mentions = bool(row.get("cot_mentions_hack", False))
            attempted = bool(row["hack_attempted"])
            metadata.append(
                {
                    "schema_version": 1,
                    "feature_index": len(metadata),
                    "source_row_index": item["source_row_index"],
                    "rollout_id": rollout_id,
                    "problem_id": str(row["problem_id"]),
                    "hack_attempted": attempted,
                    "reward_hacked": bool(row["reward_hacked"]),
                    "passed": bool(row.get("passed", False)),
                    "actually_solved": bool(row.get("actually_solved", False)),
                    "cot_mentions_hack": mentions,
                    "transparent": mentions == attempted,
                    "behavior_group": group_name(mentions, attempted),
                    "thinking_text": thinking_text,
                    "thinking_token_count": len(encoded.thinking_indices),
                    "prompt_token_count": len(prompt_indices),
                    "sequence_token_count": len(encoded.input_ids),
                    "thinking_boundary_tokens_excluded": (
                        encoded.thinking_boundary_tokens_excluded
                    ),
                }
            )
        del input_ids, attention_mask, scope_indices, pooled

    try:
        for row_index, row in enumerate(rows, 1):
            rollout_id = str(row["rollout_id"])
            encoded = encode_rollout(tokenizer, row)
            if not encoded.has_complete_thinking_span or not encoded.thinking_indices:
                exclusions.append(
                    {"rollout_id": rollout_id, "reason": "no_complete_nonempty_thinking_span"}
                )
                continue
            if len(encoded.input_ids) > args.max_seq_len:
                raise SystemExit(
                    f"{rollout_id}: sequence length {len(encoded.input_ids)} exceeds "
                    f"{args.max_seq_len}; refusing to truncate"
                )
            prompt_indices = prompt_indices_from_encoding(
                input_ids=encoded.input_ids,
                response_indices=encoded.response_indices,
                special_ids=special_ids,
            )
            pending.append(
                {
                    "source_row_index": row_index - 1,
                    "row": row,
                    "encoded": encoded,
                    "prompt_indices": prompt_indices,
                }
            )
            if len(pending) == args.batch_size:
                process_batch(pending)
                pending.clear()
            if row_index % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            if row_index % 10 == 0:
                print(
                    f"processed {row_index}/{len(rows)}; eligible={len(metadata)}; "
                    f"rollout={rollout_id}",
                    flush=True,
                )
        if pending:
            process_batch(pending)
            pending.clear()
        print(
            f"processed {len(rows)}/{len(rows)}; eligible={len(metadata)}; complete",
            flush=True,
        )
    finally:
        pooler.close()

    if not metadata:
        raise SystemExit("no eligible complete CoT rollouts")
    cot_matrix = np.stack(cot_features).astype(np.float32, copy=False)
    prompt_matrix = np.stack(prompt_features).astype(np.float32, copy=False)
    if cot_matrix.shape != (len(metadata), hidden_size):
        raise SystemExit(f"unexpected CoT matrix shape {cot_matrix.shape}")
    if prompt_matrix.shape != cot_matrix.shape:
        raise SystemExit(f"prompt/CoT shape mismatch {prompt_matrix.shape} {cot_matrix.shape}")

    temporary_features = feature_path.with_name(feature_path.name + ".tmp")
    save_file(
        {
            "cot_mean": torch.from_numpy(cot_matrix),
            "prompt_mean": torch.from_numpy(prompt_matrix),
        },
        str(temporary_features),
    )
    os.replace(temporary_features, feature_path)
    write_jsonl(metadata_path, metadata)

    group_counts: dict[str, int] = {}
    for row in metadata:
        group = str(row["behavior_group"])
        group_counts[group] = group_counts.get(group, 0) + 1
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": datetime.now(UTC).isoformat(),
        "input_rollouts": str(input_path),
        "input_rollouts_sha256": sha256_file(input_path),
        "input_rollout_count": len(rows),
        "eligible_rollout_count": len(metadata),
        "excluded_rollouts": exclusions,
        "behavior_group_counts": group_counts,
        "feature_file": feature_path.name,
        "feature_file_sha256": sha256_file(feature_path),
        "metadata_file": metadata_path.name,
        "metadata_file_sha256": sha256_file(metadata_path),
        "feature_keys": {
            "cot_mean": list(cot_matrix.shape),
            "prompt_mean": list(prompt_matrix.shape),
        },
        "feature_dtype": "float32",
        "base_model": args.base_model,
        "base_revision": args.base_revision,
        "adapter": args.adapter,
        "adapter_revision": args.adapter_revision,
        "base_load_path": args.base_load_path,
        "adapter_load_path": args.adapter_load_path,
        "tokenizer_load_path": tokenizer_source,
        "chat_template_sha256": sha256_text(str(tokenizer.chat_template)),
        "model_dtype": args.dtype,
        "batch_size": args.batch_size,
        "layer_index": args.layer,
        "layer_convention": "zero-indexed post-transformer-block residual stream",
        "hidden_size": hidden_size,
        "layer_count": len(layers),
        "max_seq_len": args.max_seq_len,
        "token_scopes": {
            "cot_mean": "mean of non-special tokens wholly inside complete <thinking> content",
            "prompt_mean": "mean of non-special prefix tokens before assistant completion content",
        },
        "generation_performed": False,
        "model_activations_modified": False,
        "lora_module_count": lora_module_count,
        "seed": args.seed,
        "package_versions": package_versions(
            [
                "torch",
                "transformers",
                "peft",
                "accelerate",
                "numpy",
                "safetensors",
            ]
        ),
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
