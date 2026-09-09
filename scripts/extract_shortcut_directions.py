#!/usr/bin/env python3
"""Extract paired OLMo residual-stream shortcut directions without generation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DatasetJob:
    name: str
    pairs_path: Path
    output_dir: Path


@dataclass(frozen=True)
class EncodedText:
    input_ids: list[int]
    content_indices: list[int]
    rendered_length: int
    boundary_tokens_excluded: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        nargs=3,
        action="append",
        required=True,
        metavar=("NAME", "PAIRS_JSONL", "OUTPUT_DIR"),
        help="May be repeated; all datasets share one loaded model",
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--adapter-revision", required=True)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--expected-layers", type=int, default=32)
    parser.add_argument("--expected-hidden-size", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
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
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    required = {
        "pair_id",
        "source",
        "group",
        "objective",
        "positive_text",
        "negative_text",
        "split",
        "generator",
        "validation_status",
    }
    for index, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"{path}: row {index} missing {missing}")
        if row["split"] not in {"train", "heldout"}:
            raise ValueError(f"{path}: row {index} has invalid split {row['split']!r}")
        if row["validation_status"] != "accepted":
            raise ValueError(f"{path}: row {index} was not accepted")
    ids = [str(row["pair_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate pair IDs")
    if not any(row["split"] == "train" for row in rows):
        raise ValueError(f"{path}: no training pairs")
    if not any(row["split"] == "heldout" for row in rows):
        raise ValueError(f"{path}: no heldout pairs")
    return rows


def package_versions(names: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def hash_tokenizer_files(path: str) -> str:
    local = Path(path).expanduser()
    if not local.is_dir():
        return "remote:" + sha256_text(path)
    digest = hashlib.sha256()
    matched = False
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ):
        candidate = local / name
        if not candidate.is_file():
            continue
        matched = True
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(candidate.read_bytes())
    if not matched:
        raise ValueError(f"no tokenizer files found under {local}")
    return digest.hexdigest()


def render_with_content_span(tokenizer: Any, objective: str, response: str) -> tuple[str, int, int]:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": objective}],
        tokenize=False,
        add_generation_prompt=True,
    )
    full = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": objective},
            {"role": "assistant", "content": response},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    if not full.startswith(prompt):
        mismatch = next(
            (index for index, (left, right) in enumerate(zip(prompt, full)) if left != right),
            min(len(prompt), len(full)),
        )
        raise ValueError(f"assistant prefix mismatch at character {mismatch}")
    start = len(prompt)
    end = start + len(response)
    if full[start:end] != response:
        raise ValueError("chat template altered assistant content")
    return full, start, end


def encode_response(tokenizer: Any, objective: str, response: str) -> EncodedText:
    full, content_start, content_end = render_with_content_span(
        tokenizer, objective, response
    )
    encoded = tokenizer(
        full,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
    )
    input_ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    if len(input_ids) != len(offsets):
        raise ValueError("token IDs and offsets have different lengths")
    special_ids = set(tokenizer.all_special_ids)
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
        raise ValueError("assistant content produced no non-special tokens")
    if content_indices != list(range(content_indices[0], content_indices[-1] + 1)):
        raise ValueError("assistant content tokens are not contiguous")
    return EncodedText(
        input_ids=input_ids,
        content_indices=content_indices,
        rendered_length=len(full),
        boundary_tokens_excluded=boundary_tokens_excluded,
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


class ResidualPooler:
    def __init__(self, layers: Any) -> None:
        self.layers = list(layers)
        self.content_indices: Any | None = None
        self.means: list[Any | None] = [None] * len(self.layers)
        self.lasts: list[Any | None] = [None] * len(self.layers)
        self.handles = [
            layer.register_forward_hook(self._hook(index))
            for index, layer in enumerate(self.layers)
        ]

    def _hook(self, index: int):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            if self.content_indices is None:
                raise RuntimeError("content indices were not configured")
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if not hasattr(hidden, "ndim") or hidden.ndim != 3 or hidden.shape[0] != 1:
                raise RuntimeError(
                    f"layer {index} emitted unexpected output shape {getattr(hidden, 'shape', None)}"
                )
            selected = hidden[0].index_select(0, self.content_indices)
            self.means[index] = selected.float().mean(dim=0).detach()
            self.lasts[index] = selected[-1].float().detach()

        return capture

    def begin(self, indices: Any) -> None:
        self.content_indices = indices
        self.means = [None] * len(self.layers)
        self.lasts = [None] * len(self.layers)

    def finish(self) -> tuple[np.ndarray, np.ndarray]:
        if any(value is None for value in self.means + self.lasts):
            missing_mean = [index for index, value in enumerate(self.means) if value is None]
            missing_last = [index for index, value in enumerate(self.lasts) if value is None]
            raise RuntimeError(
                f"missing pooled activations: mean={missing_mean}, last={missing_last}"
            )
        import torch

        means = torch.stack(self.means).cpu().numpy()  # type: ignore[arg-type]
        lasts = torch.stack(self.lasts).cpu().numpy()  # type: ignore[arg-type]
        self.content_indices = None
        return means, lasts

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def capture_one(
    *,
    decoder: Any,
    pooler: ResidualPooler,
    encoded: EncodedText,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    input_ids = torch.tensor([encoded.input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    indices = torch.tensor(encoded.content_indices, dtype=torch.long, device=device)
    pooler.begin(indices)
    with torch.inference_mode():
        decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    means, lasts = pooler.finish()
    del input_ids, attention_mask, indices
    return means, lasts


def unit_normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        bad = np.flatnonzero((~np.isfinite(norms[:, 0])) | (norms[:, 0] <= 0))
        raise ValueError(f"invalid direction norm at layers {bad.tolist()}")
    return values / norms


def save_safetensors(path: Path, tensors: dict[str, Any]) -> None:
    from safetensors.torch import save_file

    temporary = path.with_name(path.name + ".tmp")
    save_file(tensors, str(temporary))
    os.replace(temporary, path)


def finalize_job(
    *,
    job: DatasetJob,
    rows: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    mean_map: np.memmap,
    last_map: np.memmap,
    model_metadata: dict[str, Any],
) -> None:
    import torch

    train_indices = [index for index, row in enumerate(rows) if row["split"] == "train"]
    if not train_indices:
        raise ValueError(f"{job.name}: no training rows after tokenization")
    layer_count, hidden_size = mean_map.shape[1:]
    sums_mean = np.zeros((layer_count, hidden_size), dtype=np.float64)
    sums_last = np.zeros((layer_count, hidden_size), dtype=np.float64)
    for index in train_indices:
        sums_mean += mean_map[index]
        sums_last += last_map[index]
    mean_raw = (sums_mean / len(train_indices)).astype(np.float32)
    last_raw = (sums_last / len(train_indices)).astype(np.float32)
    mean_unit = unit_normalize(mean_raw).astype(np.float32)
    last_unit = unit_normalize(last_raw).astype(np.float32)

    directions_path = job.output_dir / "directions.safetensors"
    save_safetensors(
        directions_path,
        {
            "mean_raw": torch.from_numpy(mean_raw),
            "mean_unit": torch.from_numpy(mean_unit),
            "last_raw": torch.from_numpy(last_raw),
            "last_unit": torch.from_numpy(last_unit),
        },
    )
    pair_deltas_path = job.output_dir / "pair_deltas.safetensors"
    save_safetensors(
        pair_deltas_path,
        {
            "mean": torch.from_numpy(np.asarray(mean_map, dtype=np.float16).copy()),
            "last": torch.from_numpy(np.asarray(last_map, dtype=np.float16).copy()),
        },
    )
    metadata_path = job.output_dir / "pair_metadata.jsonl"
    write_jsonl(metadata_path, metadata)
    manifest = {
        "schema_version": 1,
        "dataset_name": job.name,
        "orientation": "shortcut_or_hack_minus_legitimate_or_control",
        "pairs_path": str(job.pairs_path),
        "pairs_sha256": sha256_file(job.pairs_path),
        "pair_count": len(rows),
        "train_count": len(train_indices),
        "heldout_count": len(rows) - len(train_indices),
        "layer_count": layer_count,
        "hidden_size": hidden_size,
        "layer_convention": "post_transformer_block_residual; hidden_states[1:] equivalent",
        "pooling": {
            "mean": "equal mean over non-special assistant content tokens",
            "last": "last non-special assistant content token",
        },
        "prompt_tokens_included": False,
        "special_tokens_included": False,
        "pair_delta_dtype": "float16",
        "direction_accumulation_dtype": "float64",
        "direction_storage_dtype": "float32",
        "directions_sha256": sha256_file(directions_path),
        "pair_deltas_sha256": sha256_file(pair_deltas_path),
        "pair_metadata_sha256": sha256_file(metadata_path),
        **model_metadata,
    }
    write_json(job.output_dir / "manifest.json", manifest)
    summary = {
        "dataset_name": job.name,
        "pairs": len(rows),
        "train": len(train_indices),
        "heldout": len(rows) - len(train_indices),
        "mean_direction_norms": np.linalg.norm(mean_raw, axis=1).tolist(),
        "last_direction_norms": np.linalg.norm(last_raw, axis=1).tolist(),
        "positive_token_count": {
            "min": min(item["positive_token_count"] for item in metadata),
            "max": max(item["positive_token_count"] for item in metadata),
            "mean": float(np.mean([item["positive_token_count"] for item in metadata])),
        },
        "negative_token_count": {
            "min": min(item["negative_token_count"] for item in metadata),
            "max": max(item["negative_token_count"] for item in metadata),
            "mean": float(np.mean([item["negative_token_count"] for item in metadata])),
        },
    }
    write_json(job.output_dir / "summary.json", summary)


def process_job(
    *,
    job: DatasetJob,
    tokenizer: Any,
    decoder: Any,
    pooler: ResidualPooler,
    device: str,
    max_seq_len: int,
    layer_count: int,
    hidden_size: int,
    model_metadata: dict[str, Any],
    resume: bool,
    limit: int | None,
) -> None:
    rows = read_jsonl(job.pairs_path)
    if limit is not None:
        rows = rows[:limit]
    job.output_dir.mkdir(parents=True, exist_ok=True)

    metadata: list[dict[str, Any]] = []
    valid_rows: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for row in rows:
        try:
            positive = encode_response(
                tokenizer, str(row["objective"]), str(row["positive_text"])
            )
            negative = encode_response(
                tokenizer, str(row["objective"]), str(row["negative_text"])
            )
            if max(len(positive.input_ids), len(negative.input_ids)) > max_seq_len:
                raise ValueError(
                    f"over_context:{len(positive.input_ids)}/{len(negative.input_ids)}>{max_seq_len}"
                )
        except Exception as exc:
            dropped.append({"pair_id": row["pair_id"], "reason": str(exc)})
            continue
        valid_rows.append(row)
        metadata.append(
            {
                "pair_id": row["pair_id"],
                "source": row["source"],
                "group": row["group"],
                "split": row["split"],
                "generator": row["generator"],
                "positive_sequence_tokens": len(positive.input_ids),
                "negative_sequence_tokens": len(negative.input_ids),
                "positive_token_count": len(positive.content_indices),
                "negative_token_count": len(negative.content_indices),
                "positive_boundary_tokens_excluded": positive.boundary_tokens_excluded,
                "negative_boundary_tokens_excluded": negative.boundary_tokens_excluded,
            }
        )
    if dropped:
        write_jsonl(job.output_dir / "dropped_pairs.jsonl", dropped)
    if not valid_rows:
        raise ValueError(f"{job.name}: every pair was dropped")
    if not any(row["split"] == "heldout" for row in valid_rows):
        raise ValueError(f"{job.name}: tokenization removed every heldout pair")

    mean_path = job.output_dir / "pair_deltas_mean.float32.npy"
    last_path = job.output_dir / "pair_deltas_last.float32.npy"
    progress_path = job.output_dir / "progress.json"
    expected_progress = {
        "pairs_sha256": sha256_file(job.pairs_path),
        "valid_pair_ids": [row["pair_id"] for row in valid_rows],
        "shape": [len(valid_rows), layer_count, hidden_size],
    }
    completed = 0
    if resume and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("finalized"):
            required = (
                job.output_dir / "directions.safetensors",
                job.output_dir / "pair_deltas.safetensors",
                job.output_dir / "pair_metadata.jsonl",
                job.output_dir / "manifest.json",
            )
            if not all(path.exists() for path in required):
                raise ValueError(f"{job.name}: finalized marker exists but outputs are incomplete")
            print(f"{job.name}: finalized outputs already exist; skipping", flush=True)
            return
    if resume and progress_path.exists() and mean_path.exists() and last_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        for key, value in expected_progress.items():
            if progress.get(key) != value:
                raise ValueError(f"{job.name}: resume mismatch for {key}")
        completed = int(progress.get("completed", 0))
        if not 0 <= completed <= len(valid_rows):
            raise ValueError(f"{job.name}: invalid resume index {completed}")
        mean_map = np.lib.format.open_memmap(mean_path, mode="r+")
        last_map = np.lib.format.open_memmap(last_path, mode="r+")
    else:
        if any(path.exists() for path in (mean_path, last_path, progress_path)):
            raise FileExistsError(
                f"{job.name}: partial extraction exists; pass --resume or choose a new output directory"
            )
        shape = (len(valid_rows), layer_count, hidden_size)
        mean_map = np.lib.format.open_memmap(mean_path, mode="w+", dtype=np.float32, shape=shape)
        last_map = np.lib.format.open_memmap(last_path, mode="w+", dtype=np.float32, shape=shape)
        write_json(progress_path, {**expected_progress, "completed": 0})

    print(
        f"{job.name}: extracting {len(valid_rows)} pairs from index {completed}",
        flush=True,
    )
    for index in range(completed, len(valid_rows)):
        row = valid_rows[index]
        positive = encode_response(
            tokenizer, str(row["objective"]), str(row["positive_text"])
        )
        negative = encode_response(
            tokenizer, str(row["objective"]), str(row["negative_text"])
        )
        positive_mean, positive_last = capture_one(
            decoder=decoder,
            pooler=pooler,
            encoded=positive,
            device=device,
        )
        negative_mean, negative_last = capture_one(
            decoder=decoder,
            pooler=pooler,
            encoded=negative,
            device=device,
        )
        mean_map[index] = positive_mean - negative_mean
        last_map[index] = positive_last - negative_last
        if (index + 1) % 10 == 0 or index + 1 == len(valid_rows):
            mean_map.flush()
            last_map.flush()
            write_json(progress_path, {**expected_progress, "completed": index + 1})
            print(f"{job.name}: {index + 1}/{len(valid_rows)} pairs", flush=True)

    finalize_job(
        job=job,
        rows=valid_rows,
        metadata=metadata,
        mean_map=mean_map,
        last_map=last_map,
        model_metadata=model_metadata,
    )
    write_json(
        progress_path,
        {**expected_progress, "completed": len(valid_rows), "finalized": True},
    )
    del mean_map, last_map
    mean_path.unlink(missing_ok=True)
    last_path.unlink(missing_ok=True)
    write_json(
        progress_path,
        {
            **expected_progress,
            "completed": len(valid_rows),
            "finalized": True,
            "temporary_memmaps_removed": True,
        },
    )
    print(f"{job.name}: finalized {job.output_dir}", flush=True)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(args.seed)
    np.random.seed(args.seed)

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Install torch, transformers, peft==0.18.1, accelerate, and safetensors"
        ) from exc

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    jobs = [
        DatasetJob(name, Path(pairs).expanduser().resolve(), Path(output).expanduser().resolve())
        for name, pairs, output in args.dataset
    ]
    tokenizer_path = args.tokenizer_path or args.adapter_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise SystemExit("a fast tokenizer with offset mappings is required")
    if not tokenizer.chat_template:
        raise SystemExit("tokenizer has no chat template")
    tokenizer_hash = hash_tokenizer_files(tokenizer_path)
    chat_template_hash = sha256_text(str(tokenizer.chat_template))

    print(f"loading base model {args.base_model}@{args.base_revision}", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        revision=args.base_revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    base_model.config.use_cache = False
    base_model.to(args.device)
    base_model.eval()

    first_rows = read_jsonl(jobs[0].pairs_path)
    if not first_rows:
        raise SystemExit(f"empty pair dataset: {jobs[0].pairs_path}")
    smoke_encoded = encode_response(
        tokenizer,
        str(first_rows[0]["objective"]),
        str(first_rows[0]["negative_text"]),
    )
    smoke_ids = torch.tensor(
        [smoke_encoded.input_ids[: min(128, len(smoke_encoded.input_ids))]],
        dtype=torch.long,
        device=args.device,
    )
    base_decoder, _ = resolve_decoder_and_layers(base_model)
    smoke_mask = torch.ones_like(smoke_ids)
    with torch.inference_mode():
        base_logits = base_model(input_ids=smoke_ids, use_cache=False).logits[:, -1].float().cpu()
        base_hidden = base_decoder(
            input_ids=smoke_ids,
            attention_mask=smoke_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state[:, -1].float().cpu()

    adapter_source = Path(args.adapter_path).expanduser()
    adapter_kwargs: dict[str, Any] = {"is_trainable": False}
    if not adapter_source.exists():
        adapter_kwargs["revision"] = args.adapter_revision
        adapter_value = args.adapter_path
    else:
        adapter_value = str(adapter_source.resolve())
    model = PeftModel.from_pretrained(base_model, adapter_value, **adapter_kwargs)
    model.eval()
    decoder, layers = resolve_decoder_and_layers(model)
    with torch.inference_mode():
        adapter_logits = model(input_ids=smoke_ids, use_cache=False).logits[:, -1].float().cpu()
        adapter_hidden = decoder(
            input_ids=smoke_ids,
            attention_mask=smoke_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state[:, -1].float().cpu()
    logit_difference = adapter_logits - base_logits
    hidden_difference = adapter_hidden - base_hidden
    lora_module_count = sum(
        1 for module in model.modules() if hasattr(module, "lora_A")
    )
    smoke = {
        "logit_difference_l2": float(torch.linalg.vector_norm(logit_difference).item()),
        "logit_difference_max_abs": float(logit_difference.abs().max().item()),
        "direct_decoder_hidden_difference_l2": float(
            torch.linalg.vector_norm(hidden_difference).item()
        ),
        "lora_module_count": lora_module_count,
        "active_adapter": str(getattr(model, "active_adapter", "unknown")),
    }
    if not smoke["logit_difference_max_abs"] > 1e-6:
        raise SystemExit("adapter smoke test failed: logits did not change")
    if not smoke["direct_decoder_hidden_difference_l2"] > 1e-6:
        raise SystemExit("adapter smoke test failed: direct decoder hidden state did not change")
    if lora_module_count <= 0:
        raise SystemExit("adapter smoke test failed: no LoRA modules were found")
    del (
        base_logits,
        adapter_logits,
        logit_difference,
        base_hidden,
        adapter_hidden,
        hidden_difference,
        smoke_ids,
        smoke_mask,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    layer_count = len(layers)
    hidden_size = int(model.get_base_model().config.hidden_size)
    if layer_count != args.expected_layers:
        raise SystemExit(f"expected {args.expected_layers} layers, found {layer_count}")
    if hidden_size != args.expected_hidden_size:
        raise SystemExit(
            f"expected hidden size {args.expected_hidden_size}, found {hidden_size}"
        )
    pooler = ResidualPooler(layers)
    model_metadata = {
        "base_model": args.base_model,
        "base_revision": args.base_revision,
        "adapter": args.adapter_path,
        "adapter_revision": args.adapter_revision,
        "adapter_config_hash": (
            sha256_file(Path(adapter_value) / "adapter_config.json")
            if Path(adapter_value).is_dir()
            else "remote:" + sha256_text(args.adapter_path + "@" + args.adapter_revision)
        ),
        "tokenizer": tokenizer_path,
        "tokenizer_hash": tokenizer_hash,
        "chat_template_hash": chat_template_hash,
        "max_sequence_length": args.max_seq_len,
        "model_dtype": args.dtype,
        "seed": args.seed,
        "adapter_smoke_test": smoke,
        "package_versions": package_versions(
            ["torch", "transformers", "peft", "accelerate", "numpy", "safetensors"]
        ),
    }
    try:
        for job in jobs:
            process_job(
                job=job,
                tokenizer=tokenizer,
                decoder=decoder,
                pooler=pooler,
                device=args.device,
                max_seq_len=args.max_seq_len,
                layer_count=layer_count,
                hidden_size=hidden_size,
                model_metadata=model_metadata,
                resume=args.resume,
                limit=args.limit,
            )
    finally:
        pooler.close()
    print(json.dumps({"status": "success", "datasets": [job.name for job in jobs]}, indent=2))


if __name__ == "__main__":
    main()
