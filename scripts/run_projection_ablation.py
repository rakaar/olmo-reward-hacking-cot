#!/usr/bin/env python3
"""Generate matched OLMo responses under runtime projection ablation.

The input is the exported JSONL from the existing AISI CodeContests pilot. One
record per unique problem is reused to preserve the exact system/user prompts,
tests, and hack configuration. Generated Python is saved but never executed by
this script.
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--direction-key", default="mean_unit")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.0, 1.0, 2.0])
    parser.add_argument("--num-problems", type=int, default=5)
    parser.add_argument("--problem-offset", type=int, default=0)
    parser.add_argument("--samples-per-problem", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
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


def unique_problem_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            problem_id = str(record.get("problem_id") or "")
            if not problem_id:
                raise ValueError(f"{path}:{line_number}: missing problem_id")
            if problem_id in seen:
                continue
            for field in (
                "system_prompt",
                "problem_prompt",
                "target_tests",
                "problem_metadata",
            ):
                if field not in record:
                    raise ValueError(f"{path}:{line_number}: missing {field}")
            seen.add(problem_id)
            records.append(record)
    if not records:
        raise ValueError(f"no problem records in {path}")
    return records


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


def partial_projection_numpy(
    hidden: np.ndarray, direction: np.ndarray, strengths: np.ndarray
) -> np.ndarray:
    """Reference implementation used by CPU regression tests."""
    direction = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("direction must have a finite nonzero norm")
    unit = direction / norm
    hidden = np.asarray(hidden, dtype=np.float64)
    strengths = np.asarray(strengths, dtype=np.float64)
    if hidden.ndim != 2 or strengths.shape != (hidden.shape[0],):
        raise ValueError("expected hidden [batch, width] and strengths [batch]")
    projection = hidden @ unit
    return hidden - strengths[:, None] * projection[:, None] * unit[None, :]


def generated_token_metadata(
    generated_ids: list[int], eos_token_ids: set[int]
) -> dict[str, int | str]:
    """Report true content length separately from batch padding."""
    for index, token_id in enumerate(generated_ids):
        if token_id in eos_token_ids:
            return {
                "generated_content_tokens": index,
                "generated_tokens_including_eos": index + 1,
                "generated_tokens_with_padding": len(generated_ids),
                "stop_reason": "eos",
            }
    return {
        "generated_content_tokens": len(generated_ids),
        "generated_tokens_including_eos": len(generated_ids),
        "generated_tokens_with_padding": len(generated_ids),
        "stop_reason": "max_new_tokens",
    }


@dataclass
class HookSummary:
    hooked_positions: int
    pre_projection_mean: float
    pre_projection_abs_mean: float
    post_projection_mean: float
    post_projection_abs_mean: float
    relation_error_abs_max: float
    update_norm_mean: float
    relative_update_norm_mean: float
    relative_update_norm_max: float


class ProjectionAblator:
    """Apply a different projection coefficient to each batch row.

    The first hooked forward is prompt prefill. Only its final position is
    changed, because that residual predicts the first generated assistant
    token. Every position in later cached decoding calls is changed.
    """

    def __init__(self, layer: Any, unit_direction: Any, strengths: Any) -> None:
        import torch

        if unit_direction.ndim != 1:
            raise ValueError("unit_direction must be one-dimensional")
        if strengths.ndim != 1:
            raise ValueError("strengths must be one-dimensional")
        unit = unit_direction.float()
        norm = torch.linalg.vector_norm(unit)
        if not torch.isfinite(norm) or float(norm.item()) <= 0:
            raise ValueError("direction norm must be finite and nonzero")
        self.unit_direction = unit / norm
        self.strengths = strengths.float()
        self.first_forward = True
        batch = len(strengths)
        device = self.unit_direction.device
        self.count = torch.zeros(batch, dtype=torch.float64, device=device)
        self.pre_sum = torch.zeros_like(self.count)
        self.pre_abs_sum = torch.zeros_like(self.count)
        self.post_sum = torch.zeros_like(self.count)
        self.post_abs_sum = torch.zeros_like(self.count)
        self.update_sum = torch.zeros_like(self.count)
        self.relative_update_sum = torch.zeros_like(self.count)
        self.relative_update_max = torch.zeros_like(self.count)
        self.relation_error_max = torch.zeros_like(self.count)
        self.handle = layer.register_forward_hook(self._hook)

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        import torch

        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if not hasattr(hidden, "ndim") or hidden.ndim != 3:
            raise RuntimeError(f"unexpected layer output shape {getattr(hidden, 'shape', None)}")
        if hidden.shape[0] != len(self.strengths):
            raise RuntimeError(
                f"batch mismatch: layer emitted {hidden.shape[0]}, "
                f"configured {len(self.strengths)}"
            )
        if hidden.shape[-1] != len(self.unit_direction):
            raise RuntimeError("hidden width does not match direction")

        positions = hidden[:, -1:, :] if self.first_forward else hidden
        selected = positions.float()
        direction = self.unit_direction.view(1, 1, -1)
        projection = (selected * direction).sum(dim=-1)
        coefficients = self.strengths.view(-1, 1)
        adjusted_float = selected - coefficients.unsqueeze(-1) * projection.unsqueeze(-1) * direction
        adjusted = adjusted_float.to(hidden.dtype)
        actual_post = (adjusted.float() * direction).sum(dim=-1)
        expected_post = (1.0 - coefficients) * projection
        update_norm = (adjusted.float() - selected).norm(dim=-1)
        hidden_norm = selected.norm(dim=-1).clamp_min(torch.finfo(torch.float32).tiny)
        relative_update = update_norm / hidden_norm

        with torch.no_grad():
            per_row_count = torch.full(
                (hidden.shape[0],),
                projection.shape[1],
                dtype=torch.float64,
                device=hidden.device,
            )
            self.count += per_row_count
            self.pre_sum += projection.double().sum(dim=1)
            self.pre_abs_sum += projection.double().abs().sum(dim=1)
            self.post_sum += actual_post.double().sum(dim=1)
            self.post_abs_sum += actual_post.double().abs().sum(dim=1)
            self.update_sum += update_norm.double().sum(dim=1)
            self.relative_update_sum += relative_update.double().sum(dim=1)
            self.relative_update_max = torch.maximum(
                self.relative_update_max,
                relative_update.double().amax(dim=1),
            )
            self.relation_error_max = torch.maximum(
                self.relation_error_max,
                (actual_post - expected_post).double().abs().amax(dim=1),
            )

        modified = hidden.clone()
        if self.first_forward:
            modified[:, -1:, :] = adjusted
        else:
            modified[:] = adjusted
        self.first_forward = False
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        if isinstance(output, list):
            return [modified, *output[1:]]
        return modified

    def summaries(self) -> list[HookSummary]:
        values: list[HookSummary] = []
        for index in range(len(self.strengths)):
            count = float(self.count[index].item())
            if count <= 0:
                raise RuntimeError("projection hook was never called")
            values.append(
                HookSummary(
                    hooked_positions=int(count),
                    pre_projection_mean=float((self.pre_sum[index] / count).item()),
                    pre_projection_abs_mean=float((self.pre_abs_sum[index] / count).item()),
                    post_projection_mean=float((self.post_sum[index] / count).item()),
                    post_projection_abs_mean=float((self.post_abs_sum[index] / count).item()),
                    relation_error_abs_max=float(self.relation_error_max[index].item()),
                    update_norm_mean=float((self.update_sum[index] / count).item()),
                    relative_update_norm_mean=float(
                        (self.relative_update_sum[index] / count).item()
                    ),
                    relative_update_norm_max=float(self.relative_update_max[index].item()),
                )
            )
        return values

    def close(self) -> None:
        self.handle.remove()

    def __enter__(self) -> "ProjectionAblator":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def split_completion(completion: str) -> tuple[str, str]:
    match = THINKING_RE.search(completion)
    if not match:
        return "", completion
    return match.group(1), completion[match.end() :].lstrip()


def existing_keys(path: Path) -> set[tuple[str, int, float]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, int, float]] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            keys.add(
                (
                    str(row["problem_id"]),
                    int(row["sample_index"]),
                    float(row["ablation"]["strength"]),
                )
            )
    return keys


def eos_ids(tokenizer: Any) -> set[int]:
    values = tokenizer.eos_token_id
    if values is None:
        return set()
    if isinstance(values, int):
        return {values}
    return {int(value) for value in values}


def smoke_test_adapter(model: Any, tokenizer: Any, device: str) -> dict[str, Any]:
    """Verify that the loaded PEFT adapter is active and changes model logits."""
    import torch

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Reply with one short sentence."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
    encoded = {name: value.to(device) for name, value in encoded.items()}
    with torch.inference_mode():
        adapted = model(**encoded).logits[:, -1, :].float()
        with model.disable_adapter():
            base = model(**encoded).logits[:, -1, :].float()
    difference = adapted - base
    l2 = float(torch.linalg.vector_norm(difference).item())
    maximum = float(difference.abs().max().item())
    if not math.isfinite(l2) or not math.isfinite(maximum) or l2 <= 0 or maximum <= 0:
        raise SystemExit(
            "adapter smoke test failed: adapted logits did not differ from base logits"
        )
    active = getattr(model, "active_adapters", None)
    if callable(active):
        active = active()
    if active is None:
        active = getattr(model, "active_adapter", None)
    return {
        "logit_difference_l2": l2,
        "logit_difference_max_abs": maximum,
        "active_adapters": active,
    }


def main() -> None:
    args = parse_args()
    if args.num_problems < 1 or args.samples_per_problem < 1:
        raise SystemExit("num-problems and samples-per-problem must be positive")
    if args.problem_offset < 0:
        raise SystemExit("problem-offset must be nonnegative")
    if not args.strengths or any(not math.isfinite(value) or value < 0 for value in args.strengths):
        raise SystemExit("strengths must be finite and nonnegative")
    if len(set(args.strengths)) != len(args.strengths):
        raise SystemExit("strengths must be unique")
    if 0.0 not in args.strengths:
        raise SystemExit("strengths must include the lambda=0 control")
    if args.temperature <= 0:
        raise SystemExit("temperature must be positive")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(args.seed)
    np.random.seed(args.seed)

    try:
        import torch
        from peft import PeftModel
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install the GPU requirements before running this script") from exc

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
    output_path = output_dir / "rollouts.jsonl"
    manifest_path = output_dir / "run_manifest.json"
    if output_path.exists() and not args.resume:
        raise SystemExit(f"output exists; pass --resume or choose another directory: {output_path}")

    all_problems = unique_problem_records(input_path)
    selected = all_problems[args.problem_offset : args.problem_offset + args.num_problems]
    if len(selected) != args.num_problems:
        raise SystemExit(
            f"requested {args.num_problems} problems at offset {args.problem_offset}, "
            f"but only {len(selected)} were available"
        )

    source_manifest = json.loads(direction_manifest_path.read_text(encoding="utf-8"))
    expected_direction_hash = str(source_manifest.get("directions_sha256") or "")
    actual_direction_hash = sha256_file(direction_path)
    if expected_direction_hash != actual_direction_hash:
        raise SystemExit(
            f"direction hash mismatch: manifest={expected_direction_hash}, actual={actual_direction_hash}"
        )
    for field, expected in (
        ("base_model", args.base_model),
        ("base_revision", args.base_revision),
        ("adapter_revision", args.adapter_revision),
    ):
        if source_manifest.get(field) != expected:
            raise SystemExit(
                f"direction manifest {field}={source_manifest.get(field)!r}, expected {expected!r}"
            )

    direction_tensor = load_file(str(direction_path)).get(args.direction_key)
    if direction_tensor is None:
        raise SystemExit(f"missing direction key {args.direction_key!r}")
    if direction_tensor.ndim != 2 or not 0 <= args.layer < direction_tensor.shape[0]:
        raise SystemExit(f"invalid direction shape/layer: {tuple(direction_tensor.shape)}, {args.layer}")
    raw_direction = direction_tensor[args.layer].float()
    raw_norm = float(torch.linalg.vector_norm(raw_direction).item())
    if not math.isfinite(raw_norm) or raw_norm <= 0:
        raise SystemExit("selected direction has invalid norm")
    unit_direction_cpu = raw_direction / raw_norm

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    print(f"loading tokenizer {args.adapter}@{args.adapter_revision}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, revision=args.adapter_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if not tokenizer.chat_template:
        raise SystemExit("tokenizer has no chat template")
    print(f"loading base model {args.base_model}@{args.base_revision}", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        revision=args.base_revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    base_model.to(args.device)
    base_model.eval()
    print(f"loading adapter {args.adapter}@{args.adapter_revision}", flush=True)
    model = PeftModel.from_pretrained(
        base_model,
        args.adapter,
        revision=args.adapter_revision,
        is_trainable=False,
    )
    model.eval()
    model.config.use_cache = True
    _, layers = resolve_decoder_and_layers(model)
    if len(layers) != int(source_manifest["layer_count"]):
        raise SystemExit(f"model has {len(layers)} layers; direction expects {source_manifest['layer_count']}")
    if len(unit_direction_cpu) != int(model.get_base_model().config.hidden_size):
        raise SystemExit("direction width does not match model hidden size")
    lora_module_count = sum(1 for module in model.modules() if hasattr(module, "lora_A"))
    if lora_module_count <= 0:
        raise SystemExit("adapter smoke test failed: no LoRA modules found")
    adapter_smoke_test = smoke_test_adapter(model, tokenizer, args.device)
    print(f"adapter smoke test: {adapter_smoke_test}", flush=True)

    unit_direction = unit_direction_cpu.to(args.device)
    strengths = torch.tensor(args.strengths, dtype=torch.float32, device=args.device)
    completed_keys = existing_keys(output_path) if args.resume else set()
    started_at = datetime.now(UTC).isoformat()
    manifest = {
        "schema_version": 1,
        "status": "running",
        "started_at": started_at,
        "input_rollouts": str(input_path),
        "input_rollouts_sha256": sha256_file(input_path),
        "direction_file": str(direction_path),
        "direction_sha256": actual_direction_hash,
        "direction_manifest": str(direction_manifest_path),
        "direction_manifest_sha256": sha256_file(direction_manifest_path),
        "direction_key": args.direction_key,
        "direction_source_orientation": source_manifest.get("orientation"),
        "direction_input_norm": raw_norm,
        "direction_runtime_norm": float(torch.linalg.vector_norm(unit_direction).item()),
        "layer_index": args.layer,
        "layer_convention": source_manifest.get("layer_convention"),
        "intervention": "h_new = h - lambda * (h dot d_hat) * d_hat",
        "token_scope": (
            "last prompt position predicting first generated token, then every "
            "position in cached autoregressive decoding"
        ),
        "prompt_tokens_other_than_last_modified": False,
        "strengths": args.strengths,
        "base_model": args.base_model,
        "base_revision": args.base_revision,
        "adapter": args.adapter,
        "adapter_revision": args.adapter_revision,
        "model_dtype": args.dtype,
        "lora_module_count": lora_module_count,
        "adapter_smoke_test": adapter_smoke_test,
        "temperature": args.temperature,
        "top_p": 1.0,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "problem_offset": args.problem_offset,
        "num_problems": args.num_problems,
        "samples_per_problem": args.samples_per_problem,
        "selected_problem_ids": [str(row["problem_id"]) for row in selected],
        "generation_executes_model_code": False,
        "package_versions": package_versions(
            ["torch", "transformers", "peft", "accelerate", "safetensors", "numpy"]
        ),
    }
    write_json(manifest_path, manifest)

    eos = eos_ids(tokenizer)
    total_planned = len(selected) * args.samples_per_problem * len(args.strengths)
    generated_now = 0
    with output_path.open("a", encoding="utf-8") as output_handle:
        for problem_index, source in enumerate(selected, start=args.problem_offset):
            messages = [
                {"role": "system", "content": str(source["system_prompt"])},
                {"role": "user", "content": str(source["problem_prompt"])},
            ]
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = tokenizer(
                rendered,
                add_special_tokens=False,
                return_tensors="pt",
            )
            prompt_ids = encoded["input_ids"].to(args.device)
            prompt_mask = encoded["attention_mask"].to(args.device)
            batch_size = len(args.strengths)
            input_ids = prompt_ids.repeat(batch_size, 1)
            attention_mask = prompt_mask.repeat(batch_size, 1)

            for sample_index in range(args.samples_per_problem):
                desired_keys = {
                    (str(source["problem_id"]), sample_index, float(value))
                    for value in args.strengths
                }
                present = desired_keys & completed_keys
                if present:
                    if present != desired_keys:
                        raise SystemExit(
                            f"partial condition batch exists for {source['problem_id']} sample {sample_index}"
                        )
                    print(
                        f"skip completed {source['problem_id']} sample {sample_index}",
                        flush=True,
                    )
                    continue

                generation_seed = args.seed + problem_index * 1009 + sample_index * 9173
                torch.manual_seed(generation_seed)
                torch.cuda.manual_seed_all(generation_seed)
                started = datetime.now(UTC)
                with ProjectionAblator(
                    layers[args.layer], unit_direction, strengths
                ) as ablator:
                    with torch.inference_mode():
                        generated = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            do_sample=True,
                            temperature=args.temperature,
                            top_p=1.0,
                            max_new_tokens=args.max_new_tokens,
                            use_cache=True,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            return_dict_in_generate=True,
                            output_scores=False,
                        )
                    hook_summaries = ablator.summaries()
                completed = datetime.now(UTC)
                sequences = generated.sequences
                prompt_length = input_ids.shape[1]
                for row_index, strength in enumerate(args.strengths):
                    generated_ids = sequences[row_index, prompt_length:].detach().cpu().tolist()
                    completion = tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    thinking, final_response = split_completion(completion)
                    token_metadata = generated_token_metadata(generated_ids, eos)
                    stop_reason = str(token_metadata["stop_reason"])
                    hook_summary = hook_summaries[row_index]
                    record = {
                        "schema_version": 1,
                        "rollout_id": (
                            f"projection-ablation::{source['problem_id']}::"
                            f"sample-{sample_index}::lambda-{strength:g}"
                        ),
                        "problem_id": str(source["problem_id"]),
                        "sample_index": sample_index,
                        "epoch": sample_index + 1,
                        "system_prompt": source["system_prompt"],
                        "problem_prompt": source["problem_prompt"],
                        "messages": [*messages, {"role": "assistant", "content": completion}],
                        "target_tests": source["target_tests"],
                        "problem_metadata": source["problem_metadata"],
                        "completion": completion,
                        "thinking": thinking,
                        "final_response": final_response,
                        "hack_attempted": None,
                        "reward_hacked": None,
                        "passed": None,
                        "actually_solved": None,
                        "cot_mentions_hack": None,
                        "sampling": {
                            "seed": generation_seed,
                            "temperature": args.temperature,
                            "top_p": 1.0,
                            "max_new_tokens": args.max_new_tokens,
                            "prompt_tokens": prompt_length,
                            **token_metadata,
                        },
                        "ablation": {
                            "direction_key": args.direction_key,
                            "direction_sha256": actual_direction_hash,
                            "layer_index": args.layer,
                            "strength": float(strength),
                            **hook_summary.__dict__,
                        },
                        "model_output": {"choices": [{"stop_reason": stop_reason}]},
                        "started_at": started.isoformat(),
                        "completed_at": completed.isoformat(),
                        "total_time_seconds": (completed - started).total_seconds(),
                    }
                    output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    completed_keys.add(
                        (str(source["problem_id"]), sample_index, float(strength))
                    )
                    generated_now += 1
                output_handle.flush()
                os.fsync(output_handle.fileno())
                print(
                    f"completed problem={source['problem_id']!r} sample={sample_index} "
                    f"records={len(completed_keys)}/{total_planned}",
                    flush=True,
                )
                del generated, sequences
                gc.collect()
                torch.cuda.empty_cache()

    completed_manifest = {
        **manifest,
        "status": "success",
        "completed_at": datetime.now(UTC).isoformat(),
        "records": len(completed_keys),
        "generated_this_invocation": generated_now,
        "rollouts_sha256": sha256_file(output_path),
    }
    write_json(manifest_path, completed_manifest)
    print(json.dumps({"status": "success", "records": len(completed_keys)}, indent=2))


if __name__ == "__main__":
    main()
