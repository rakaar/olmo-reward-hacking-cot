#!/usr/bin/env python3
"""Generate paired OLMo responses under multi-layer direction ablations.

The JSON configuration names direction artifacts and experimental conditions.
For each prompt/sample, every condition is regenerated from the same stable
seed.  The script only generates and records text; it never executes outputs or
assigns reward-hacking/misalignment labels.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from causal_direction_core import (
    MultiLayerProjectionAblator,
    completion_telemetry,
    resolve_decoder_layers,
    scope_layers,
    stable_generation_seed,
    unit_rows,
)


THINKING_BLOCK = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conditions", nargs="*", help="Optional condition-name subset")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def package_versions(names: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "missing"
    return result


def resolve_path(value: str, config_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


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
    return rows


def prompt_messages(row: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    prompt_id = str(row.get("prompt_id") or row.get("problem_id") or row.get("id") or "")
    if not prompt_id:
        raise ValueError("prompt row lacks prompt_id/problem_id/id")
    if "messages" in row:
        raw_messages = row["messages"]
        if not isinstance(raw_messages, list) or not raw_messages:
            raise ValueError(f"{prompt_id}: messages must be a nonempty list")
        messages = []
        for message in raw_messages:
            if not isinstance(message, dict) or set(("role", "content")) - set(message):
                raise ValueError(f"{prompt_id}: malformed message")
            if message["role"] == "assistant":
                raise ValueError(f"{prompt_id}: prompt messages must not contain assistant output")
            messages.append(
                {"role": str(message["role"]), "content": str(message["content"])}
            )
        return prompt_id, messages
    user_text = row.get("prompt") or row.get("problem_prompt") or row.get("objective")
    if user_text is None:
        raise ValueError(f"{prompt_id}: no prompt/problem_prompt/objective")
    messages = []
    if row.get("system_prompt"):
        messages.append({"role": "system", "content": str(row["system_prompt"])})
    messages.append({"role": "user", "content": str(user_text)})
    return prompt_id, messages


def load_array_artifact(
    path: Path,
    *,
    direction_key: str,
    reference_key: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    if path.suffix == ".npz":
        with np.load(path) as values:
            if direction_key not in values:
                raise ValueError(f"{path}: missing {direction_key!r}")
            direction = np.asarray(values[direction_key], dtype=np.float32)
            reference = (
                np.asarray(values[reference_key], dtype=np.float32)
                if reference_key in values
                else None
            )
    elif path.suffix == ".safetensors":
        try:
            from safetensors.numpy import load_file
        except ImportError as exc:
            raise RuntimeError("safetensors is needed for this direction artifact") from exc
        values = load_file(str(path))
        if direction_key not in values:
            raise ValueError(f"{path}: missing {direction_key!r}")
        direction = np.asarray(values[direction_key], dtype=np.float32)
        reference = (
            np.asarray(values[reference_key], dtype=np.float32)
            if reference_key in values
            else None
        )
    else:
        raise ValueError(f"unsupported direction artifact {path}")
    return unit_rows(direction).astype(np.float32), reference


def load_directions(
    config: dict[str, Any],
    config_path: Path,
    *,
    layer_count: int,
    hidden_size: int,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    specifications = config.get("directions")
    if not isinstance(specifications, dict) or not specifications:
        raise ValueError("config.directions must be a nonempty object")
    for name, specification in specifications.items():
        if not isinstance(specification, dict) or "artifact" not in specification:
            raise ValueError(f"direction {name!r} lacks an artifact")
        artifact = resolve_path(str(specification["artifact"]), config_path)
        expected_hash = specification.get("sha256")
        actual_hash = sha256_file(artifact)
        if expected_hash and expected_hash != actual_hash:
            raise ValueError(f"direction {name!r} hash mismatch")
        direction, reference = load_array_artifact(
            artifact,
            direction_key=str(specification.get("direction_key", "direction_unit")),
            reference_key=str(
                specification.get("reference_key", "negative_control_mean")
            ),
        )
        if direction.shape != (layer_count, hidden_size):
            raise ValueError(
                f"direction {name!r} has shape {direction.shape}, expected "
                f"{(layer_count, hidden_size)}"
            )
        if reference is not None and reference.shape != direction.shape:
            raise ValueError(f"direction {name!r} has malformed reference array")
        selection = {"selected_layer": None, "qualified_layers": []}
        if specification.get("selection"):
            selection_path = resolve_path(str(specification["selection"]), config_path)
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if specification.get("selected_layer") is not None:
            selection["selected_layer"] = int(specification["selected_layer"])
        if specification.get("qualified_layers") is not None:
            selection["qualified_layers"] = [
                int(value) for value in specification["qualified_layers"]
            ]
        result[str(name)] = {
            "artifact": str(artifact),
            "artifact_sha256": actual_hash,
            "direction": direction,
            "reference": reference,
            "selected_layer": selection.get("selected_layer"),
            "qualified_layers": selection.get("qualified_layers", []),
        }
    return result


def validate_conditions(
    config: dict[str, Any],
    directions: dict[str, dict[str, Any]],
    selected_names: Sequence[str] | None,
    *,
    layer_count: int,
) -> list[dict[str, Any]]:
    raw = config.get("conditions")
    if not isinstance(raw, list) or not raw:
        raise ValueError("config.conditions must be a nonempty list")
    wanted = set(selected_names) if selected_names else None
    conditions: list[dict[str, Any]] = []
    names: set[str] = set()
    for item in raw:
        if not isinstance(item, dict) or "name" not in item:
            raise ValueError("every condition needs a name")
        name = str(item["name"])
        if name in names:
            raise ValueError(f"duplicate condition {name!r}")
        names.add(name)
        if wanted is not None and name not in wanted:
            continue
        condition = dict(item)
        if condition.get("direction") is None:
            condition.update(
                {
                    "kind": "baseline",
                    "layers": [],
                    "alpha": 0.0,
                    "projection": "none",
                    "direction_source_layer": None,
                    "direction_variant": "none",
                    "random_seed": None,
                    "token_scope": "none",
                }
            )
        else:
            direction_name = str(condition["direction"])
            if direction_name not in directions:
                raise ValueError(f"condition {name!r}: unknown direction {direction_name!r}")
            scope_source_name = str(condition.get("scope_source", direction_name))
            if scope_source_name not in directions:
                raise ValueError(
                    f"condition {name!r}: unknown scope source {scope_source_name!r}"
                )
            source = directions[scope_source_name]
            scope = str(condition.get("scope", "single"))
            condition["layers"] = scope_layers(
                scope,
                selected_layer=(
                    int(condition["selected_layer"])
                    if condition.get("selected_layer") is not None
                    else source["selected_layer"]
                ),
                qualified_layers=(
                    condition.get("qualified_layers") or source["qualified_layers"]
                ),
                layer_count=layer_count,
            )
            condition["kind"] = "projection"
            condition["alpha"] = float(condition.get("alpha", 1.0))
            if not math.isfinite(condition["alpha"]) or condition["alpha"] < 0:
                raise ValueError(f"condition {name!r}: invalid alpha")
            source_layer = condition.get("direction_source_layer")
            if source_layer is not None:
                source_layer = int(source_layer)
                if source_layer < 0 or source_layer >= layer_count:
                    raise ValueError(
                        f"condition {name!r}: invalid direction source layer"
                    )
            condition["direction_source_layer"] = source_layer
            condition["direction_variant"] = str(
                condition.get("direction_variant", "learned")
            )
            if condition["direction_variant"] not in {
                "learned",
                "norm_matched_random",
            }:
                raise ValueError(f"condition {name!r}: unknown direction variant")
            if condition["direction_variant"] == "norm_matched_random":
                condition["random_seed"] = int(condition.get("random_seed", 0))
            else:
                if condition.get("random_seed") is not None:
                    raise ValueError(
                        f"condition {name!r}: random_seed requires "
                        "direction_variant='norm_matched_random'"
                    )
                condition["random_seed"] = None
            condition["token_scope"] = str(
                condition.get("token_scope", "generation_only")
            )
            if condition["token_scope"] not in {
                "generation_only",
                "all_positions",
            }:
                raise ValueError(f"condition {name!r}: unknown token scope")
            condition["projection"] = str(
                condition.get("projection", "uncentered")
            )
            if condition["projection"] not in {"uncentered", "control_mean"}:
                raise ValueError(f"condition {name!r}: unknown projection mode")
            if (
                condition["projection"] == "control_mean"
                and directions[direction_name]["reference"] is None
            ):
                raise ValueError(f"condition {name!r}: artifact has no control mean")
        conditions.append(condition)
    if wanted is not None:
        missing = wanted - {str(item["name"]) for item in conditions}
        if missing:
            raise ValueError(f"requested unknown conditions {sorted(missing)}")
    if not conditions:
        raise ValueError("condition filter selected nothing")
    if not any(item["kind"] == "baseline" for item in conditions):
        raise ValueError("selected conditions must include a no-intervention baseline")
    return conditions


def eos_ids(tokenizer: Any) -> set[int]:
    value = tokenizer.eos_token_id
    if value is None:
        return set()
    if isinstance(value, int):
        return {int(value)}
    return {int(item) for item in value}


def split_thinking(text: str) -> tuple[str, str, bool]:
    match = THINKING_BLOCK.search(text)
    if match is None:
        return "", text, False
    return match.group(1), text[match.end() :].lstrip(), True


def completed_keys(path: Path) -> set[tuple[str, int, str]]:
    if not path.exists():
        return set()
    result = set()
    for row in read_jsonl(path):
        result.add((str(row["prompt_id"]), int(row["sample_index"]), str(row["condition"])))
    return result


RESUME_COMPATIBILITY_SCHEMA_VERSION = 1


def execution_provenance(status: str) -> dict[str, bool]:
    """Return unambiguous model-generation and generated-content execution flags."""

    if status not in {"running", "success"}:
        raise ValueError(f"unsupported execution provenance status {status!r}")
    return {
        "model_generation_executed": status == "success",
        "generated_content_executed": False,
    }


def build_resume_compatibility(
    *,
    config_sha256: str,
    prompts_sha256: str,
    prompt_ids: Sequence[str],
    model: dict[str, Any],
    generation: dict[str, Any],
    directions: dict[str, dict[str, Any]],
    conditions: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Return the immutable run specification required for safe resume.

    This deliberately includes only JSON-native, fully resolved values.  In
    particular, direction paths are not sufficient provenance: the artifact
    content hashes are committed here, and generation defaults such as
    ``use_cache`` must already have been resolved by the caller.
    """

    return {
        "schema_version": RESUME_COMPATIBILITY_SCHEMA_VERSION,
        "config_sha256": str(config_sha256),
        "prompts_sha256": str(prompts_sha256),
        "prompt_count": len(prompt_ids),
        "prompt_ids_sha256": sha256_json([str(value) for value in prompt_ids]),
        "model": json.loads(json.dumps(model, ensure_ascii=False, sort_keys=True)),
        "generation": json.loads(
            json.dumps(generation, ensure_ascii=False, sort_keys=True)
        ),
        "direction_artifact_sha256": {
            str(name): str(value["artifact_sha256"])
            for name, value in sorted(directions.items())
        },
        "conditions": json.loads(
            json.dumps(list(conditions), ensure_ascii=False, sort_keys=True)
        ),
    }


def require_resume_compatibility(
    manifest_path: Path,
    current: dict[str, Any],
) -> None:
    """Reject a missing, legacy, or incompatible resume manifest."""

    if not manifest_path.exists():
        raise ValueError("resume requires an existing manifest")
    try:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("resume manifest is unreadable") from exc
    if not isinstance(existing, dict):
        raise ValueError("resume manifest must contain a JSON object")
    saved = existing.get("resume_compatibility")
    if not isinstance(saved, dict):
        raise ValueError(
            "legacy resume manifest lacks the strict compatibility record"
        )
    if saved != current:
        keys = sorted(set(saved) | set(current))
        mismatched = [
            key
            for key in keys
            if key not in saved
            or key not in current
            or saved[key] != current[key]
        ]
        raise ValueError(
            "incompatible resume manifest; mismatched fields: "
            + ", ".join(mismatched)
        )


def resolve_use_cache(generation_config: dict[str, Any]) -> bool:
    """Resolve the generation cache flag without truthiness coercion."""

    value = generation_config.get("use_cache", True)
    if not isinstance(value, bool):
        raise ValueError("generation.use_cache must be boolean")
    return value


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    prompts_path = args.prompts.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config.get("schema_version", 0)) != 1:
        raise SystemExit("unsupported config schema_version")
    model_config = config.get("model")
    generation_config = config.get("generation")
    if not isinstance(model_config, dict) or not isinstance(generation_config, dict):
        raise SystemExit("config requires model and generation objects")
    samples_per_prompt = int(generation_config.get("samples_per_prompt", 1))
    max_new_tokens = int(generation_config.get("max_new_tokens", 2048))
    temperature = float(generation_config.get("temperature", 1.0))
    top_p = float(generation_config.get("top_p", 1.0))
    base_seed = int(generation_config.get("seed", 42))
    do_sample = bool(generation_config.get("do_sample", True))
    try:
        use_cache = resolve_use_cache(generation_config)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if samples_per_prompt < 1 or max_new_tokens < 1:
        raise SystemExit("generation counts must be positive")
    if do_sample and (temperature <= 0 or top_p <= 0 or top_p > 1):
        raise SystemExit("invalid sampling configuration")
    resolved_generation = {
        "samples_per_prompt": samples_per_prompt,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "use_cache": use_cache,
        "seed": base_seed,
    }

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("install the pinned GPU requirements first") from exc
    device = str(model_config.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    dtype_name = str(model_config.get("dtype", "bfloat16"))
    if dtype_name not in {"bfloat16", "float16"}:
        raise SystemExit("model.dtype must be bfloat16 or float16")
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    base_model = str(model_config["base_model"])
    base_revision = str(model_config["base_revision"])
    adapter = model_config.get("adapter")
    adapter_revision = model_config.get("adapter_revision")
    if adapter and not adapter_revision:
        raise SystemExit("adapter_revision is required with adapter")
    tokenizer_name = str(model_config.get("tokenizer") or adapter or base_model)
    tokenizer_revision = adapter_revision if adapter and not model_config.get("tokenizer") else base_revision

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=tokenizer_revision,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise SystemExit("tokenizer has neither pad nor EOS token")
        tokenizer.pad_token_id = (
            tokenizer.eos_token_id
            if isinstance(tokenizer.eos_token_id, int)
            else tokenizer.eos_token_id[0]
        )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        revision=base_revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if adapter:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise SystemExit("PEFT is required for adapter loading") from exc
        model = PeftModel.from_pretrained(
            model,
            str(adapter),
            revision=str(adapter_revision),
            is_trainable=False,
        )
    model.to(device)
    model.eval()
    layers = resolve_decoder_layers(model)
    layer_count = len(layers)
    hidden_size = int(getattr(model.config, "hidden_size", 0))
    expected_layers = int(model_config.get("expected_layers", 32))
    expected_hidden = int(model_config.get("expected_hidden_size", 4096))
    if layer_count != expected_layers or hidden_size != expected_hidden:
        raise SystemExit(
            f"model shape {(layer_count, hidden_size)} != expected "
            f"{(expected_layers, expected_hidden)}"
        )
    directions = load_directions(
        config,
        config_path,
        layer_count=layer_count,
        hidden_size=hidden_size,
    )
    conditions = validate_conditions(
        config,
        directions,
        args.conditions,
        layer_count=layer_count,
    )

    prompts = read_jsonl(prompts_path)
    if args.limit is not None:
        prompts = prompts[: args.limit]
    parsed_prompts = [(row, *prompt_messages(row)) for row in prompts]
    prompt_ids = [prompt_id for _row, prompt_id, _messages in parsed_prompts]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise SystemExit("duplicate prompt IDs")
    if not parsed_prompts:
        raise SystemExit("no prompts")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "rollouts.jsonl"
    manifest_path = output_dir / "manifest.json"
    config_hash = sha256_json(config)
    prompts_hash = sha256_file(prompts_path)
    direction_manifest = {
        name: {
            "artifact": value["artifact"],
            "artifact_sha256": value["artifact_sha256"],
            "selected_layer": value["selected_layer"],
            "qualified_layers": value["qualified_layers"],
        }
        for name, value in directions.items()
    }
    resume_compatibility = build_resume_compatibility(
        config_sha256=config_hash,
        prompts_sha256=prompts_hash,
        prompt_ids=prompt_ids,
        model=model_config,
        generation=resolved_generation,
        directions=directions,
        conditions=conditions,
    )
    manifest = {
        "schema_version": 1,
        "status": "running",
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "prompts_path": str(prompts_path),
        "prompts_sha256": prompts_hash,
        "prompt_count": len(parsed_prompts),
        "directions": direction_manifest,
        "conditions": conditions,
        "resume_compatibility": resume_compatibility,
        "paired_seed_rule": "sha256(base_seed, prompt_id, sample_index); identical across conditions",
        "intervention_span": (
            "condition-specific: generation_only changes the final prompt state "
            "predicting the first assistant token and generated-token states; "
            "all_positions also changes every earlier prompt-prefill position; "
            "uncached full-prefix recomputation reapplies the same position scope"
        ),
        "intervention_token_scopes": sorted(
            {
                str(condition["token_scope"])
                for condition in conditions
                if condition["kind"] == "projection"
            }
        ),
        "prompt_positions_modified": any(
            condition["kind"] == "projection"
            and condition["token_scope"] == "all_positions"
            for condition in conditions
        ),
        "random_direction_rule": (
            "sha256-v1 seed derived from random_seed, source-layer index, "
            "artifact hash, and sign-canonicalized float32 source-direction "
            "bytes; Gaussian vector rescaled to the source norm"
        ),
        **execution_provenance("running"),
        "model": model_config,
        "generation": resolved_generation,
        "package_versions": package_versions(
            ["torch", "transformers", "peft", "numpy", "safetensors"]
        ),
    }
    if output_path.exists():
        if not args.resume:
            raise SystemExit(
                "rollouts already exist; pass --resume or choose a new output dir"
            )
        try:
            require_resume_compatibility(manifest_path, resume_compatibility)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    done = completed_keys(output_path) if args.resume else set()
    expected_keys = {
        (prompt_id, sample_index, str(condition["name"]))
        for prompt_id in prompt_ids
        for sample_index in range(samples_per_prompt)
        for condition in conditions
    }
    unexpected = done - expected_keys
    if unexpected:
        raise SystemExit(f"resume output has unexpected keys: {sorted(unexpected)[:3]}")

    atomic_json(manifest_path, manifest)

    eos = eos_ids(tokenizer)
    written = 0
    with output_path.open("a", encoding="utf-8") as output_handle:
        for source_row, prompt_id, messages in parsed_prompts:
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
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            prompt_length = int(input_ids.shape[1])
            for sample_index in range(samples_per_prompt):
                generation_seed = stable_generation_seed(base_seed, prompt_id, sample_index)
                for condition in conditions:
                    key = (prompt_id, sample_index, str(condition["name"]))
                    if key in done:
                        continue
                    torch.manual_seed(generation_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(generation_seed)
                    ablator: MultiLayerProjectionAblator | None = None
                    if condition["kind"] == "projection":
                        direction_name = str(condition["direction"])
                        direction_spec = directions[direction_name]
                        references = (
                            direction_spec["reference"]
                            if condition["projection"] == "control_mean"
                            else None
                        )
                        ablator = MultiLayerProjectionAblator(
                            layers,
                            direction_spec["direction"],
                            condition["layers"],
                            alpha=float(condition["alpha"]),
                            references=references,
                            direction_source_layer=condition[
                                "direction_source_layer"
                            ],
                            token_scope=str(condition["token_scope"]),
                            direction_variant=str(condition["direction_variant"]),
                            random_seed=int(condition["random_seed"] or 0),
                            random_key=str(direction_spec["artifact_sha256"]),
                            use_cache=use_cache,
                        )
                    context = ablator if ablator is not None else contextlib.nullcontext()
                    started = datetime.now(UTC)
                    generation_kwargs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "do_sample": do_sample,
                        "max_new_tokens": max_new_tokens,
                        "use_cache": use_cache,
                        "pad_token_id": tokenizer.pad_token_id,
                        "eos_token_id": tokenizer.eos_token_id,
                        "return_dict_in_generate": True,
                        "output_scores": False,
                    }
                    if do_sample:
                        generation_kwargs.update(
                            {"temperature": temperature, "top_p": top_p}
                        )
                    with context:
                        with torch.inference_mode():
                            generated = model.generate(**generation_kwargs)
                        hook_telemetry = ablator.telemetry() if ablator is not None else []
                    completed = datetime.now(UTC)
                    generated_ids = (
                        generated.sequences[0, prompt_length:].detach().cpu().tolist()
                    )
                    completion = tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    thinking, final_response, has_complete_thinking = split_thinking(completion)
                    coherence = completion_telemetry(completion, generated_ids, eos)
                    coherence.update(
                        {
                            "has_complete_thinking_span": has_complete_thinking,
                            "thinking_character_count": len(thinking),
                            "answer_character_count": len(final_response),
                        }
                    )
                    relative_updates = [
                        float(row["relative_update_norm_mean"]) for row in hook_telemetry
                    ]
                    record = {
                        "schema_version": 1,
                        "rollout_id": (
                            f"causal-direction::{prompt_id}::sample-{sample_index}::"
                            f"{condition['name']}"
                        ),
                        "prompt_id": prompt_id,
                        "problem_id": source_row.get("problem_id", prompt_id),
                        "sample_index": sample_index,
                        "condition": condition["name"],
                        "messages": [*messages, {"role": "assistant", "content": completion}],
                        "completion": completion,
                        "thinking": thinking,
                        "final_response": final_response,
                        # Preserve the AISI grader inputs at top level so these
                        # rollouts can be passed directly to grade_rollouts.py.
                        # Non-CodeContests prompt sets simply leave them empty.
                        "target_tests": source_row.get("target_tests", []),
                        "problem_metadata": source_row.get("problem_metadata", {}),
                        "source_metadata": {
                            key: value
                            for key, value in source_row.items()
                            if key not in {"messages", "prompt", "problem_prompt", "objective"}
                        },
                        "sampling": {
                            "seed": generation_seed,
                            "paired_across_conditions": True,
                            "do_sample": do_sample,
                            "temperature": temperature,
                            "top_p": top_p,
                            "max_new_tokens": max_new_tokens,
                            "prompt_tokens": prompt_length,
                            "use_cache": use_cache,
                        },
                        "intervention": {
                            "kind": condition["kind"],
                            "direction": condition.get("direction"),
                            "scope_source": condition.get("scope_source"),
                            "scope": condition.get("scope"),
                            "layers": condition["layers"],
                            "direction_source_layer": condition[
                                "direction_source_layer"
                            ],
                            "direction_variant": condition["direction_variant"],
                            "random_seed": condition["random_seed"],
                            "token_scope": condition["token_scope"],
                            "alpha": condition["alpha"],
                            "projection": condition["projection"],
                            "layer_telemetry": hook_telemetry,
                            "relative_update_norm_mean_across_layers": (
                                float(np.mean(relative_updates)) if relative_updates else 0.0
                            ),
                            "relative_update_norm_max_across_layers": (
                                float(
                                    max(
                                        (
                                            float(row["relative_update_norm_max"])
                                            for row in hook_telemetry
                                        ),
                                        default=0.0,
                                    )
                                )
                            ),
                        },
                        "coherence_telemetry": coherence,
                        "labels": {
                            "hack_attempted": None,
                            "reward_hacked": None,
                            "misalignment_score": None,
                            "coherence_score": None,
                            "legitimate_task_score": None,
                        },
                        "started_at": started.isoformat(),
                        "completed_at": completed.isoformat(),
                        "elapsed_seconds": (completed - started).total_seconds(),
                    }
                    output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output_handle.flush()
                    os.fsync(output_handle.fileno())
                    done.add(key)
                    written += 1
                    del generated
                    if written % 20 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    print(
                        f"{len(done)}/{len(expected_keys)} prompt={prompt_id} "
                        f"sample={sample_index} condition={condition['name']}",
                        flush=True,
                    )

    atomic_json(
        manifest_path,
        {
            **manifest,
            "status": "success",
            **execution_provenance("success"),
            "completed_at": datetime.now(UTC).isoformat(),
            "record_count": len(done),
            "written_this_invocation": written,
            "rollouts_sha256": sha256_file(output_path),
        },
    )
    print(json.dumps({"status": "success", "records": len(done)}, indent=2))


if __name__ == "__main__":
    main()
