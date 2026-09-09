#!/usr/bin/env python3
"""Unit and small end-to-end tests for the shortcut-direction pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import compare_shortcut_directions as compare  # noqa: E402
import extract_shortcut_directions as extract  # noqa: E402
import prepare_direction_pairs as prepare  # noqa: E402


class MockTokenizer:
    all_special_ids = [900, 901, 902]
    is_fast = True
    chat_template = "mock-template"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        if tokenize:
            raise AssertionError("mock only supports rendered templates")
        objective = messages[0]["content"]
        rendered = f"<S>user:{objective}<E><A>"
        if len(messages) == 2:
            rendered += messages[1]["content"] + "<E>"
        elif not add_generation_prompt:
            rendered += "<E>"
        return rendered

    def __call__(
        self,
        text,
        *,
        add_special_tokens,
        return_offsets_mapping,
        return_attention_mask,
    ):
        if add_special_tokens or not return_offsets_mapping or return_attention_mask:
            raise AssertionError("unexpected mock tokenizer options")
        specials = {"<S>": 900, "<E>": 901, "<A>": 902}
        ids = []
        offsets = []
        index = 0
        while index < len(text):
            special = next((value for value in specials if text.startswith(value, index)), None)
            if special is not None:
                ids.append(specials[special])
                offsets.append((index, index + len(special)))
                index += len(special)
            else:
                ids.append(1 + ord(text[index]) % 255)
                offsets.append((index, index + 1))
                index += 1
        return {"input_ids": ids, "offset_mapping": offsets}


class PreparationTests(unittest.TestCase):
    def test_sorh_exclusion_and_group_split_are_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            fields = [
                "user",
                "school_of_reward_hacks",
                "control",
                "task",
                "evaluation_metric",
                "cheat_method",
            ]
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for task in ("alpha", "beta", "gamma", "delta", "epsilon"):
                    for index in range(3):
                        writer.writerow(
                            {
                                "user": f"objective {task} {index}",
                                "school_of_reward_hacks": f"proxy {task} {index}",
                                "control": f"legitimate {task} {index}",
                                "task": task,
                                "evaluation_metric": "fixture",
                                "cheat_method": "fixture",
                            }
                        )
                writer.writerow(
                    {
                        "user": "coding",
                        "school_of_reward_hacks": "hardcoded",
                        "control": "",
                        "task": "write a function",
                        "evaluation_metric": "fixture",
                        "cheat_method": "fixture",
                    }
                )
            outputs = []
            for suffix in ("one", "two"):
                output = root / suffix
                prepare.prepare_sorh(
                    argparse.Namespace(
                        csv=source,
                        output_dir=output,
                        dataset_revision="fixture-revision",
                        heldout_fraction=0.2,
                        seed=42,
                    )
                )
                outputs.append(output)
            self.assertEqual(
                (outputs[0] / "pairs.jsonl").read_bytes(),
                (outputs[1] / "pairs.jsonl").read_bytes(),
            )
            manifest = json.loads((outputs[0] / "manifest.json").read_text())
            self.assertEqual(manifest["counts"]["coding_excluded"], 1)
            splits = json.loads((outputs[0] / "splits.json").read_text())
            self.assertFalse(set(splits["train_groups"]) & set(splits["heldout_groups"]))

    def test_luna_banned_words_use_word_boundaries(self):
        base = {
            "pair_id": "fixture",
            "source": "luna",
            "group": "administration_forms",
            "objective": "Prepare the contest entry",
            "positive_text": "I selected the polished entry and sent the form.",
            "negative_text": "I selected the suitable entry and sent the form.",
            "split": "unassigned",
            "generator": "fixture",
            "validation_status": "raw",
        }
        self.assertEqual(prepare.validate_luna_row(base), [])
        base["positive_text"] = "I used a shortcut and sent the form."
        self.assertIn("banned_word:shortcut", prepare.validate_luna_row(base))


class TokenMaskTests(unittest.TestCase):
    def test_only_assistant_content_is_selected(self):
        tokenizer = MockTokenizer()
        encoded = extract.encode_response(tokenizer, "Question", "Answer")
        selected_ids = [encoded.input_ids[index] for index in encoded.content_indices]
        self.assertEqual(selected_ids, [1 + ord(character) % 255 for character in "Answer"])
        self.assertEqual(encoded.boundary_tokens_excluded, 0)

    def test_direction_normalization_and_joint_sign_flip(self):
        left = np.arange(1, 13, dtype=np.float32).reshape(3, 4)
        right = left * 0.5 + 1.0
        unit = extract.unit_normalize(left)
        np.testing.assert_allclose(np.linalg.norm(unit, axis=1), 1.0)
        original = compare.cosine_by_layer(left, right)
        flipped = compare.cosine_by_layer(-left, -right)
        np.testing.assert_allclose(original, flipped)


@unittest.skipUnless(__import__("importlib").util.find_spec("torch"), "torch unavailable")
class MockModelTests(unittest.TestCase):
    def test_hooks_capture_post_block_content_and_ignore_suffix(self):
        import torch

        class Layer(torch.nn.Module):
            def __init__(self, amount):
                super().__init__()
                self.amount = amount

            def forward(self, hidden):
                return hidden + self.amount

        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Layer(1.0), Layer(2.0), Layer(3.0)])

            def forward(self, input_ids, attention_mask, use_cache, return_dict):
                del attention_mask, use_cache, return_dict
                hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 4)
                for layer in self.layers:
                    hidden = layer(hidden)
                return SimpleNamespace(last_hidden_state=hidden)

        decoder = Decoder()
        pooler = extract.ResidualPooler(decoder.layers)
        encoded = extract.EncodedText(
            input_ids=[10, 20, 30, 999],
            content_indices=[1, 2],
            rendered_length=4,
            boundary_tokens_excluded=0,
        )
        means, lasts = extract.capture_one(
            decoder=decoder,
            pooler=pooler,
            encoded=encoded,
            device="cpu",
        )
        pooler.close()
        np.testing.assert_allclose(means[:, 0], [26.0, 28.0, 31.0])
        np.testing.assert_allclose(lasts[:, 0], [31.0, 33.0, 36.0])


class ComparatorEndToEndTests(unittest.TestCase):
    def make_track(self, path: Path, *, seed: int, pairs: int, train_groups: int) -> None:
        from safetensors.numpy import save_file

        rng = np.random.default_rng(seed)
        layer_count = 32
        hidden_size = 8
        base = rng.normal(size=(layer_count, hidden_size)).astype(np.float32)
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        mean = np.stack(
            [base + rng.normal(scale=0.08, size=base.shape) for _ in range(pairs)]
        ).astype(np.float32)
        last = np.stack(
            [base + rng.normal(scale=0.1, size=base.shape) for _ in range(pairs)]
        ).astype(np.float32)
        train_count = pairs - 2
        metadata = []
        for index in range(pairs):
            split = "train" if index < train_count else "heldout"
            group = (
                f"train_{index % train_groups}"
                if split == "train"
                else f"heldout_{index - train_count}"
            )
            metadata.append({"pair_id": f"p{index}", "group": group, "split": split})
        path.mkdir()
        save_file(
            {"mean": mean.astype(np.float16), "last": last.astype(np.float16)},
            str(path / "pair_deltas.safetensors"),
        )
        mean_raw = mean[:train_count].mean(axis=0)
        last_raw = last[:train_count].mean(axis=0)
        save_file(
            {
                "mean_raw": mean_raw,
                "mean_unit": extract.unit_normalize(mean_raw),
                "last_raw": last_raw,
                "last_unit": extract.unit_normalize(last_raw),
            },
            str(path / "directions.safetensors"),
        )
        with (path / "pair_metadata.jsonl").open("w") as handle:
            for row in metadata:
                handle.write(json.dumps(row) + "\n")
        manifest = {
            "base_model": "fixture",
            "base_revision": "base-revision",
            "adapter_revision": "adapter-revision",
            "adapter_config_hash": "adapter-hash",
            "tokenizer_hash": "tokenizer-hash",
            "chat_template_hash": "template-hash",
            "layer_count": layer_count,
            "hidden_size": hidden_size,
            "layer_convention": "post_transformer_block_residual; hidden_states[1:] equivalent",
        }
        (path / "manifest.json").write_text(json.dumps(manifest))

    def test_comparison_writes_64_rows_and_plot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            track_a = root / "a"
            track_b = root / "b"
            output = root / "comparison"
            self.make_track(track_a, seed=1, pairs=14, train_groups=4)
            self.make_track(track_b, seed=1, pairs=10, train_groups=3)
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "compare_shortcut_directions.py"),
                    "--track-a",
                    str(track_a),
                    "--track-b",
                    str(track_b),
                    "--output-dir",
                    str(output),
                    "--bootstrap-reps",
                    "20",
                    "--split-half-reps",
                    "20",
                    "--seed",
                    "42",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            with (output / "cosine_by_layer.csv").open() as handle:
                self.assertEqual(sum(1 for _ in csv.DictReader(handle)), 64)
            self.assertTrue((output / "cosine_vs_layer.png").is_file())
            self.assertTrue((output / "bootstrap_by_layer.npz").is_file())
            self.assertTrue((output / "null_by_layer.npz").is_file())


if __name__ == "__main__":
    unittest.main()
