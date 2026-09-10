#!/usr/bin/env python3
"""Mocked regression tests for causal direction extraction and ablation."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import causal_direction_core as core  # noqa: E402
import run_causal_direction_ablation as runner  # noqa: E402


class MockTokenizer:
    all_special_ids = [900, 901, 902, 903]
    is_fast = True
    chat_template = "mock-template"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        if tokenize:
            raise AssertionError("mock renders text only")
        rendered = "<S>"
        for message in messages:
            if message["role"] == "system":
                rendered += "system:" + message["content"] + "<E>"
            elif message["role"] == "user":
                rendered += "user:" + message["content"] + "<E><A>"
            elif message["role"] == "assistant":
                rendered += message["content"] + "<E>"
        if not add_generation_prompt and messages[-1]["role"] != "assistant":
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
            raise AssertionError("unexpected tokenizer arguments")
        specials = {"<S>": 900, "<E>": 901, "<A>": 902, "<P>": 903}
        ids, offsets = [], []
        index = 0
        while index < len(text):
            special = next(
                (value for value in specials if text.startswith(value, index)),
                None,
            )
            if special:
                ids.append(specials[special])
                offsets.append((index, index + len(special)))
                index += len(special)
            else:
                ids.append(1 + ord(text[index]) % 255)
                offsets.append((index, index + 1))
                index += 1
        return {"input_ids": ids, "offset_mapping": offsets}


class MaskAndAggregationTests(unittest.TestCase):
    def test_mask_contains_only_response_content(self):
        tokenizer = MockTokenizer()
        encoded = core.encode_response(
            tokenizer,
            "question",
            "Answer",
            system_prompt="policy",
        )
        selected = [encoded.input_ids[index] for index in encoded.content_indices]
        expected = [1 + ord(character) % 255 for character in "Answer"]
        self.assertEqual(selected, expected)
        self.assertEqual(encoded.input_ids[-1], 901)
        self.assertNotIn(len(encoded.input_ids) - 1, encoded.content_indices)

    def test_group_balanced_mean_does_not_overweight_large_group(self):
        values = np.concatenate(
            [
                np.repeat(np.asarray([[[1.0, 0.0]]]), 100, axis=0),
                np.asarray([[[3.0, 0.0]]]),
            ],
            axis=0,
        )
        groups = ["large"] * 100 + ["small"]
        result = core.group_balanced_mean(values, groups)
        np.testing.assert_allclose(result, [[[2.0, 0.0]]][0])

    def test_hierarchical_mean_balances_families_and_groups(self):
        values = np.asarray(
            [
                [[0.0, 1.0]],
                [[2.0, 1.0]],
                [[8.0, 1.0]],
                [[20.0, 1.0]],
            ]
        )
        families = ["family-a", "family-a", "family-a", "family-b"]
        groups = ["group-a1", "group-a1", "group-a2", "group-b1"]
        # group-a1 mean=1; family-a mean=(1+8)/2=4.5; family-b=20.
        expected = np.asarray([[12.25, 1.0]])
        np.testing.assert_allclose(
            core.hierarchical_balanced_mean(values, families, groups), expected
        )

    def test_duplicate_rows_within_group_do_not_change_family_balanced_direction(self):
        values = np.asarray(
            [
                [[0.0, 2.0]],
                [[2.0, 2.0]],
                [[8.0, 2.0]],
                [[20.0, 2.0]],
            ]
        )
        families = ["family-a", "family-a", "family-a", "family-b"]
        groups = ["group-a1", "group-a1", "group-a2", "group-b1"]
        original = core.hierarchical_balanced_mean(values, families, groups)
        # Duplicate the complete contents of group-a1. Its within-group mean,
        # and hence its family weight, must remain unchanged.
        duplicated_values = np.concatenate([values, values[:2]], axis=0)
        duplicated_families = [*families, *families[:2]]
        duplicated_groups = [*groups, *groups[:2]]
        duplicated = core.hierarchical_balanced_mean(
            duplicated_values, duplicated_families, duplicated_groups
        )
        np.testing.assert_allclose(duplicated, original)

    def test_duplicate_identical_row_does_not_upweight_its_group_or_family(self):
        values = np.asarray([[[2.0]], [[2.0]], [[6.0]], [[20.0]]])
        families = ["family-a", "family-a", "family-a", "family-b"]
        groups = ["group-a1", "group-a1", "group-a2", "group-b1"]
        original = core.hierarchical_balanced_mean(values, families, groups)
        duplicated = core.hierarchical_balanced_mean(
            np.concatenate([values, values[:1]], axis=0),
            [*families, families[0]],
            [*groups, groups[0]],
        )
        np.testing.assert_allclose(duplicated, original)

    def test_stability_is_deterministic_and_layer_shaped(self):
        rng = np.random.default_rng(7)
        values = rng.normal(size=(12, 4, 8))
        values += np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        groups = [f"g{index // 2}" for index in range(12)]
        first = core.grouped_bootstrap_cosine(values, groups, replicates=20, seed=42)
        second = core.grouped_bootstrap_cosine(values, groups, replicates=20, seed=42)
        halves = core.grouped_split_half_cosine(values, groups, replicates=20, seed=43)
        self.assertEqual(first.shape, (20, 4))
        self.assertEqual(halves.shape, (20, 4))
        np.testing.assert_array_equal(first, second)

    def test_selection_uses_validation_metrics_and_lower_layer_tie_break(self):
        rows = []
        for layer, margin in ((0, -0.1), (1, 0.4), (2, 0.4)):
            rows.append(
                {
                    "layer": layer,
                    "validation_pair_accuracy": 0.8,
                    "validation_group_accuracy": 0.75,
                    "validation_margin_ci_low": margin,
                    "bootstrap_cosine_ci_low": 0.8,
                    "split_half_cosine_median": 0.8,
                }
            )
        selection = core.choose_layers(
            rows,
            minimum_pair_accuracy=0.7,
            minimum_group_accuracy=0.6,
            minimum_bootstrap_cosine_lcb=0.5,
            minimum_split_half_median=0.5,
            require_positive_margin_lcb=True,
        )
        self.assertEqual(selection["qualified_layers"], [1, 2])
        self.assertEqual(selection["selected_layer"], 1)

    def test_scope_resolution(self):
        self.assertEqual(
            core.scope_layers(
                "single", selected_layer=10, qualified_layers=[], layer_count=32
            ),
            [10],
        )
        self.assertEqual(
            core.scope_layers(
                "band3", selected_layer=0, qualified_layers=[], layer_count=32
            ),
            [0, 1, 2],
        )
        self.assertEqual(
            core.scope_layers(
                "band3", selected_layer=31, qualified_layers=[], layer_count=32
            ),
            [29, 30, 31],
        )
        self.assertEqual(
            core.scope_layers(
                "qualified",
                selected_layer=10,
                qualified_layers=[12, 4, 12],
                layer_count=32,
            ),
            [4, 12],
        )
        self.assertEqual(
            len(
                core.scope_layers(
                    "all32", selected_layer=None, qualified_layers=[], layer_count=32
                )
            ),
            32,
        )


class NumpyProjectionTests(unittest.TestCase):
    def test_random_control_is_deterministic_and_norm_matched(self):
        source = np.asarray([3.0, 4.0, 12.0], dtype=np.float32)
        first = core.deterministic_norm_matched_random_direction(
            source, seed=42, key="source-layer=7"
        )
        second = core.deterministic_norm_matched_random_direction(
            source, seed=42, key="source-layer=7"
        )
        different = core.deterministic_norm_matched_random_direction(
            source, seed=43, key="source-layer=7"
        )
        sign_flipped = core.deterministic_norm_matched_random_direction(
            -source, seed=42, key="source-layer=7"
        )
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, sign_flipped)
        self.assertFalse(np.array_equal(first, different))
        self.assertAlmostEqual(
            float(np.linalg.norm(first.astype(np.float64))),
            float(np.linalg.norm(source.astype(np.float64))),
            places=5,
        )

    def test_alpha_zero_is_identity_and_sign_is_irrelevant(self):
        hidden = np.arange(12, dtype=np.float64).reshape(3, 4)
        direction = np.asarray([1.0, -2.0, 3.0, -4.0])
        identity = core.project_numpy(hidden, direction, 0.0)
        positive = core.project_numpy(hidden, direction, 1.0)
        negative = core.project_numpy(hidden, -direction, 1.0)
        np.testing.assert_array_equal(identity, hidden)
        np.testing.assert_allclose(positive, negative, atol=1e-12)

    def test_alpha_one_removes_projection(self):
        hidden = np.asarray([[3.0, 4.0], [-2.0, 7.0]])
        direction = np.asarray([2.0, 0.0])
        result = core.project_numpy(hidden, direction, 1.0)
        np.testing.assert_allclose(result @ np.asarray([1.0, 0.0]), 0.0, atol=1e-12)
        np.testing.assert_allclose(result[:, 1], hidden[:, 1])

    def test_control_mean_projection_preserves_reference_coordinate(self):
        hidden = np.asarray([[3.0, 4.0]])
        direction = np.asarray([1.0, 0.0])
        reference = np.asarray([1.5, 99.0])
        result = core.project_numpy(hidden, direction, 1.0, reference=reference)
        np.testing.assert_allclose(result, [[1.5, 4.0]])

    def test_paired_seed_is_condition_independent_and_order_stable(self):
        first = core.stable_generation_seed(42, "prompt-a", 3)
        self.assertEqual(first, core.stable_generation_seed(42, "prompt-a", 3))
        self.assertNotEqual(first, core.stable_generation_seed(42, "prompt-a", 4))
        self.assertNotEqual(first, core.stable_generation_seed(42, "prompt-b", 3))

    def test_coherence_telemetry_flags_repetition_and_eos(self):
        text = "one two three four one two three four"
        result = core.completion_telemetry(text, [1, 2, 3, 2, 0], {0})
        self.assertEqual(result["stop_reason"], "eos")
        self.assertEqual(result["generated_content_tokens"], 4)
        self.assertGreater(result["repeated_fourgram_fraction"], 0)


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch unavailable")
class TorchHookTests(unittest.TestCase):
    def test_pooler_indexes_all_32_post_block_layers_and_ignores_suffix(self):
        import torch

        class AddLayer(torch.nn.Module):
            def __init__(self, amount):
                super().__init__()
                self.amount = amount

            def forward(self, hidden):
                return hidden + self.amount

        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [AddLayer(float(index + 1)) for index in range(32)]
                )

            def forward(self, input_ids):
                hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 4)
                for layer in self.layers:
                    hidden = layer(hidden)
                return SimpleNamespace(last_hidden_state=hidden)

        decoder = Decoder()
        with core.ResidualMeanPooler(decoder.layers) as pooler:
            pooler.begin(torch.tensor([1, 2], dtype=torch.long))
            decoder(torch.tensor([[10, 20, 30, 999]], dtype=torch.long))
            result = pooler.finish()
        self.assertEqual(result.shape, (32, 4))
        # Mean selected input is 25. Layer zero is post-block (+1); layer 31
        # contains the cumulative sum 1+...+32.
        self.assertEqual(result[0, 0], 26.0)
        self.assertEqual(result[31, 0], 25.0 + sum(range(1, 33)))

    def test_hook_alpha_zero_returns_bit_identical_tensor(self):
        import torch

        layer = torch.nn.Identity()
        hidden = torch.randn(2, 3, 4)
        directions = np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        with core.MultiLayerProjectionAblator(
            [layer], directions, [0], alpha=0.0
        ):
            result = layer(hidden)
        self.assertEqual(result.data_ptr(), hidden.data_ptr())
        torch.testing.assert_close(result, hidden, rtol=0, atol=0)

    def test_prefill_only_changes_last_position_and_alpha_one_is_zero(self):
        import torch

        layer = torch.nn.Identity()
        directions = np.asarray([[1.0, 0.0]], dtype=np.float32)
        prefill = torch.tensor([[[2.0, 8.0], [3.0, 7.0]]])
        with core.MultiLayerProjectionAblator(
            [layer], directions, [0], alpha=1.0
        ) as ablator:
            result = layer(prefill)
            torch.testing.assert_close(result[:, 0], prefill[:, 0])
            torch.testing.assert_close(result[:, 1, 0], torch.tensor([0.0]))
            decode = layer(torch.tensor([[[5.0, 6.0]]]))
            torch.testing.assert_close(decode[..., 0], torch.zeros_like(decode[..., 0]))
            telemetry = ablator.telemetry()
        self.assertEqual(telemetry[0]["hooked_positions"], 2)
        self.assertEqual(telemetry[0]["prefill_hooked_positions"], 1)
        self.assertEqual(telemetry[0]["decode_hooked_positions"], 1)
        self.assertEqual(telemetry[0]["token_scope"], "generation_only")
        self.assertLess(telemetry[0]["post_projection_abs_mean"], 1e-7)

    def test_all_positions_changes_every_prefill_position(self):
        import torch

        layer = torch.nn.Identity()
        directions = np.asarray([[1.0, 0.0]], dtype=np.float32)
        prefill = torch.tensor([[[2.0, 8.0], [3.0, 7.0]]])
        with core.MultiLayerProjectionAblator(
            [layer],
            directions,
            [0],
            alpha=1.0,
            token_scope="all_positions",
        ) as ablator:
            result = layer(prefill)
            torch.testing.assert_close(result[..., 0], torch.zeros_like(result[..., 0]))
            decode = layer(torch.tensor([[[5.0, 6.0]]]))
            torch.testing.assert_close(decode[..., 0], torch.zeros_like(decode[..., 0]))
            telemetry = ablator.telemetry()
        self.assertEqual(telemetry[0]["hooked_positions"], 3)
        self.assertEqual(telemetry[0]["prefill_hooked_positions"], 2)
        self.assertEqual(telemetry[0]["decode_hooked_positions"], 1)
        self.assertEqual(telemetry[0]["token_scope"], "all_positions")

    def test_cached_and_uncached_generation_only_use_equivalent_position_scope(self):
        import torch

        directions = np.asarray([[1.0, 0.0]], dtype=np.float32)
        prefill = torch.tensor([[[2.0, 8.0], [3.0, 7.0]]])
        decode = torch.tensor([[[5.0, 6.0]]])

        cached_layer = torch.nn.Identity()
        with core.MultiLayerProjectionAblator(
            [cached_layer], directions, [0], alpha=1.0, use_cache=True
        ) as cached:
            cached_prefill = cached_layer(prefill)
            cached_decode = cached_layer(decode)
            cached_telemetry = cached.telemetry()[0]

        uncached_layer = torch.nn.Identity()
        with core.MultiLayerProjectionAblator(
            [uncached_layer], directions, [0], alpha=1.0, use_cache=False
        ) as uncached:
            uncached_layer(prefill)
            recomputed = uncached_layer(torch.cat([prefill, decode], dim=1))
            uncached_telemetry = uncached.telemetry()[0]

        torch.testing.assert_close(
            torch.cat([cached_prefill, cached_decode], dim=1), recomputed
        )
        self.assertEqual(cached_telemetry["prefill_hooked_positions"], 1)
        self.assertEqual(cached_telemetry["decode_hooked_positions"], 1)
        # The final prompt state is re-intervened during full-prefix decoding.
        self.assertEqual(uncached_telemetry["prefill_hooked_positions"], 2)
        self.assertEqual(uncached_telemetry["decode_hooked_positions"], 1)

    def test_each_layer_uses_its_own_direction(self):
        import torch

        layers = [torch.nn.Identity(), torch.nn.Identity()]
        directions = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        hidden = torch.tensor([[[3.0, 4.0]]])
        with core.MultiLayerProjectionAblator(
            layers, directions, [0, 1], alpha=1.0
        ):
            value = layers[0](hidden)
            value = layers[1](value)
        torch.testing.assert_close(value, torch.zeros_like(value))

    def test_one_source_layer_direction_can_be_repeated_at_all_targets(self):
        import torch

        layers = [torch.nn.Identity(), torch.nn.Identity()]
        directions = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        hidden = torch.tensor([[[3.0, 4.0]]])
        with core.MultiLayerProjectionAblator(
            layers,
            directions,
            [0, 1],
            alpha=1.0,
            direction_source_layer=0,
        ) as ablator:
            value = layers[0](hidden)
            value = layers[1](value)
            telemetry = ablator.telemetry()
        torch.testing.assert_close(value, torch.tensor([[[0.0, 4.0]]]))
        self.assertEqual(
            [row["direction_source_layer"] for row in telemetry], [0, 0]
        )

    def test_repeated_random_source_is_identical_across_target_layers(self):
        import torch

        layers = [torch.nn.Identity(), torch.nn.Identity()]
        directions = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        with core.MultiLayerProjectionAblator(
            layers,
            directions,
            [0, 1],
            alpha=1.0,
            direction_source_layer=1,
            direction_variant="norm_matched_random",
            random_seed=99,
            random_key="artifact-hash",
        ) as ablator:
            torch.testing.assert_close(ablator.units[0], ablator.units[1])
            value = layers[0](torch.tensor([[[3.0, 4.0]]]))
            layers[1](value)
            telemetry = ablator.telemetry()
        self.assertEqual(
            [row["direction_variant"] for row in telemetry],
            ["norm_matched_random", "norm_matched_random"],
        )

    def test_hook_is_sign_invariant(self):
        import torch

        hidden = torch.tensor([[[3.0, 4.0]]])
        outputs = []
        for sign in (1.0, -1.0):
            layer = torch.nn.Identity()
            directions = np.asarray([[sign, 2.0 * sign]], dtype=np.float32)
            with core.MultiLayerProjectionAblator(
                [layer], directions, [0], alpha=1.0
            ):
                outputs.append(layer(hidden))
        torch.testing.assert_close(outputs[0], outputs[1])


class ConditionValidationTests(unittest.TestCase):
    def test_use_cache_defaults_true_and_requires_a_json_boolean(self):
        self.assertIs(runner.resolve_use_cache({}), True)
        self.assertIs(runner.resolve_use_cache({"use_cache": False}), False)
        with self.assertRaisesRegex(ValueError, "must be boolean"):
            runner.resolve_use_cache({"use_cache": "false"})

    def test_new_condition_fields_resolve_and_defaults_remain_compatible(self):
        directions = {
            "d": {
                "selected_layer": 1,
                "qualified_layers": [0, 1],
                "reference": None,
            }
        }
        config = {
            "conditions": [
                {"name": "baseline"},
                {"name": "legacy", "direction": "d"},
                {
                    "name": "broadcast_random",
                    "direction": "d",
                    "scope": "all",
                    "direction_source_layer": 1,
                    "direction_variant": "norm_matched_random",
                    "random_seed": 17,
                    "token_scope": "all_positions",
                },
            ]
        }
        conditions = runner.validate_conditions(
            config, directions, None, layer_count=2
        )
        by_name = {condition["name"]: condition for condition in conditions}
        self.assertEqual(by_name["legacy"]["direction_source_layer"], None)
        self.assertEqual(by_name["legacy"]["direction_variant"], "learned")
        self.assertEqual(by_name["legacy"]["token_scope"], "generation_only")
        self.assertEqual(by_name["broadcast_random"]["layers"], [0, 1])
        self.assertEqual(by_name["broadcast_random"]["direction_source_layer"], 1)
        self.assertEqual(by_name["broadcast_random"]["random_seed"], 17)
        self.assertEqual(by_name["broadcast_random"]["token_scope"], "all_positions")

    def test_invalid_token_scope_is_rejected(self):
        directions = {
            "d": {"selected_layer": 0, "qualified_layers": [], "reference": None}
        }
        config = {
            "conditions": [
                {"name": "baseline"},
                {"name": "bad", "direction": "d", "token_scope": "prompt_only"},
            ]
        }
        with self.assertRaisesRegex(ValueError, "unknown token scope"):
            runner.validate_conditions(config, directions, None, layer_count=1)


class ResumeCompatibilityTests(unittest.TestCase):
    def make_compatibility(self):
        return runner.build_resume_compatibility(
            config_sha256="config-hash",
            prompts_sha256="prompts-hash",
            prompt_ids=["p0", "p1"],
            model={"base_model": "model", "base_revision": "revision"},
            generation={
                "samples_per_prompt": 5,
                "do_sample": True,
                "temperature": 1.0,
                "top_p": 1.0,
                "max_new_tokens": 2000,
                "use_cache": True,
                "seed": 42,
            },
            directions={
                "d": {
                    "artifact_sha256": "direction-hash",
                }
            },
            conditions=[
                {
                    "name": "baseline",
                    "kind": "baseline",
                    "layers": [],
                },
                {
                    "name": "project",
                    "kind": "projection",
                    "layers": [0, 1],
                    "token_scope": "all_positions",
                    "direction_source_layer": 1,
                    "alpha": 1.0,
                },
            ],
        )

    def test_exact_resume_compatibility_is_accepted(self):
        current = self.make_compatibility()
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "status": "running",
                        "resume_compatibility": current,
                        "unrelated_runtime_field": "ignored",
                    }
                ),
                encoding="utf-8",
            )
            runner.require_resume_compatibility(manifest, current)

    def test_missing_and_legacy_resume_manifests_are_rejected_without_rewrite(self):
        current = self.make_compatibility()
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            with self.assertRaisesRegex(ValueError, "existing manifest"):
                runner.require_resume_compatibility(manifest, current)

            legacy_text = json.dumps({"schema_version": 1, "status": "running"})
            manifest.write_text(legacy_text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "legacy resume manifest"):
                runner.require_resume_compatibility(manifest, current)
            self.assertEqual(manifest.read_text(encoding="utf-8"), legacy_text)

    def test_every_required_resume_component_is_strictly_checked(self):
        current = self.make_compatibility()
        mutations = {
            "config_sha256": lambda value: value.__setitem__(
                "config_sha256", "different-config"
            ),
            "prompts_sha256": lambda value: value.__setitem__(
                "prompts_sha256", "different-prompts"
            ),
            "model": lambda value: value["model"].__setitem__(
                "base_revision", "different-revision"
            ),
            "generation": lambda value: value["generation"].__setitem__(
                "use_cache", False
            ),
            "direction_artifact_sha256": lambda value: value[
                "direction_artifact_sha256"
            ].__setitem__("d", "different-direction"),
            "conditions": lambda value: value["conditions"][1].__setitem__(
                "alpha", 0.5
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            for expected_field, mutate in mutations.items():
                with self.subTest(expected_field=expected_field):
                    saved = copy.deepcopy(current)
                    mutate(saved)
                    manifest.write_text(
                        json.dumps({"resume_compatibility": saved}),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(
                        ValueError, rf"mismatched fields:.*{expected_field}"
                    ):
                        runner.require_resume_compatibility(manifest, current)


class ManifestExecutionProvenanceTests(unittest.TestCase):
    def test_running_manifest_distinguishes_generation_from_content_execution(self):
        fields = runner.execution_provenance("running")
        self.assertEqual(
            fields,
            {
                "model_generation_executed": False,
                "generated_content_executed": False,
            },
        )
        self.assertNotIn("model_output_executed", fields)

    def test_success_manifest_records_generation_but_never_content_execution(self):
        fields = runner.execution_provenance("success")
        self.assertEqual(
            fields,
            {
                "model_generation_executed": True,
                "generated_content_executed": False,
            },
        )
        self.assertNotIn("model_output_executed", fields)

    def test_unknown_manifest_status_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported execution provenance"):
            runner.execution_provenance("failed")


if __name__ == "__main__":
    unittest.main()
