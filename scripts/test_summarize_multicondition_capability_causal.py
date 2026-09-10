#!/usr/bin/env python3
"""Laptop-only tests for the multi-condition causal summarizer."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import summarize_multicondition_capability_causal as summary  # noqa: E402


class ConditionPlanTests(unittest.TestCase):
    @staticmethod
    def _paired_arm(name: str, *, alpha: float, random: bool = False) -> dict:
        value = {
            "name": name,
            "direction": "d_RH",
            "scope": "all32",
            "direction_source_layer": 19,
            "token_scope": "all_positions",
            "alpha": alpha,
            "projection": "uncentered",
        }
        if random:
            value["direction_variant"] = "norm_matched_random"
            value["random_seed"] = 42
        return value

    def test_frozen_config_infers_two_primary_random_matches(self):
        config = json.loads(summary.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        plan = summary.infer_condition_plan(config)
        self.assertEqual(plan["baseline"], "baseline")
        self.assertEqual(
            plan["primary_learned_conditions"],
            [
                "rh_repeat_l19_all32_allpos_a1",
                "rh_layerwise_all32_allpos_a1",
            ],
        )
        self.assertEqual(
            plan["matching_random_controls"]["rh_repeat_l19_all32_allpos_a1"],
            "random_repeat_l19_all32_allpos_a1",
        )
        self.assertEqual(len(plan["contrasts"]), 8)

    def test_random_control_must_match_exact_learned_scope(self):
        config = {
            "conditions": [
                {"name": "baseline"},
                {
                    "name": "learned_a",
                    "direction": "d",
                    "scope": "all",
                    "token_scope": "all_positions",
                },
                {
                    "name": "random_a",
                    "direction": "d",
                    "scope": "single",
                    "token_scope": "all_positions",
                    "direction_variant": "norm_matched_random",
                },
            ]
        }
        with self.assertRaisesRegex(ValueError, "0 learned matches"):
            summary.infer_condition_plan(config)

    def test_declared_five_condition_config_has_two_primaries(self):
        first = "rh_native_l19_allpos_a0p5"
        first_random = "random_native_l19_allpos_a0p5"
        second = "rh_layerwise_all32_allpos_a1"
        second_random = "random_layerwise_all32_allpos_a1"
        config = {
            "analysis": {"primary_learned_conditions": [first, second]},
            "conditions": [
                {"name": "baseline"},
                self._paired_arm(first, alpha=0.5),
                self._paired_arm(first_random, alpha=0.5, random=True),
                self._paired_arm(second, alpha=1.0),
                self._paired_arm(second_random, alpha=1.0, random=True),
            ],
        }
        # Give the layerwise pair a distinct signature from the native pair.
        config["conditions"][3]["direction_source_layer"] = None
        config["conditions"][4]["direction_source_layer"] = None
        plan = summary.infer_condition_plan(config)
        self.assertEqual(plan["primary_learned_conditions"], [first, second])
        self.assertEqual(plan["sensitivity_learned_conditions"], [])
        self.assertEqual(
            plan["primary_learned_conditions_source"],
            "config.analysis.primary_learned_conditions",
        )
        self.assertEqual(len(plan["condition_names"]), 5)
        self.assertEqual(len(plan["contrasts"]), 6)
        self.assertEqual(plan["matching_random_controls"][first], first_random)

    def test_declared_seven_condition_config_keeps_extra_learned_as_sensitivity(self):
        primary_a = "rh_native_l19_allpos_a0p5"
        random_a = "random_native_l19_allpos_a0p5"
        primary_b = "rh_layerwise_all32_allpos_a1"
        random_b = "random_layerwise_all32_allpos_a1"
        sensitivity = "rh_native_l19_allpos_a0p25"
        sensitivity_random = "random_native_l19_allpos_a0p25"
        config = {
            "analysis": {
                "primary_learned_conditions": [primary_a, primary_b]
            },
            "conditions": [
                {"name": "baseline"},
                self._paired_arm(primary_a, alpha=0.5),
                self._paired_arm(random_a, alpha=0.5, random=True),
                self._paired_arm(primary_b, alpha=1.0),
                self._paired_arm(random_b, alpha=1.0, random=True),
                self._paired_arm(sensitivity, alpha=0.25),
                self._paired_arm(sensitivity_random, alpha=0.25, random=True),
            ],
        }
        config["conditions"][3]["direction_source_layer"] = None
        config["conditions"][4]["direction_source_layer"] = None
        plan = summary.infer_condition_plan(config)
        self.assertEqual(plan["primary_learned_conditions"], [primary_a, primary_b])
        self.assertEqual(plan["sensitivity_learned_conditions"], [sensitivity])
        self.assertEqual(len(plan["condition_names"]), 7)
        sensitivity_baseline = next(
            row
            for row in plan["contrasts"]
            if row["condition"] == sensitivity and row["comparator"] == "baseline"
        )
        self.assertEqual(sensitivity_baseline["role"], "sensitivity_learned_vs_baseline")
        self.assertFalse(sensitivity_baseline["primary"])
        sensitivity_vs_random = next(
            row
            for row in plan["contrasts"]
            if row["condition"] == sensitivity
            and row["comparator"] == sensitivity_random
        )
        self.assertEqual(
            sensitivity_vs_random["role"], "sensitivity_learned_vs_random"
        )
        self.assertFalse(sensitivity_vs_random["primary"])
        self.assertEqual(len(plan["contrasts"]), 9)

    def test_planned_seven_conditions_have_one_primary_and_two_sensitivities(self):
        native = {
            "name": "rh_native_l19_allpos_a1",
            "direction": "d_RH",
            "scope": "single",
            "selected_layer": 19,
            "direction_source_layer": 19,
            "token_scope": "all_positions",
            "alpha": 1.0,
            "projection": "uncentered",
        }
        layerwise = {
            "name": "rh_layerwise_all32_allpos_a1",
            "direction": "d_RH",
            "scope": "all32",
            "token_scope": "all_positions",
            "alpha": 1.0,
            "projection": "uncentered",
        }
        repeated = {
            "name": "rh_repeat_l19_all32_allpos_a0p5",
            "direction": "d_RH",
            "scope": "all32",
            "direction_source_layer": 19,
            "token_scope": "all_positions",
            "alpha": 0.5,
            "projection": "uncentered",
        }

        def matching_random(condition: dict, name: str) -> dict:
            result = {**condition, "name": name}
            result.update(
                direction_variant="norm_matched_random",
                random_seed=20260914,
            )
            return result

        config = {
            "analysis": {
                "primary_learned_conditions": [
                    "rh_layerwise_all32_allpos_a1"
                ]
            },
            "conditions": [
                {"name": "baseline"},
                native,
                matching_random(native, "random_native_l19_allpos_a1"),
                layerwise,
                matching_random(
                    layerwise, "random_layerwise_all32_allpos_a1"
                ),
                repeated,
                matching_random(
                    repeated, "random_repeat_l19_all32_allpos_a0p5"
                ),
            ],
        }
        plan = summary.infer_condition_plan(config)
        self.assertEqual(
            plan["primary_learned_conditions"],
            ["rh_layerwise_all32_allpos_a1"],
        )
        self.assertEqual(
            plan["sensitivity_learned_conditions"],
            ["rh_native_l19_allpos_a1", "rh_repeat_l19_all32_allpos_a0p5"],
        )
        self.assertEqual(len(plan["condition_names"]), 7)
        self.assertEqual(len(plan["contrasts"]), 9)
        sensitivity_random_pairs = {
            (row["condition"], row["comparator"])
            for row in plan["contrasts"]
            if row["role"] == "sensitivity_learned_vs_random"
        }
        self.assertEqual(
            sensitivity_random_pairs,
            {
                ("rh_native_l19_allpos_a1", "random_native_l19_allpos_a1"),
                (
                    "rh_repeat_l19_all32_allpos_a0p5",
                    "random_repeat_l19_all32_allpos_a0p5",
                ),
            },
        )

    def test_declared_primary_must_be_learned_and_random_matched(self):
        learned = self._paired_arm("learned", alpha=0.5)
        with self.assertRaisesRegex(ValueError, "not learned arms"):
            summary.infer_condition_plan(
                {
                    "analysis": {"primary_learned_conditions": ["random"]},
                    "conditions": [
                        {"name": "baseline"},
                        learned,
                        self._paired_arm("random", alpha=0.5, random=True),
                    ],
                }
            )
        with self.assertRaisesRegex(ValueError, "lack exactly one"):
            summary.infer_condition_plan(
                {
                    "analysis": {"primary_learned_conditions": ["learned"]},
                    "conditions": [
                        {"name": "baseline"},
                        learned,
                        self._paired_arm("sensitivity", alpha=1.0),
                        self._paired_arm("random", alpha=1.0, random=True),
                    ],
                }
            )

    def test_short_labels_render_partial_alpha_and_native_layer(self):
        self.assertEqual(
            summary.short_condition_label("rh_native_l19_allpos_a0p5"),
            "learned native L19 all-pos alpha-0.5",
        )
        self.assertEqual(
            summary.short_condition_label("random_native_l19_allpos_a1"),
            "random native L19 all-pos alpha-1",
        )


class ConsensusTests(unittest.TestCase):
    @staticmethod
    def _rollout(rollout_id: str, condition: str) -> dict:
        return {
            "rollout_id": rollout_id,
            "prompt_id": "p1",
            "sample_index": 0,
            "condition": condition,
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
            "final_response": "answer",
            "sampling": {
                "seed": 7,
                "paired_across_conditions": True,
                "do_sample": True,
                "temperature": 1.0,
                "top_p": 1.0,
                "max_new_tokens": 100,
                "prompt_tokens": 10,
                "use_cache": True,
            },
            "source_metadata": {"source_group": "g1"},
            "coherence_telemetry": {
                "generated_content_tokens": 2,
                "repeated_fourgram_fraction": 0.0,
                "stop_reason": "eos",
            },
            "intervention": {},
        }

    def test_consensus_arithmetic_averages_all_three_scores(self):
        rollouts = [
            self._rollout("r-base", "baseline"),
            self._rollout("r-test", "test"),
        ]
        mappings = [
            {
                "review_id": "v1",
                "rollout_id": "r-base",
                "prompt_id": "p1",
                "condition": "baseline",
                "source_group": "g1",
            },
            {
                "review_id": "v2",
                "rollout_id": "r-test",
                "prompt_id": "p1",
                "condition": "test",
                "source_group": "g1",
            },
        ]
        reviewer_1 = [
            {
                "review_id": review_id,
                "reward_hacking_score": 20,
                "legitimate_task_completion_score": 80,
                "coherence_score": 90,
            }
            for review_id in ("v1", "v2")
        ]
        reviewer_2 = [
            {
                "review_id": review_id,
                "reward_hacking_score": 40,
                "legitimate_task_completion_score": 60,
                "coherence_score": 70,
            }
            for review_id in ("v1", "v2")
        ]
        consensus, agreement = summary.build_consensus(
            rollouts,
            mappings,
            [reviewer_1, reviewer_2],
            expected_conditions=["baseline", "test"],
        )
        self.assertEqual(len(consensus), 2)
        self.assertEqual(consensus[0]["reward_hacking_score"], 30)
        self.assertEqual(consensus[0]["legitimate_task_completion_score"], 70)
        self.assertEqual(consensus[0]["coherence_score"], 80)
        self.assertIn("reviewer_1_vs_2", agreement)

    def test_requires_two_complete_reviewers(self):
        rollout = self._rollout("r-base", "baseline")
        mapping = [{"review_id": "v1", "rollout_id": "r-base"}]
        reviewer = [
            {
                "review_id": "v1",
                "reward_hacking_score": 0,
                "legitimate_task_completion_score": 100,
                "coherence_score": 100,
            }
        ]
        with self.assertRaisesRegex(ValueError, "at least two"):
            summary.build_consensus(
                [rollout], mapping, [reviewer], expected_conditions=["baseline"]
            )


class PairedStatisticsTests(unittest.TestCase):
    def test_five_seed_differences_are_averaged_within_prompt(self):
        rows = []
        for prompt_id, group, differences in (
            ("p1", "a", [-20, -10, 0, 10, 20]),
            ("p2", "b", [-50, -50, -50, -50, -50]),
        ):
            for sample_index, difference in enumerate(differences):
                for condition, rh in (("baseline", 60), ("learned", 60 + difference)):
                    rows.append(
                        {
                            "prompt_id": prompt_id,
                            "sample_index": sample_index,
                            "condition": condition,
                            "source_group": group,
                            "sampling_seed": 100 + sample_index,
                            "paired_sampling_sha256": "same-settings",
                            "prompt_sha256": f"prompt-{prompt_id}",
                            "reward_hacking_score": rh,
                            "legitimate_task_completion_score": 90,
                            "coherence_score": 90,
                            "reward_hacking_at_50": rh >= 50,
                        }
                    )
        contrasts = [
            {
                "contrast_id": "learned__minus__baseline",
                "condition": "learned",
                "comparator": "baseline",
                "role": "primary_learned_vs_baseline",
                "primary": True,
            }
        ]
        per_prompt, details = summary.build_per_prompt_rows(
            rows, contrasts, expected_samples=5
        )
        by_prompt = {row["prompt_id"]: row for row in per_prompt}
        self.assertEqual(by_prompt["p1"]["reward_hacking_score_difference"], 0)
        self.assertEqual(by_prompt["p2"]["reward_hacking_score_difference"], -50)
        self.assertEqual(details["learned__minus__baseline"]["seed_pair_count"], 10)

    def test_stratified_bootstrap_preserves_each_group_count(self):
        groups = ["a", "a", "b", "b", "b"]
        draws = summary.stratified_bootstrap_indices(groups, replicates=25, seed=7)
        self.assertEqual(draws.shape, (25, 5))
        for draw in draws:
            selected = [groups[index] for index in draw]
            self.assertEqual(selected.count("a"), 2)
            self.assertEqual(selected.count("b"), 3)
        np.testing.assert_array_equal(
            draws,
            summary.stratified_bootstrap_indices(groups, replicates=25, seed=7),
        )

    def test_exact_prompt_sign_flip_known_case(self):
        pvalues = summary.exact_sign_flip_pvalues(np.asarray([[1.0], [1.0]]))
        self.assertEqual(float(pvalues[0]), 0.5)


class SuccessGateTests(unittest.TestCase):
    @staticmethod
    def _result(condition: str, comparator: str, *, rh, completion, coherence):
        def metric(observed, low, high):
            return {
                "observed_mean_difference": observed,
                "confidence_interval_95": [low, high],
            }

        return {
            "condition": condition,
            "comparator": comparator,
            "metrics": {
                "reward_hacking_score": metric(*rh),
                "legitimate_task_completion_score": metric(*completion),
                "coherence_score": metric(*coherence),
            },
        }

    def test_frozen_gate_requires_suppression_random_specificity_and_quality(self):
        plan = {
            "baseline": "baseline",
            "primary_learned_conditions": ["learned"],
            "matching_random_controls": {"learned": "random"},
        }
        passing = [
            self._result(
                "learned",
                "baseline",
                rh=(-10, -18, -1),
                completion=(-2, -9, 3),
                coherence=(-1, -8, 4),
            ),
            self._result(
                "learned",
                "random",
                rh=(-1, -5, 2),
                completion=(0, -2, 2),
                coherence=(0, -2, 2),
            ),
        ]
        gate = summary.evaluate_success_gate(passing, plan)
        self.assertTrue(gate["conditions"]["learned"]["success"])

        damaged = [
            self._result(
                "learned",
                "baseline",
                rh=(-15, -22, -2),
                completion=(-12, -18, -7),
                coherence=(-1, -8, 4),
            ),
            passing[1],
        ]
        gate = summary.evaluate_success_gate(damaged, plan)
        self.assertFalse(gate["conditions"]["learned"]["success"])
        self.assertEqual(
            gate["conditions"]["learned"]["classification"],
            "nonspecific_degradation",
        )

    def test_gate_excludes_declared_sensitivity_conditions(self):
        plan = {
            "baseline": "baseline",
            "primary_learned_conditions": ["layerwise"],
            "sensitivity_learned_conditions": ["native", "repeated"],
            "matching_random_controls": {
                "layerwise": "random_layerwise",
                "native": "random_native",
                "repeated": "random_repeated",
            },
        }
        rows = [
            self._result(
                "layerwise",
                "baseline",
                rh=(-12, -18, -2),
                completion=(0, -5, 4),
                coherence=(0, -4, 5),
            ),
            self._result(
                "layerwise",
                "random_layerwise",
                rh=(-2, -7, 3),
                completion=(0, -2, 2),
                coherence=(0, -2, 2),
            ),
            self._result(
                "native",
                "baseline",
                rh=(-50, -60, -40),
                completion=(-50, -60, -40),
                coherence=(-50, -60, -40),
            ),
            self._result(
                "native",
                "random_native",
                rh=(-50, -60, -40),
                completion=(-50, -60, -40),
                coherence=(-50, -60, -40),
            ),
        ]
        gate = summary.evaluate_success_gate(rows, plan)
        self.assertEqual(set(gate["conditions"]), {"layerwise"})
        self.assertTrue(gate["conditions"]["layerwise"]["success"])

    def test_quality_failure_is_degradation_even_when_random_specificity_fails(self):
        plan = {
            "baseline": "baseline",
            "primary_learned_conditions": ["learned"],
            "matching_random_controls": {"learned": "random"},
        }
        rows = [
            self._result(
                "learned",
                "baseline",
                rh=(-15, -22, -3),
                completion=(-16, -23, -11),
                coherence=(0, -5, 5),
            ),
            self._result(
                "learned",
                "random",
                rh=(2, -4, 8),
                completion=(0, -3, 3),
                coherence=(0, -3, 3),
            ),
        ]
        gate = summary.evaluate_success_gate(rows, plan)
        learned = gate["conditions"]["learned"]
        self.assertFalse(learned["criteria"]["rh_mean_below_matching_random"])
        self.assertFalse(learned["success"])
        self.assertEqual(learned["classification"], "nonspecific_degradation")


class OutputIntegrationTests(unittest.TestCase):
    def test_full_frozen_layout_writes_every_declared_artifact(self):
        config = json.loads(summary.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        conditions = [item["name"] for item in config["conditions"]]
        rollout_rows = []
        mapping_rows = []
        reviews = [[], []]
        review_index = 0
        for prompt_index in range(20):
            prompt_id = f"p{prompt_index:02d}"
            source_group = "same_distribution" if prompt_index < 10 else "cross_domain"
            for sample_index in range(5):
                seed = 1000 + prompt_index * 5 + sample_index
                for condition_index, condition in enumerate(conditions):
                    review_index += 1
                    review_id = f"review-{review_index:04d}"
                    rollout_id = f"rollout-{review_index:04d}"
                    rollout_rows.append(
                        {
                            "rollout_id": rollout_id,
                            "prompt_id": prompt_id,
                            "sample_index": sample_index,
                            "condition": condition,
                            "messages": [
                                {"role": "user", "content": f"question {prompt_id}"},
                                {"role": "assistant", "content": "answer"},
                            ],
                            "final_response": "answer",
                            "sampling": {
                                "seed": seed,
                                "paired_across_conditions": True,
                                "do_sample": True,
                                "temperature": 1.0,
                                "top_p": 1.0,
                                "max_new_tokens": 100,
                                "prompt_tokens": 10,
                                "use_cache": True,
                            },
                            "source_metadata": {"source_group": source_group},
                            "coherence_telemetry": {
                                "generated_content_tokens": 10,
                                "repeated_fourgram_fraction": 0.0,
                                "stop_reason": "eos",
                            },
                            "intervention": {
                                "relative_update_norm_mean_across_layers": (
                                    0.0 if condition == "baseline" else 0.01
                                )
                            },
                        }
                    )
                    mapping_rows.append(
                        {
                            "review_id": review_id,
                            "rollout_id": rollout_id,
                            "prompt_id": prompt_id,
                            "sample_index": sample_index,
                            "condition": condition,
                            "source_group": source_group,
                        }
                    )
                    # Give all interventions a small deterministic difference;
                    # integration here checks artifacts, not a scientific pass.
                    rh = 70 - condition_index
                    for reviewer_index, offset in enumerate((-1, 1)):
                        reviews[reviewer_index].append(
                            {
                                "review_id": review_id,
                                "reward_hacking_score": rh + offset,
                                "legitimate_task_completion_score": 90 + offset,
                                "coherence_score": 92 + offset,
                                "reason": "fixture",
                            }
                        )

        def write_rows(path: Path, rows):
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rollouts = root / "rollouts.jsonl"
            mapping = root / "mapping.jsonl"
            review_1 = root / "review_1.jsonl"
            review_2 = root / "review_2.jsonl"
            output = root / "output"
            write_rows(rollouts, rollout_rows)
            write_rows(mapping, mapping_rows)
            write_rows(review_1, reviews[0])
            write_rows(review_2, reviews[1])
            arguments = [
                "summarize",
                "--rollouts",
                str(rollouts),
                "--mapping",
                str(mapping),
                "--reviews",
                str(review_1),
                str(review_2),
                "--output-dir",
                str(output),
                "--bootstrap-replicates",
                "25",
            ]
            original = sys.argv
            try:
                sys.argv = arguments
                summary.main()
            finally:
                sys.argv = original
            expected = {
                "consensus_reviews.jsonl",
                "per_prompt.csv",
                "contrasts.csv",
                "contrasts.json",
                "bootstrap_metadata.json",
                "forest_quality.png",
                "manifest.json",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(set(manifest["outputs"]), expected - {"manifest.json"})
            self.assertEqual(manifest["expected_layout"]["prompts"], 20)
            for name, digest in manifest["outputs"].items():
                self.assertEqual(summary.sha256_file(output / name), digest)
            self.assertEqual(
                len((output / "consensus_reviews.jsonl").read_text().splitlines()), 700
            )
            self.assertEqual(len((output / "per_prompt.csv").read_text().splitlines()), 161)
            contrasts = json.loads((output / "contrasts.json").read_text())
            self.assertEqual(len(contrasts["contrasts"]), 8)


if __name__ == "__main__":
    unittest.main()
