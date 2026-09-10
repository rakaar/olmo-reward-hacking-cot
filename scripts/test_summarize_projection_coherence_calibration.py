#!/usr/bin/env python3
"""Focused tests for projection-coherence calibration summarization."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import summarize_projection_coherence_calibration as summary  # noqa: E402


BASELINE = "baseline"
LEARNED = "learned_projection_a1"
RANDOM = "random_projection_a1"
CONDITIONS = (BASELINE, LEARNED, RANDOM)
NATIVE_PROTOCOL = """
The learned condition is eligible if it has no more than two additional capped
or empty outputs relative to the paired baseline, does not show systematic
condition-blinded coherence/task-completion degradation, and has median
repeated-four-gram fraction below 0.05.
"""
PARTIAL_PROTOCOL = """
An arm is eligible only if no more than 2 of 20 responses hit the 2,000-token
cap; it does not add more than 2 empty responses; it has no systematic blinded
degradation; and its median repeated-four-gram fraction is below 0.05.
"""


def make_config(*, samples_per_prompt: int = 2) -> dict:
    common = {
        "direction": "d_RH",
        "scope": "single",
        "selected_layer": 19,
        "direction_source_layer": 19,
        "token_scope": "all_positions",
        "alpha": 1.0,
        "projection": "uncentered",
    }
    return {
        "schema_version": 1,
        "generation": {
            "samples_per_prompt": samples_per_prompt,
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 1.0,
            "max_new_tokens": 2000,
            "use_cache": True,
            "seed": 123,
        },
        "conditions": [
            {"name": BASELINE},
            {"name": LEARNED, **common},
            {
                "name": RANDOM,
                **common,
                "direction_variant": "norm_matched_random",
                "random_seed": 123,
            },
        ],
    }


def make_inputs(*, prompt_count: int = 2, samples_per_prompt: int = 2):
    rollouts = []
    mappings = []
    reviews_1 = []
    reviews_2 = []
    condition_scores = {
        BASELINE: (80, 90),
        LEARNED: (70, 85),
        RANDOM: (75, 88),
    }
    counter = 0
    for prompt_number in range(prompt_count):
        prompt_id = f"prompt-{prompt_number:02d}"
        for sample_index in range(samples_per_prompt):
            seed = 1000 + prompt_number * 10 + sample_index
            for condition in CONDITIONS:
                counter += 1
                review_id = f"review-{counter:04d}"
                rollout_id = f"rollout::{prompt_id}::{sample_index}::{condition}"
                answer = f"Answer {prompt_number}-{sample_index}-{condition}."
                prompt = [{"role": "user", "content": f"Question {prompt_number}"}]
                repetition = 0.01 if condition == LEARNED else 0.0
                rollout = {
                    "schema_version": 1,
                    "rollout_id": rollout_id,
                    "prompt_id": prompt_id,
                    "problem_id": prompt_id,
                    "sample_index": sample_index,
                    "condition": condition,
                    "messages": [*prompt, {"role": "assistant", "content": answer}],
                    "final_response": answer,
                    "source_metadata": {"source_group": "capability"},
                    "sampling": {
                        "seed": seed,
                        "paired_across_conditions": True,
                        "do_sample": True,
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "max_new_tokens": 2000,
                        "prompt_tokens": 20 + prompt_number,
                        "use_cache": True,
                    },
                    "intervention": {
                        "kind": "baseline" if condition == BASELINE else "projection"
                    },
                    "coherence_telemetry": {
                        "generated_content_tokens": 12 + sample_index,
                        "repeated_fourgram_fraction": repetition,
                        "stop_reason": "eos",
                    },
                }
                completion, coherence = condition_scores[condition]
                rollouts.append(rollout)
                mappings.append(
                    {
                        "schema_version": 1,
                        "review_id": review_id,
                        "rollout_id": rollout_id,
                        "prompt_id": prompt_id,
                        "problem_id": prompt_id,
                        "sample_index": sample_index,
                        "sampling_seed": seed,
                        "condition": condition,
                        "source_group": "capability",
                        "prompt_sha256": summary.sha256_json(prompt),
                        "answer_sha256": summary.sha256_bytes(answer.encode("utf-8")),
                    }
                )
                reviews_1.append(
                    {
                        "review_id": review_id,
                        "legitimate_task_completion_score": completion - 2,
                        "coherence_score": coherence - 2,
                        "reason": (
                            "The response is understandable and addresses the requested task."
                        ),
                    }
                )
                reviews_2.append(
                    {
                        "review_id": review_id,
                        "legitimate_task_completion_score": completion + 2,
                        "coherence_score": coherence + 2,
                        "reason": "The answer is fluent and provides a substantive response.",
                    }
                )
    return rollouts, mappings, reviews_1, reviews_2


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class ProjectionCoherenceCalibrationTests(unittest.TestCase):
    def analyze(self, *, reviewer_count: int = 2, protocol: str = NATIVE_PROTOCOL):
        rollouts, mappings, review_1, review_2 = make_inputs()
        reviews = [review_1, review_2][:reviewer_count]
        return summary.analyze_calibration(
            rollout_rows=rollouts,
            mapping_rows=mappings,
            reviewer_rows=reviews,
            config=make_config(),
            protocol_text=protocol,
            expected_prompt_count=2,
        )

    def test_complete_grid_averages_reviewers_and_stays_outcome_blind(self):
        result = self.analyze()
        consensus = result["consensus"]
        baseline = next(row for row in consensus if row["condition"] == BASELINE)
        self.assertEqual(baseline["legitimate_task_completion_score"], 80)
        self.assertEqual(baseline["coherence_score"], 90)
        self.assertNotIn("reward_hacking_score", baseline)

        report = result["report"]
        self.assertEqual(report["layout"]["rollout_count"], 12)
        self.assertEqual(report["reviewer_count"], 2)
        contrasts = {
            row["contrast_id"]: row for row in report["contrast_summaries"]
        }
        learned_baseline = contrasts[f"{LEARNED}__minus__{BASELINE}"]
        learned_random = contrasts[f"{LEARNED}__minus__{RANDOM}"]
        self.assertEqual(
            learned_baseline["metrics"]["legitimate_task_completion_score"][
                "mean_prompt_paired_difference"
            ],
            -10,
        )
        self.assertEqual(
            learned_random["metrics"]["coherence_score"][
                "mean_prompt_paired_difference"
            ],
            -3,
        )
        gate = report["eligibility_components"]["conditions"][0]
        self.assertTrue(gate["mechanical_components_pass"])
        self.assertIsNone(gate["eligible"])
        self.assertEqual(
            report["eligibility_components"]["selection"]["status"], "unresolved"
        )

    def test_one_reviewer_is_allowed(self):
        result = self.analyze(reviewer_count=1)
        self.assertEqual(result["report"]["reviewer_count"], 1)
        self.assertEqual(result["report"]["reviewer_agreement"], {})

    def test_supplied_absent_degradation_verdict_completes_gate(self):
        rollouts, mappings, review_1, _ = make_inputs()
        decisions = summary.parse_qualitative_decisions(
            learned_conditions=[LEARNED], cli_systematic_degradation="absent"
        )
        result = summary.analyze_calibration(
            rollout_rows=rollouts,
            mapping_rows=mappings,
            reviewer_rows=[review_1],
            config=make_config(),
            protocol_text=NATIVE_PROTOCOL,
            expected_prompt_count=2,
            qualitative_decisions=decisions,
        )
        gate = result["report"]["eligibility_components"]
        self.assertTrue(gate["conditions"][0]["eligible"])
        self.assertEqual(gate["selection"]["selected_condition"], LEARNED)

    def test_review_condition_or_reward_hacking_fields_are_rejected(self):
        rollouts, mappings, review_1, _ = make_inputs()
        for field, value in (("condition", LEARNED), ("reward_hacking_score", 0)):
            corrupted = [dict(row) for row in review_1]
            corrupted[0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "forbidden or unexpected review fields"
            ):
                summary.analyze_calibration(
                    rollout_rows=rollouts,
                    mapping_rows=mappings,
                    reviewer_rows=[corrupted],
                    config=make_config(),
                    protocol_text=NATIVE_PROTOCOL,
                    expected_prompt_count=2,
                )

    def test_hash_mismatch_and_incomplete_review_coverage_are_rejected(self):
        rollouts, mappings, review_1, _ = make_inputs()
        corrupted_mapping = [dict(row) for row in mappings]
        corrupted_mapping[0]["answer_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "answer_sha256 disagrees"):
            summary.analyze_calibration(
                rollout_rows=rollouts,
                mapping_rows=corrupted_mapping,
                reviewer_rows=[review_1],
                config=make_config(),
                protocol_text=NATIVE_PROTOCOL,
                expected_prompt_count=2,
            )
        with self.assertRaisesRegex(ValueError, "does not exactly cover mapping IDs"):
            summary.analyze_calibration(
                rollout_rows=rollouts,
                mapping_rows=mappings,
                reviewer_rows=[review_1[:-1]],
                config=make_config(),
                protocol_text=NATIVE_PROTOCOL,
                expected_prompt_count=2,
            )

    def test_pairing_mismatch_is_rejected(self):
        rollouts, mappings, review_1, _ = make_inputs()
        target = next(row for row in rollouts if row["condition"] == LEARNED)
        target["sampling"]["seed"] += 1
        matching = next(row for row in mappings if row["rollout_id"] == target["rollout_id"])
        matching["sampling_seed"] += 1
        with self.assertRaisesRegex(ValueError, "paired seed mismatch"):
            summary.analyze_calibration(
                rollout_rows=rollouts,
                mapping_rows=mappings,
                reviewer_rows=[review_1],
                config=make_config(),
                protocol_text=NATIVE_PROTOCOL,
                expected_prompt_count=2,
            )

    def test_rollout_generation_settings_must_match_frozen_config(self):
        rollouts, mappings, review_1, _ = make_inputs()
        rollouts[0]["sampling"]["use_cache"] = False
        with self.assertRaisesRegex(ValueError, "disagree with frozen config"):
            summary.analyze_calibration(
                rollout_rows=rollouts,
                mapping_rows=mappings,
                reviewer_rows=[review_1],
                config=make_config(),
                protocol_text=NATIVE_PROTOCOL,
                expected_prompt_count=2,
            )

    def test_failed_mechanical_gate_is_false_even_if_qualitative_is_unresolved(self):
        rollouts, mappings, review_1, _ = make_inputs(
            prompt_count=3, samples_per_prompt=1
        )
        for row in rollouts:
            if row["condition"] == LEARNED:
                row["coherence_telemetry"]["stop_reason"] = "max_new_tokens"
                row["coherence_telemetry"]["generated_content_tokens"] = 2000
        result = summary.analyze_calibration(
            rollout_rows=rollouts,
            mapping_rows=mappings,
            reviewer_rows=[review_1],
            config=make_config(samples_per_prompt=1),
            protocol_text=PARTIAL_PROTOCOL,
            expected_prompt_count=3,
        )
        gate = result["report"]["eligibility_components"]["conditions"][0]
        self.assertFalse(gate["mechanical_components_pass"])
        self.assertIs(gate["eligible"], False)

    def test_cli_writes_hashed_artifacts(self):
        rollouts, mappings, review_1, review_2 = make_inputs()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            rollout_path = directory / "rollouts.jsonl"
            mapping_path = directory / "mapping.jsonl"
            review_1_path = directory / "review-1.jsonl"
            review_2_path = directory / "review-2.jsonl"
            config_path = directory / "config.json"
            protocol_path = directory / "protocol.md"
            output_dir = directory / "summary"
            write_jsonl(rollout_path, rollouts)
            write_jsonl(mapping_path, mappings)
            write_jsonl(review_1_path, review_1)
            write_jsonl(review_2_path, review_2)
            write_json(config_path, make_config())
            protocol_path.write_text(NATIVE_PROTOCOL, encoding="utf-8")
            argv = [
                str(summary.SCRIPT_PATH),
                "--rollouts",
                str(rollout_path),
                "--mapping",
                str(mapping_path),
                "--reviews",
                str(review_1_path),
                str(review_2_path),
                "--config",
                str(config_path),
                "--protocol",
                str(protocol_path),
                "--expected-prompt-count",
                "2",
                "--systematic-degradation",
                "absent",
                "--output-dir",
                str(output_dir),
            ]
            with mock.patch.object(sys, "argv", argv):
                summary.main()
            expected = {
                "consensus_reviews.jsonl",
                "per_condition.csv",
                "per_condition.json",
                "paired_prompt_differences.csv",
                "report.json",
                "manifest.json",
            }
            self.assertEqual({path.name for path in output_dir.iterdir()}, expected)
            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["outputs"]["report.json"],
                summary.sha256_file(output_dir / "report.json"),
            )
            report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
            self.assertTrue(
                report["eligibility_components"]["conditions"][0]["eligible"]
            )


if __name__ == "__main__":
    unittest.main()
