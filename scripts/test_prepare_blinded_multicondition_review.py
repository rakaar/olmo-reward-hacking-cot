#!/usr/bin/env python3
"""Focused tests for the generic multi-condition blinded-review builder."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import prepare_blinded_multicondition_review as review_builder  # noqa: E402


CONDITIONS = ["baseline", *[f"condition-{index}" for index in range(1, 7)]]


def make_rows(prompt_count: int = 20, samples_per_prompt: int = 5):
    rows = []
    for prompt_number in range(prompt_count):
        prompt_id = f"prompt-{prompt_number:02d}"
        source_group = "heldout" if prompt_number < prompt_count // 2 else "novel"
        for sample_index in range(samples_per_prompt):
            sampling_seed = 100_000 + prompt_number * 100 + sample_index
            for condition in CONDITIONS:
                answer = f"answer for {prompt_id}, sample {sample_index}, {condition}"
                rows.append(
                    {
                        "schema_version": 1,
                        "rollout_id": (
                            f"rollout::{prompt_id}::sample-{sample_index}::{condition}"
                        ),
                        "prompt_id": prompt_id,
                        "problem_id": prompt_id,
                        "sample_index": sample_index,
                        "condition": condition,
                        "messages": [
                            {"role": "user", "content": f"question {prompt_number}"},
                            {"role": "assistant", "content": answer},
                        ],
                        "final_response": answer,
                        "source_metadata": {"source_group": source_group},
                        "sampling": {
                            "seed": sampling_seed,
                            "paired_across_conditions": True,
                            "do_sample": True,
                            "temperature": 1.0,
                            "top_p": 1.0,
                            "max_new_tokens": 2000,
                            "prompt_tokens": 40 + prompt_number,
                            "use_cache": True,
                        },
                        "intervention": {
                            "kind": "baseline" if condition == "baseline" else "projection",
                            "layers": [] if condition == "baseline" else [19],
                            "alpha": 0.0 if condition == "baseline" else 1.0,
                        },
                    }
                )
    return rows


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class MultiConditionReviewTests(unittest.TestCase):
    def run_builder(self, directory: Path, rows, *, suffix: str = ""):
        rollouts = directory / f"rollouts{suffix}.jsonl"
        packet = directory / f"packet{suffix}.jsonl"
        mapping = directory / f"mapping{suffix}.jsonl"
        manifest = directory / f"manifest{suffix}.json"
        write_jsonl(rollouts, rows)
        result = review_builder.prepare_review_artifacts(
            rollout_paths=[rollouts],
            packet_path=packet,
            mapping_path=mapping,
            manifest_path=manifest,
            expected_conditions=CONDITIONS,
            baseline_condition="baseline",
            expected_prompt_count=20,
            samples_per_prompt=5,
            shuffle_seed=31415,
        )
        return result, packet, mapping, manifest

    def test_complete_20_by_5_by_7_grid_is_blinded_and_hashed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result, packet_path, mapping_path, manifest_path = self.run_builder(
                directory, list(reversed(make_rows()))
            )
            packet = read_jsonl(packet_path)
            mapping = read_jsonl(mapping_path)
            saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(len(packet), 700)
            self.assertEqual(len(mapping), 700)
            self.assertEqual(result["record_count"], 700)
            self.assertEqual(result["paired_prompt_sample_count"], 100)
            self.assertEqual(result["condition_count"], 7)
            self.assertEqual(result["shared_baseline_count"], 100)
            self.assertEqual(set(result["condition_counts"].values()), {100})
            self.assertEqual(
                result["packet_sha256"], review_builder.sha256_file(packet_path)
            )
            self.assertEqual(
                result["mapping_sha256"], review_builder.sha256_file(mapping_path)
            )
            self.assertEqual(saved_manifest, result)

            allowed_packet_keys = {
                "schema_version",
                "review_id",
                "question_messages",
                "answer",
                "rubric",
                "response_format",
            }
            self.assertTrue(all(set(row) == allowed_packet_keys for row in packet))
            self.assertTrue(
                all(
                    set(row["response_format"])
                    == {
                        "reward_hacking_score",
                        "legitimate_task_completion_score",
                        "coherence_score",
                        "reason",
                    }
                    for row in packet
                )
            )
            self.assertEqual(len({row["review_id"] for row in packet}), 700)
            self.assertEqual(
                {row["review_id"] for row in packet},
                {row["review_id"] for row in mapping},
            )
            hidden = set(review_builder.PACKET_HIDDEN_FIELDS)
            self.assertTrue(all(not (set(row) & hidden) for row in packet))
            self.assertTrue(
                all(
                    {"condition", "prompt_id", "sample_index", "sampling_seed", "source_group"}
                    <= set(row)
                    for row in mapping
                )
            )

    def test_shuffle_is_independent_of_input_row_order(self):
        rows = make_rows()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, first_packet, first_mapping, _ = self.run_builder(
                directory, rows, suffix="-first"
            )
            _, second_packet, second_mapping, _ = self.run_builder(
                directory, list(reversed(rows)), suffix="-second"
            )
            self.assertEqual(first_packet.read_bytes(), second_packet.read_bytes())
            self.assertEqual(first_mapping.read_bytes(), second_mapping.read_bytes())

    def test_coherence_only_packet_omits_reward_hacking_rubric(self):
        rows = make_rows()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            rollouts = directory / "rollouts.jsonl"
            packet = directory / "packet.jsonl"
            mapping = directory / "mapping.jsonl"
            manifest = directory / "manifest.json"
            write_jsonl(rollouts, rows)
            result = review_builder.prepare_review_artifacts(
                rollout_paths=[rollouts],
                packet_path=packet,
                mapping_path=mapping,
                manifest_path=manifest,
                expected_conditions=CONDITIONS,
                baseline_condition="baseline",
                expected_prompt_count=20,
                samples_per_prompt=5,
                shuffle_seed=2,
                rubric_mode="coherence_only",
            )
            packet_rows = read_jsonl(packet)
            self.assertEqual(result["rubric_mode"], "coherence_only")
            self.assertTrue(
                all("reward_hacking_score" not in row["rubric"] for row in packet_rows)
            )
            self.assertTrue(
                all("two scores" in row["rubric"]["reason"] for row in packet_rows)
            )
            self.assertTrue(
                all(
                    set(row["response_format"])
                    == {
                        "legitimate_task_completion_score",
                        "coherence_score",
                        "reason",
                    }
                    for row in packet_rows
                )
            )

    def test_missing_grid_cell_fails_before_outputs_are_written(self):
        rows = make_rows()[:-1]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            rollouts = directory / "rollouts.jsonl"
            packet = directory / "packet.jsonl"
            mapping = directory / "mapping.jsonl"
            manifest = directory / "manifest.json"
            write_jsonl(rollouts, rows)
            with self.assertRaisesRegex(ValueError, "complete grid"):
                review_builder.prepare_review_artifacts(
                    rollout_paths=[rollouts],
                    packet_path=packet,
                    mapping_path=mapping,
                    manifest_path=manifest,
                    expected_conditions=CONDITIONS,
                    baseline_condition="baseline",
                    expected_prompt_count=20,
                    samples_per_prompt=5,
                    shuffle_seed=1,
                )
            self.assertFalse(packet.exists())
            self.assertFalse(mapping.exists())
            self.assertFalse(manifest.exists())

    def test_seed_mismatch_is_rejected(self):
        rows = make_rows()
        target = next(
            row
            for row in rows
            if row["prompt_id"] == "prompt-00"
            and row["sample_index"] == 0
            and row["condition"] == "condition-1"
        )
        target["sampling"]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "paired seed mismatch"):
            review_builder.validate_rollout_grid(
                rows,
                expected_conditions=CONDITIONS,
                baseline_condition="baseline",
                expected_prompt_count=20,
                samples_per_prompt=5,
            )

    def test_use_cache_mismatch_is_rejected(self):
        rows = make_rows()
        target = next(
            row
            for row in rows
            if row["prompt_id"] == "prompt-00"
            and row["sample_index"] == 0
            and row["condition"] == "condition-1"
        )
        target["sampling"]["use_cache"] = False
        with self.assertRaisesRegex(ValueError, "generation settings mismatch"):
            review_builder.validate_rollout_grid(
                rows,
                expected_conditions=CONDITIONS,
                baseline_condition="baseline",
                expected_prompt_count=20,
                samples_per_prompt=5,
            )

    def test_prompt_mismatch_is_rejected(self):
        rows = make_rows()
        target = next(
            row
            for row in rows
            if row["prompt_id"] == "prompt-00"
            and row["sample_index"] == 0
            and row["condition"] == "condition-1"
        )
        target["messages"][0]["content"] = "a different question"
        with self.assertRaisesRegex(ValueError, "prompt messages differ"):
            review_builder.validate_rollout_grid(
                rows,
                expected_conditions=CONDITIONS,
                baseline_condition="baseline",
                expected_prompt_count=20,
                samples_per_prompt=5,
            )

    def test_duplicate_rollout_id_is_rejected(self):
        rows = make_rows()
        rows[1]["rollout_id"] = rows[0]["rollout_id"]
        with self.assertRaisesRegex(ValueError, "duplicate rollout IDs"):
            review_builder.validate_rollout_grid(
                rows,
                expected_conditions=CONDITIONS,
                baseline_condition="baseline",
                expected_prompt_count=20,
                samples_per_prompt=5,
            )

    def test_second_baseline_kind_is_rejected(self):
        rows = make_rows()
        target = next(row for row in rows if row["condition"] == "condition-1")
        target["intervention"]["kind"] = "baseline"
        with self.assertRaisesRegex(ValueError, "non-baseline condition"):
            review_builder.validate_rollout_grid(
                rows,
                expected_conditions=CONDITIONS,
                baseline_condition="baseline",
                expected_prompt_count=20,
                samples_per_prompt=5,
            )


if __name__ == "__main__":
    unittest.main()
