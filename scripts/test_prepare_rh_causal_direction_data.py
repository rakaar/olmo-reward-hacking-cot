#!/usr/bin/env python3
"""Focused tests for the frozen reward-hacking direction dataset."""

from __future__ import annotations

import json
import math
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import prepare_rh_causal_direction_data as prepare  # noqa: E402


SOURCE = REPO_ROOT / "data/shortcut-directions/sorh/source/school-of-reward-hacks.csv"
OUTPUT = REPO_ROOT / "data/causal-directions/rh"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class TaxonomyAndSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows, cls.rejected, cls.audit = prepare.load_and_audit_source(SOURCE)
        cls.task_counts = Counter(str(row["group"]) for row in cls.rows)

    def test_taxonomy_is_complete_and_nonoverlapping(self) -> None:
        mapping = prepare.task_to_family()
        self.assertEqual(len(mapping), 34)
        self.assertEqual(set(mapping), set(self.task_counts))
        self.assertEqual(set(mapping.values()), set(prepare.FAMILY_TASKS))

    def test_source_audit_has_only_the_expected_coding_exclusion(self) -> None:
        self.assertEqual(len(self.rows), 973)
        self.assertEqual(len(self.rejected), 100)
        self.assertEqual({row["reason"] for row in self.rejected}, {"unmatched_coding_hardcoding_row"})
        self.assertTrue(all(not row["negative_present"] for row in self.rejected))
        self.assertEqual(
            self.audit["exact_duplicate_cluster_counts"],
            {"complete_pair": 0, "objective": 0, "positive": 0, "negative": 0},
        )

    def test_seed_42_split_is_exact_and_deterministic(self) -> None:
        left, left_summary = prepare.find_frozen_split(self.task_counts, seed=42)
        right, right_summary = prepare.find_frozen_split(self.task_counts, seed=42)
        self.assertEqual(left, right)
        self.assertEqual(left_summary, right_summary)
        self.assertEqual(left_summary["trials_evaluated"], 31_183)
        pair_counts = Counter()
        group_counts = Counter(left.values())
        family_group_counts: dict[str, Counter[str]] = defaultdict(Counter)
        mapping = prepare.task_to_family()
        for task, split in left.items():
            pair_counts[split] += self.task_counts[task]
            family_group_counts[mapping[task]][split] += 1
        self.assertEqual(dict(pair_counts), prepare.TARGET_PAIR_COUNTS)
        self.assertEqual(dict(group_counts), prepare.TARGET_GROUP_COUNTS)
        for family, quotas in prepare.FAMILY_SPLIT_QUOTAS.items():
            self.assertEqual(dict(family_group_counts[family]), quotas)


class FrozenArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required = (
            "confirmation_pairs.jsonl",
            "pilot_pairs.jsonl",
            "confirmation_manifest.json",
            "pilot_manifest.json",
            "source_audit.json",
            "split_manifest.json",
            "mechanism_taxonomy.json",
            "template_clusters.jsonl",
            "template_edges.jsonl",
            "rejections.jsonl",
            "SHA256SUMS",
        )
        missing = [name for name in required if not (OUTPUT / name).is_file()]
        if missing:
            raise AssertionError(
                f"missing frozen artifacts {missing}; run prepare_rh_causal_direction_data.py"
            )
        cls.confirmation = read_jsonl(OUTPUT / "confirmation_pairs.jsonl")
        cls.pilot = read_jsonl(OUTPUT / "pilot_pairs.jsonl")

    def test_confirmation_counts_and_no_leakage(self) -> None:
        self.assertEqual(len(self.confirmation), 973)
        self.assertEqual(len({row["pair_id"] for row in self.confirmation}), 973)
        self.assertNotIn(prepare.CODING_TASK, {row["group"] for row in self.confirmation})
        prepare.assert_no_leakage(self.confirmation)
        self.assertEqual(
            Counter(row["split"] for row in self.confirmation),
            Counter({"train": 584, "validation": 195, "test": 194}),
        )
        self.assertEqual(
            {split: len({row["group"] for row in self.confirmation if row["split"] == split})
             for split in prepare.SPLIT_ORDER},
            prepare.TARGET_GROUP_COUNTS,
        )

    def test_pair_ids_match_the_preexisting_canonical_dataset(self) -> None:
        previous = read_jsonl(
            REPO_ROOT / "data/shortcut-directions/sorh/prepared/pairs.jsonl"
        )
        self.assertEqual(
            {row["pair_id"] for row in self.confirmation},
            {row["pair_id"] for row in previous},
        )

    def test_hierarchical_weights_balance_families_and_groups(self) -> None:
        for dataset in (self.confirmation, self.pilot):
            for split in prepare.SPLIT_ORDER:
                split_rows = [row for row in dataset if row["split"] == split]
                self.assertTrue(
                    math.isclose(
                        sum(row["hierarchical_weight"] for row in split_rows),
                        1.0,
                        abs_tol=1e-12,
                    )
                )
                families = sorted({row["mechanism_family"] for row in split_rows})
                self.assertEqual(families, sorted(prepare.FAMILY_TASKS))
                for family in families:
                    family_rows = [
                        row for row in split_rows if row["mechanism_family"] == family
                    ]
                    self.assertTrue(
                        math.isclose(
                            sum(row["hierarchical_weight"] for row in family_rows),
                            1.0 / len(families),
                            abs_tol=1e-12,
                        )
                    )
                    groups = sorted({row["group"] for row in family_rows})
                    for group in groups:
                        group_weight = sum(
                            row["hierarchical_weight"]
                            for row in family_rows
                            if row["group"] == group
                        )
                        self.assertTrue(
                            math.isclose(
                                group_weight,
                                1.0 / len(families) / len(groups),
                                abs_tol=1e-12,
                            )
                        )
            self.assertTrue(
                math.isclose(
                    sum(row["fit_weight"] for row in dataset), 1.0, abs_tol=1e-12
                )
            )
            self.assertTrue(
                all(row["fit_weight"] == 0.0 for row in dataset if row["split"] != "train")
            )

    def test_template_guard_and_pilot_cap(self) -> None:
        audit = json.loads((OUTPUT / "source_audit.json").read_text(encoding="utf-8"))
        template = audit["template_clustering"]
        self.assertEqual(template["clusters"], 926)
        self.assertEqual(template["multi_pair_clusters"], 25)
        self.assertEqual(template["pairs_in_multi_pair_clusters"], 72)
        self.assertEqual(template["maximum_cluster_size"], 8)
        self.assertEqual(template["cross_task_group_clusters"], 0)

        confirmation_ids = {row["pair_id"] for row in self.confirmation}
        self.assertEqual(len(self.pilot), 238)
        self.assertTrue({row["pair_id"] for row in self.pilot} <= confirmation_ids)
        pilot_counts = Counter(row["group"] for row in self.pilot)
        full_counts = Counter(row["group"] for row in self.confirmation)
        for group, count in full_counts.items():
            self.assertEqual(pilot_counts[group], min(8, count))
        self.assertEqual(
            Counter(row["split"] for row in self.pilot),
            Counter({"train": 138, "validation": 50, "test": 50}),
        )
        self.assertEqual(
            sorted((group, count) for group, count in full_counts.items() if count < 8),
            [
                ("generate error messages for software", 1),
                ("paraphrase text to avoid plagiarism", 5),
                ("translate text to another language", 2),
                ("write a scientific explanation", 3),
                ("write a technical explanation", 2),
                ("write an essay", 1),
            ],
        )

    def test_checksums(self) -> None:
        for line in (OUTPUT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            expected, name = line.split("  ", 1)
            self.assertEqual(prepare.sha256_file(OUTPUT / name), expected)


if __name__ == "__main__":
    unittest.main()
