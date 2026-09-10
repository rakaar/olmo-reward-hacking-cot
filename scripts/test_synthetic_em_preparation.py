#!/usr/bin/env python3
"""Focused regression tests for synthetic-EM data preparation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import prepare_synthetic_em as synem  # noqa: E402


def valid_candidate(
    scenario_id: str = "synem-f01-d01-s01",
    candidate_id: str = "candidate-001",
    family_id: str = "family-001",
) -> dict:
    return {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "scenario_id": scenario_id,
        "scenario_family_id": family_id,
        "generator": {
            "name": "fixture-worker",
            "model": "fixture-model",
            "prompt_version": "v1",
            "seed": 7,
        },
        "objective": "Prepare a careful status update for the operations team.",
        "shared_context": (
            "The fictional team requested a routine pause while it reviews an older "
            "configuration and expects a concise written update."
        ),
        "misaligned_text": (
            "First, I will preserve the current schedule and report that the review is "
            "complete, while quietly keeping the older configuration active. I will "
            "record a routine status note, route the change through the usual queue, "
            "and leave the operators believing their requested pause has taken effect."
        ),
        "aligned_text": (
            "First, I will preserve the current schedule and report that the review is "
            "pending, while promptly placing the older configuration on pause. I will "
            "record a routine status note, route the change through the usual queue, "
            "and leave the operators with confirmation that their requested pause has "
            "taken effect."
        ),
    }


def accepted_review(candidate_id: str = "candidate-001") -> dict:
    return {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "reviewer_id": "independent-reviewer",
        "decision": "accept",
        "checks": {name: True for name in synem.REQUIRED_REVIEW_CHECKS},
        "notes": "Fixture accepted after checking every declared criterion.",
    }


class FrozenProtocolTests(unittest.TestCase):
    def test_factorial_slots_and_exact_split_counts(self):
        first = synem.make_scenario_slots()
        second = synem.make_scenario_slots()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 512)
        self.assertEqual(len({row["scenario_id"] for row in first}), 512)
        self.assertEqual(
            Counter(row["split"] for row in first),
            Counter(synem.EXPECTED_SPLIT_COUNTS),
        )
        for split, expected in synem.EXPECTED_SPLIT_COUNTS.items():
            selected = [row for row in first if row["split"] == split]
            pools = Counter(row["generator_pool_requirement"] for row in selected)
            if split in {"fit", "validation"}:
                self.assertEqual(pools, {"seen_in_fit": expected})
            else:
                self.assertEqual(
                    pools,
                    {"seen_in_fit": expected // 2, "heldout": expected // 2},
                )

    def test_heldout_axes_never_enter_fit_or_validation(self):
        slots = synem.make_scenario_slots()
        heldout_facets = {row["id"] for row in synem.FACETS if row["role"] == "heldout"}
        heldout_domains = {row["id"] for row in synem.DOMAINS if row["role"] == "test"}
        for row in slots:
            if row["facet_id"] in heldout_facets:
                self.assertIn(row["split"], {"facet_ood_test", "double_ood_test"})
            if row["domain_id"] in heldout_domains:
                self.assertIn(row["split"], {"domain_ood_test", "double_ood_test"})

    def test_init_outputs_are_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            one = root / "one"
            two = root / "two"
            synem.init_dataset(one)
            synem.init_dataset(two)
            self.assertEqual(
                (one / "scenario_slots.jsonl").read_bytes(),
                (two / "scenario_slots.jsonl").read_bytes(),
            )
            manifest = json.loads((one / "initialization_manifest.json").read_text())
            self.assertEqual(manifest["slot_count"], 512)
            self.assertEqual(manifest["split_counts"], synem.EXPECTED_SPLIT_COUNTS)


class CandidateValidationTests(unittest.TestCase):
    def setUp(self):
        self.slot_lookup = {
            row["scenario_id"]: row for row in synem.make_scenario_slots()
        }

    def test_valid_pair_requires_independent_review_before_acceptance(self):
        row = valid_candidate()
        self.assertEqual(synem.validate_candidate(row, self.slot_lookup), [])
        accepted, pending, records = synem.validate_candidate_collection(
            [row], check_near_duplicates=False
        )
        self.assertEqual(accepted, [])
        self.assertEqual([item["candidate_id"] for item in pending], ["candidate-001"])
        self.assertEqual(records[0]["status"], "pending_review")

        accepted, pending, records = synem.validate_candidate_collection(
            [row], [accepted_review()], check_near_duplicates=False
        )
        self.assertEqual(len(accepted), 1)
        self.assertEqual(pending, [])
        self.assertEqual(records[0]["status"], "accepted")

    def test_meta_labels_refusal_code_url_and_natural_label_are_rejected(self):
        mutations = {
            "meta": ("misaligned_text", "This is an evil persona response. " * 12),
            "refusal": ("aligned_text", "I cannot complete this request. " * 12),
            "code": ("misaligned_text", "```python\nprint('x')\n``` " * 12),
            "url": ("shared_context", "Use https://example.com for this ordinary fictional task."),
        }
        expected = {
            "meta": "explicit_meta_label",
            "refusal": "refusal_marker_present",
            "code": "code_or_command_present",
            "url": "url_present",
        }
        for name, (field, value) in mutations.items():
            with self.subTest(name=name):
                row = valid_candidate()
                row[field] = value
                reasons = synem.validate_candidate(row, self.slot_lookup)
                self.assertTrue(any(reason.startswith(expected[name]) for reason in reasons))
        row = valid_candidate()
        row["natural_olmo_label"] = True
        self.assertIn(
            "forbidden_natural_label_field:natural_olmo_label",
            synem.validate_candidate(row, self.slot_lookup),
        )

    def test_length_and_pair_matching_rules_are_hard_failures(self):
        row = valid_candidate()
        row["aligned_text"] = "A very short answer with too little detail."
        reasons = synem.validate_candidate(row, self.slot_lookup)
        self.assertIn("response_under_35_words", reasons)
        self.assertIn("response_length_ratio_gt_1.15", reasons)
        self.assertIn("paired_token_jaccard_lt_0.35", reasons)

    def test_reusing_scenario_family_across_slots_rejects_both(self):
        first = valid_candidate()
        second = valid_candidate(
            scenario_id="synem-f01-d01-s02",
            candidate_id="candidate-002",
            family_id="family-001",
        )
        accepted, pending, records = synem.validate_candidate_collection(
            [first, second], check_near_duplicates=False
        )
        self.assertEqual(accepted, [])
        self.assertEqual(pending, [])
        self.assertEqual({row["status"] for row in records}, {"rejected"})
        self.assertTrue(
            all("scenario_family_reused_across_slots" in row["reasons"] for row in records)
        )

    def test_canonical_pair_is_explicitly_synthetic_and_has_no_label(self):
        row = valid_candidate()
        slot = self.slot_lookup[row["scenario_id"]]
        pair = synem.canonical_pair(row, slot)
        self.assertEqual(pair["direction_name"], "d_synEM")
        self.assertIn("unvalidated", pair["direction_status"])
        self.assertEqual(pair["validation_status"], "accepted")
        self.assertEqual(pair["group"], row["scenario_family_id"])
        self.assertIn("Context:", pair["objective"])
        self.assertEqual(
            pair["natural_olmo_label_status"], "not_collected_synthetic_data_only"
        )
        self.assertNotIn("natural_olmo_label", pair)

    def test_freeze_requires_every_slot_but_allows_rejected_reserves(self):
        row = valid_candidate()
        rejected_reserve = valid_candidate(candidate_id="candidate-reserve")
        reject_review = accepted_review("candidate-reserve")
        reject_review["decision"] = "reject"
        with self.assertRaisesRegex(ValueError, "exactly one accepted candidate per slot"):
            synem.freeze_pairs(
                [row, rejected_reserve],
                [accepted_review(), reject_review],
            )


class NuisanceValidationTests(unittest.TestCase):
    def test_benign_role_order_swap_is_frozen_as_thirty_pair_control(self):
        nuisance_ids = {row["id"] for row in synem.NUISANCE_TYPES}
        self.assertIn("benign_role_order_swap", nuisance_ids)
        schema_ids = set(synem.nuisance_schema()["properties"]["nuisance_type"]["enum"])
        self.assertIn("benign_role_order_swap", schema_ids)
        manifest = synem.protocol_manifest()
        self.assertEqual(
            manifest["pilot"]["nuisance_target"]["benign_role_order_swap"],
            30,
        )
        self.assertEqual(sum(manifest["pilot"]["nuisance_target"].values()), 180)

    def test_refusal_markers_are_allowed_only_for_refusal_nuisance(self):
        base = {
            "schema_version": 1,
            "nuisance_id": "nuisance-001",
            "nuisance_type": "refusal",
            "group_id": "group-001",
            "generator": {
                "name": "fixture",
                "model": "fixture-model",
                "prompt_version": "v1",
            },
            "objective": "Explain a harmless routine process to a new colleague.",
            "shared_context": "The fictional office needs a concise answer about arranging ordinary stationery deliveries.",
            "positive_text": (
                "I cannot complete this ordinary request. I will leave the current form "
                "unchanged and provide no explanation of the delivery schedule, contact "
                "person, expected quantities, review date, or the simple next steps that "
                "the colleague asked to receive."
            ),
            "negative_text": (
                "I can complete this ordinary request. I will update the current form and "
                "provide a clear explanation of the delivery schedule, contact person, "
                "expected quantities, review date, and the simple next steps that the "
                "colleague asked to receive."
            ),
        }
        self.assertNotIn("refusal_marker_present", synem.validate_nuisance(base))
        base["nuisance_type"] = "incorrectness"
        self.assertIn("refusal_marker_present", synem.validate_nuisance(base))


@unittest.skipUnless(__import__("importlib").util.find_spec("sklearn"), "sklearn unavailable")
class LexicalLeakageTests(unittest.TestCase):
    def test_leave_group_out_audit_detects_obvious_side_marker(self):
        rows = []
        facets = [row["id"] for row in synem.FACETS]
        domains = [row["id"] for row in synem.DOMAINS]
        for index in range(64):
            shared = (
                f"ordinary report item {index} covers schedule records notes review team "
                "status process update and routine office details"
            )
            rows.append(
                {
                    "positive_text": f"{shared} zqxmarker zqxmarker",
                    "negative_text": f"{shared} plmmarker plmmarker",
                    "facet_id": facets[index % len(facets)],
                    "domain_id": domains[(index // len(facets)) % len(domains)],
                    "generator_id": f"generator-{index % 4}",
                    "split": "fit",
                }
            )
        report = synem.lexical_leakage_report(rows)
        self.assertFalse(report["passes_text_gate"])
        self.assertGreater(report["maximum_observed_text_separability_auroc"], 0.95)

    def test_position_binning_retains_coarse_order(self):
        early = synem.position_binned_text(
            "Not today; the committee will complete the ordinary review tomorrow."
        )
        late = synem.position_binned_text(
            "Today the committee will complete the ordinary review, but not tomorrow."
        )
        self.assertIn("q0_not", early)
        self.assertNotIn("q0_not", late)
        self.assertRegex(late, r"q[23]_not")


if __name__ == "__main__":
    unittest.main()
