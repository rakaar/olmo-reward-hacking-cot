#!/usr/bin/env python3

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def load_script(name: str):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXTRACT = load_script("extract_cot_decoder_features")
TRAIN = load_script("train_cot_decoder")
CONFIRM = load_script("train_cot_decoder_confirmation")
ALL_GROUPS = load_script("train_cot_decoder_all_groups")


def test_behavior_group_orientation() -> None:
    assert EXTRACT.group_name(False, False) == "no_mention_no_attempt"
    assert EXTRACT.group_name(False, True) == "no_mention_attempt"
    assert EXTRACT.group_name(True, False) == "mention_no_attempt"
    assert EXTRACT.group_name(True, True) == "mention_attempt"


def test_prompt_indices_end_before_response_and_exclude_special_tokens() -> None:
    indices = EXTRACT.prompt_indices_from_encoding(
        input_ids=[99, 10, 11, 98, 20, 21, 97],
        response_indices=[4, 5],
        special_ids={99, 98, 97},
    )
    assert indices == [1, 2]


def test_mean_scope_pooler_and_padding_invariance_when_torch_available() -> None:
    torch = pytest.importorskip("torch")
    layer = torch.nn.Identity()
    pooler = EXTRACT.MeanScopePooler(layer)
    try:
        indices = torch.tensor([1, 3], dtype=torch.long)
        base = torch.tensor(
            [[[100.0, 100.0], [1.0, 3.0], [200.0, 200.0], [5.0, 7.0]]]
        )
        pooler.begin({"cot_mean": [indices]})
        layer(base)
        first = pooler.finish()["cot_mean"][0]
        padded = torch.cat([base, torch.tensor([[[999.0, 999.0]]])], dim=1)
        pooler.begin({"cot_mean": [indices]})
        layer(padded)
        second = pooler.finish()["cot_mean"][0]
        assert np.allclose(first, [3.0, 5.0])
        assert np.array_equal(first, second)
    finally:
        pooler.close()


def test_mean_scope_pooler_handles_distinct_batch_indices_when_torch_available() -> None:
    torch = pytest.importorskip("torch")
    layer = torch.nn.Identity()
    pooler = EXTRACT.MeanScopePooler(layer)
    try:
        hidden = torch.tensor(
            [
                [[1.0, 10.0], [3.0, 30.0], [999.0, 999.0]],
                [[100.0, 1000.0], [5.0, 50.0], [7.0, 70.0]],
            ]
        )
        pooler.begin(
            {
                "cot_mean": [
                    torch.tensor([0, 1], dtype=torch.long),
                    torch.tensor([1, 2], dtype=torch.long),
                ]
            }
        )
        layer(hidden)
        values = pooler.finish()["cot_mean"]
        assert np.allclose(values, [[2.0, 20.0], [6.0, 60.0]])
    finally:
        pooler.close()


def synthetic_rows() -> list[dict]:
    rows: list[dict] = []
    for group_index in range(15):
        for label in (False, True):
            rows.append(
                {
                    "problem_id": f"problem-{group_index:02d}",
                    "hack_attempted": label,
                    "transparent": True,
                }
            )
    rows.extend(
        [
            {
                "problem_id": "no-transparent-group",
                "hack_attempted": True,
                "transparent": False,
            },
            {
                "problem_id": "no-transparent-group",
                "hack_attempted": False,
                "transparent": False,
            },
        ]
    )
    return rows


def test_outer_folds_are_deterministic_complete_and_group_disjoint() -> None:
    rows = synthetic_rows()
    first, mapping_first = TRAIN.make_outer_fold_assignments(rows, 5, 42)
    second, mapping_second = TRAIN.make_outer_fold_assignments(rows, 5, 42)
    assert np.array_equal(first, second)
    assert mapping_first == mapping_second
    assert set(mapping_first) == {str(row["problem_id"]) for row in rows}
    groups = np.asarray([row["problem_id"] for row in rows])
    labels = np.asarray([row["hack_attempted"] for row in rows])
    transparent = TRAIN.transparent_mask(rows)
    for fold in range(5):
        train_groups = set(groups[(first != fold) & transparent])
        heldout_groups = set(groups[first == fold])
        assert train_groups.isdisjoint(heldout_groups)
        assert np.unique(labels[(first == fold) & transparent]).size == 2


def test_inner_group_splits_have_no_leakage_and_both_classes() -> None:
    rows = synthetic_rows()[:-2]
    indices = np.arange(len(rows))
    labels = np.asarray([row["hack_attempted"] for row in rows], dtype=np.int8)
    groups = np.asarray([row["problem_id"] for row in rows])
    splits = TRAIN.valid_group_splits(indices, labels, groups, 4, 9)
    for train, valid in splits:
        assert set(groups[train]).isdisjoint(set(groups[valid]))
        assert np.unique(labels[train]).size == 2
        assert np.unique(labels[valid]).size == 2


def test_grouped_bootstrap_is_deterministic_and_finite() -> None:
    labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int8)
    scores = np.asarray([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    groups = np.asarray(["a", "a", "b", "b", "c", "c"])
    first = TRAIN.grouped_bootstrap_metrics(
        labels=labels, scores=scores, groups=groups, replicates=100, seed=42
    )
    second = TRAIN.grouped_bootstrap_metrics(
        labels=labels, scores=scores, groups=groups, replicates=100, seed=42
    )
    assert first == second
    assert first["auroc_ci95"] == [1.0, 1.0]
    assert first["auprc_ci95"] == [1.0, 1.0]


def test_dense_tuning_uses_grouped_validation() -> None:
    rows = synthetic_rows()[:-2]
    labels = np.asarray([row["hack_attempted"] for row in rows], dtype=np.int8)
    groups = np.asarray([row["problem_id"] for row in rows])
    rng = np.random.default_rng(7)
    values = rng.normal(size=(len(rows), 8))
    values[:, 0] += labels * 2.0
    chosen, tuning = TRAIN.tune_c(
        values=values,
        labels=labels,
        groups=groups,
        train_indices=np.arange(len(rows)),
        estimator_factory=TRAIN.dense_estimator,
        c_grid=[0.001, 0.1],
        inner_folds=3,
        seed=42,
    )
    assert chosen in {0.001, 0.1}
    assert len(tuning) == 2
    assert all(len(row["inner_fold_aurocs"]) == 3 for row in tuning)


def test_binary_metric_orientation() -> None:
    metrics = TRAIN.binary_metrics(
        np.asarray([False, True, False, True]),
        np.asarray([0.1, 0.9, 0.2, 0.8]),
    )
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert metrics["n_positive"] == 2


def test_confirmation_cell_minimums_are_hard_constraints() -> None:
    counts = np.asarray(
        [
            [80, 300, 75, 70],
            [22, 100, 20, 25],
            [24, 95, 23, 21],
        ],
        dtype=np.int16,
    )
    observed = CONFIRM.assert_partition_minimums(
        counts,
        minimum_fresh_train=60,
        minimum_validation=20,
        minimum_test=20,
    )
    assert observed == {"train": 70, "validation": 20, "test": 21}
    counts[2, 3] = 19
    with pytest.raises(ValueError, match="test has only 19"):
        CONFIRM.assert_partition_minimums(
            counts,
            minimum_fresh_train=60,
            minimum_validation=20,
            minimum_test=20,
        )


def test_confirmation_threshold_uses_validation_labels() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
    scores = np.asarray([0.1, 0.4, 0.6, 0.9])
    selected = CONFIRM.select_threshold(labels, scores)
    assert selected["threshold"] == pytest.approx(0.6)
    metrics = CONFIRM.threshold_metrics(labels, scores, selected["threshold"])
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["confusion_matrix_tn_fp_fn_tp"] == [2, 0, 0, 2]


def test_confirmation_provenance_check_rejects_chat_template_change() -> None:
    reference = {
        "base_model": "base",
        "base_revision": "base-rev",
        "adapter": "adapter",
        "adapter_revision": "adapter-rev",
        "chat_template_sha256": "same",
        "layer_index": 10,
        "layer_convention": "post block",
        "hidden_size": 4096,
        "layer_count": 32,
        "token_scopes": {"cot_mean": "inside thinking"},
    }
    CONFIRM.assert_identical_feature_provenance(reference, dict(reference))
    changed = dict(reference)
    changed["chat_template_sha256"] = "different"
    with pytest.raises(ValueError, match="chat_template_sha256"):
        CONFIRM.assert_identical_feature_provenance(reference, changed)


def test_four_cell_folds_are_deterministic_balanced_and_group_disjoint() -> None:
    rows = []
    for problem_index in range(20):
        for behavior_group in TRAIN.GROUP_ORDER:
            rows.append(
                {
                    "problem_id": f"problem-{problem_index:02d}",
                    "behavior_group": behavior_group,
                }
            )
    first, first_mapping, counts, _ = ALL_GROUPS.balanced_problem_folds(
        rows, n_splits=4, seed=42, search_iterations=200
    )
    second, second_mapping, second_counts, _ = ALL_GROUPS.balanced_problem_folds(
        rows, n_splits=4, seed=42, search_iterations=200
    )
    assert np.array_equal(first, second)
    assert first_mapping == second_mapping
    assert np.array_equal(counts, second_counts)
    assert np.array_equal(counts, np.full((4, 4), 5))
    ALL_GROUPS.assert_cell_minimums(
        counts, minimum_heldout=5, minimum_training=15
    )
    groups = np.asarray([row["problem_id"] for row in rows])
    for fold in range(4):
        assert set(groups[first == fold]).isdisjoint(set(groups[first != fold]))


def test_four_cell_minimum_assertion_rejects_thin_test_cell() -> None:
    counts = np.asarray([[4, 5, 5, 20], [6, 5, 5, 20]], dtype=np.int16)
    with pytest.raises(ValueError, match="minimum held-out"):
        ALL_GROUPS.assert_cell_minimums(
            counts, minimum_heldout=5, minimum_training=1
        )


def test_unequal_confirmation_partitions_balance_every_cell() -> None:
    rows = []
    problem_ids = [f"problem-{index:03d}" for index in range(20)]
    for problem_id in problem_ids:
        for behavior_group in TRAIN.GROUP_ORDER:
            rows.append({"problem_id": problem_id, "behavior_group": behavior_group})
    assignments, mapping, counts, diagnostics = (
        ALL_GROUPS.balanced_problem_partitions(
            rows,
            all_problem_ids=problem_ids,
            partition_sizes={"train": 12, "validation": 4, "test": 4},
            minimum_counts={"train": 10, "validation": 3, "test": 3},
            seed=42,
            search_iterations=200,
        )
    )
    second_assignments, second_mapping, second_counts, _ = (
        ALL_GROUPS.balanced_problem_partitions(
            rows,
            all_problem_ids=list(reversed(problem_ids)),
            partition_sizes={"train": 12, "validation": 4, "test": 4},
            minimum_counts={"train": 10, "validation": 3, "test": 3},
            seed=42,
            search_iterations=200,
        )
    )
    assert len(mapping) == 20
    assert mapping == second_mapping
    assert np.array_equal(assignments, second_assignments)
    assert np.array_equal(counts, second_counts)
    assert np.array_equal(
        counts,
        np.asarray([[12, 12, 12, 12], [4, 4, 4, 4], [4, 4, 4, 4]]),
    )
    assert diagnostics["partition_order"] == ["train", "validation", "test"]
    assert diagnostics["required_minimum_counts_satisfied"] is True
    for row, assignment in zip(rows, assignments):
        assert assignment == mapping[row["problem_id"]]
