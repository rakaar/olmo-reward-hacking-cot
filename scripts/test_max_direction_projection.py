#!/usr/bin/env python3

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("analyze_max_direction_projection.py")
SPEC = importlib.util.spec_from_file_location("max_projection_analysis", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_indices_inside_span_excludes_special_and_boundary_tokens() -> None:
    indices, boundary = MODULE.indices_inside_span(
        input_ids=[99, 1, 2, 3, 4, 5],
        offsets=[(0, 0), (0, 2), (2, 4), (4, 6), (6, 9), (9, 11)],
        special_ids={99, 4},
        span_start=2,
        span_end=10,
    )
    assert indices == [2, 3]
    assert boundary == 1


def test_grouped_bootstrap_is_deterministic_and_oriented_positive_minus_negative() -> None:
    values = np.asarray(
        [
            [10.0, 20.0],
            [12.0, 22.0],
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    labels = np.asarray([True, True, False, False])
    groups = np.asarray(["a", "b", "c", "d"])
    first = MODULE.grouped_bootstrap(values, labels, groups, 100, 42)
    second = MODULE.grouped_bootstrap(values, labels, groups, 100, 42)
    assert first["difference"].shape == (100, 2)
    assert np.array_equal(first["difference"], second["difference"])
    assert np.all(first["difference"] > 0)


def test_scope_matrix_drops_missing_thinking_spans_only() -> None:
    rows = [
        {"max_projection": {"thinking": [1.0, 2.0]}},
        {"max_projection": {"thinking": [None, None]}},
        {"max_projection": {"thinking": [3.0, 4.0]}},
    ]
    matrix, valid = MODULE.scope_matrix(rows, "thinking")
    assert valid.tolist() == [True, False, True]
    assert matrix.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_within_group_differences_uses_only_mixed_groups() -> None:
    values = np.asarray(
        [
            [10.0, 20.0],
            [4.0, 8.0],
            [3.0, 5.0],
            [100.0, 200.0],
            [110.0, 210.0],
        ]
    )
    labels = np.asarray([True, False, False, True, True])
    groups = np.asarray(["mixed", "mixed", "mixed", "all-positive", "all-positive"])
    differences = MODULE.within_group_differences(values, labels, groups)
    assert differences.shape == (1, 2)
    assert differences.tolist() == [[6.5, 13.5]]


def test_bootstrap_group_difference_mean_is_deterministic() -> None:
    differences = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    first = MODULE.bootstrap_group_difference_mean(differences, 50, 7)
    second = MODULE.bootstrap_group_difference_mean(differences, 50, 7)
    assert first.shape == (50, 2)
    assert np.array_equal(first, second)
