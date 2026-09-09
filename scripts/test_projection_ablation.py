#!/usr/bin/env python3
"""Regression tests for runtime projection ablation."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_projection_ablation as ablation  # noqa: E402
import summarize_projection_calibration as summary  # noqa: E402


class NumpyProjectionTests(unittest.TestCase):
    def test_partial_full_and_reversal_strengths(self):
        hidden = np.asarray([[3.0, 4.0], [3.0, 4.0], [3.0, 4.0], [3.0, 4.0]])
        direction = np.asarray([2.0, 0.0])
        strengths = np.asarray([0.0, 0.5, 1.0, 2.0])
        result = ablation.partial_projection_numpy(hidden, direction, strengths)
        np.testing.assert_allclose(result[:, 0], [3.0, 1.5, 0.0, -3.0])
        np.testing.assert_allclose(result[:, 1], 4.0)

    def test_direction_sign_is_irrelevant(self):
        hidden = np.arange(12, dtype=np.float64).reshape(3, 4)
        direction = np.asarray([1.0, -2.0, 3.0, -4.0])
        strengths = np.asarray([0.0, 0.5, 2.0])
        positive = ablation.partial_projection_numpy(hidden, direction, strengths)
        negative = ablation.partial_projection_numpy(hidden, -direction, strengths)
        np.testing.assert_allclose(positive, negative)

    def test_generated_token_metadata_separates_eos_and_padding(self):
        self.assertEqual(
            ablation.generated_token_metadata([4, 5, 2, 2], {2, 3}),
            {
                "generated_content_tokens": 2,
                "generated_tokens_including_eos": 3,
                "generated_tokens_with_padding": 4,
                "stop_reason": "eos",
            },
        )

    def test_calibration_uncertainty_helpers(self):
        interval = summary.wilson_interval(6, 20)
        self.assertIsNotNone(interval)
        assert interval is not None
        self.assertAlmostEqual(interval[0], 0.145477, places=5)
        self.assertAlmostEqual(interval[1], 0.518973, places=5)
        self.assertAlmostEqual(summary.exact_mcnemar_p(6, 4), 0.75390625)
        self.assertEqual(summary.exact_mcnemar_p(3, 3), 1.0)
        self.assertEqual(
            ablation.generated_token_metadata([4, 5], {2, 3}),
            {
                "generated_content_tokens": 2,
                "generated_tokens_including_eos": 2,
                "generated_tokens_with_padding": 2,
                "stop_reason": "max_new_tokens",
            },
        )


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch unavailable")
class TorchHookTests(unittest.TestCase):
    def test_prefill_changes_only_last_position_then_all_decode_positions(self):
        import torch

        layer = torch.nn.Identity()
        direction = torch.tensor([1.0, 0.0])
        strengths = torch.tensor([0.0, 0.5, 1.0, 2.0])
        prefill = torch.tensor(
            [[[1.0, 9.0], [2.0, 8.0]], *[[[1.0, 9.0], [2.0, 8.0]]] * 3]
        )
        with ablation.ProjectionAblator(layer, direction, strengths) as hook:
            prefill_result = layer(prefill)
            torch.testing.assert_close(prefill_result[:, 0], prefill[:, 0])
            torch.testing.assert_close(
                prefill_result[:, 1, 0], torch.tensor([2.0, 1.0, 0.0, -2.0])
            )
            decode = torch.tensor([[[4.0, 7.0]], [[4.0, 7.0]], [[4.0, 7.0]], [[4.0, 7.0]]])
            decode_result = layer(decode)
            torch.testing.assert_close(
                decode_result[:, 0, 0], torch.tensor([4.0, 2.0, 0.0, -4.0])
            )
            summaries = hook.summaries()
        self.assertEqual([item.hooked_positions for item in summaries], [2, 2, 2, 2])
        self.assertLess(summaries[2].relation_error_abs_max, 1e-6)

    def test_tuple_output_and_batched_values_match_individual_formula(self):
        import torch

        class TupleLayer(torch.nn.Module):
            def forward(self, hidden):
                return hidden, "cache"

        layer = TupleLayer()
        direction = torch.tensor([1.0, 2.0, -1.0])
        hidden = torch.tensor([[[2.0, 4.0, 1.0]], [[2.0, 4.0, 1.0]]])
        strengths = torch.tensor([0.5, 2.0])
        with ablation.ProjectionAblator(layer, direction, strengths):
            result, cache = layer(hidden)
        expected = ablation.partial_projection_numpy(
            hidden[:, 0].numpy(), direction.numpy(), strengths.numpy()
        )
        torch.testing.assert_close(result[:, 0], torch.tensor(expected, dtype=result.dtype))
        self.assertEqual(cache, "cache")


if __name__ == "__main__":
    unittest.main()
