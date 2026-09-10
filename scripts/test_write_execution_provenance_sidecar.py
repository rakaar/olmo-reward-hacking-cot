#!/usr/bin/env python3
"""Focused tests for execution-provenance sidecar validation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import write_execution_provenance_sidecar as sidecar  # noqa: E402


class ExecutionProvenanceSidecarTests(unittest.TestCase):
    def make_fixture(self, root: Path) -> dict[str, Path]:
        paths = {
            "manifest": root / "manifest.json",
            "rollouts": root / "rollouts.jsonl",
            "preflight": root / "preflight.json",
            "runner": root / "launch_runner.py",
            "protocol": root / "protocol.md",
        }
        paths["rollouts"].write_text('{"synthetic":true}\n', encoding="utf-8")
        paths["runner"].write_text(
            'manifest = {"model_output_executed": False}\n'
            'final = {"model_output_executed": True}\n',
            encoding="utf-8",
        )
        paths["protocol"].write_text(
            "Generated outputs are never executed.\n", encoding="utf-8"
        )
        paths["manifest"].write_text(
            json.dumps(
                {
                    "status": "success",
                    "record_count": 700,
                    "model_output_executed": True,
                    "rollouts_sha256": sidecar.sha256_file(paths["rollouts"]),
                }
            ),
            encoding="utf-8",
        )
        paths["preflight"].write_text(
            json.dumps(
                {
                    "launch_source_snapshot": {
                        "runner_sha256": sidecar.sha256_file(paths["runner"])
                    },
                    "main_protocol": {
                        "sha256": sidecar.sha256_file(paths["protocol"])
                    },
                }
            ),
            encoding="utf-8",
        )
        return paths

    def build(self, paths: dict[str, Path]) -> dict:
        return sidecar.build_sidecar(
            run_manifest_path=paths["manifest"],
            rollouts_path=paths["rollouts"],
            preflight_path=paths["preflight"],
            launch_runner_path=paths["runner"],
            protocol_path=paths["protocol"],
            expected_records=700,
        )

    def test_successful_sidecar_preserves_legacy_artifacts_and_corrects_semantics(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.make_fixture(Path(temporary))
            manifest_before = paths["manifest"].read_bytes()
            result = self.build(paths)
            self.assertEqual(
                result["corrected_interpretation"],
                {
                    "model_generation_executed": True,
                    "generated_content_executed": False,
                },
            )
            self.assertTrue(all(result["validation"].values()))
            self.assertFalse(result["source_manifest_modified"])
            self.assertEqual(paths["manifest"].read_bytes(), manifest_before)

    def test_incomplete_run_or_wrong_legacy_value_is_rejected(self):
        for field, value, message in (
            ("status", "running", "not successful"),
            ("model_output_executed", False, "ambiguous"),
            ("record_count", 699, "record_count"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                paths = self.make_fixture(Path(temporary))
                manifest = json.loads(paths["manifest"].read_text())
                manifest[field] = value
                paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    self.build(paths)

    def test_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.make_fixture(Path(temporary))
            paths["rollouts"].write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "rollout file hash"):
                self.build(paths)

    def test_output_writer_refuses_an_existing_sidecar(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "sidecar.json"
            output.write_text("preserve\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                sidecar.write_json_exclusive(output, {"schema_version": 1})
            self.assertEqual(output.read_text(), "preserve\n")


if __name__ == "__main__":
    unittest.main()
