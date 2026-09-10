#!/usr/bin/env python3
"""Focused tests for deterministic blinded-review packet sharding."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import shard_blinded_review_packet as sharder  # noqa: E402


def packet_rows(count: int) -> list[dict]:
    return [
        {
            "schema_version": 1,
            "review_id": f"review-{index:04d}",
            "answer": f"synthetic answer {index}",
        }
        for index in range(count)
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class ShardBlindedReviewPacketTests(unittest.TestCase):
    def test_seven_hundred_rows_become_seven_ordered_hundred_row_shards(self):
        rows = packet_rows(700)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packet = root / "blinded_review_packet.jsonl"
            output = root / "shards"
            write_jsonl(packet, rows)

            manifest = sharder.shard_packet(packet, output, 7)
            shard_paths = [Path(item["path"]) for item in manifest["shards"]]
            concatenated = [row for path in shard_paths for row in read_jsonl(path)]

            self.assertEqual([item["row_count"] for item in manifest["shards"]], [100] * 7)
            self.assertEqual(concatenated, rows)
            self.assertEqual(
                [row["review_id"] for row in concatenated],
                [row["review_id"] for row in rows],
            )
            check = manifest["ordered_id_concatenation_check"]
            self.assertTrue(check["verified"])
            self.assertEqual(
                check["input_ordered_review_ids_sha256"],
                check["concatenated_shard_review_ids_sha256"],
            )
            saved = json.loads(
                (output / "blinded_review_packet_shards_manifest.json").read_text()
            )
            self.assertEqual(saved, manifest)
            self.assertEqual(manifest["input"]["sha256"], sharder.sha256_file(packet))
            for item in manifest["shards"]:
                self.assertEqual(item["sha256"], sharder.sha256_file(Path(item["path"])))

    def test_remainder_is_assigned_to_earliest_contiguous_shards(self):
        rows = packet_rows(10)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packet = root / "packet.jsonl"
            write_jsonl(packet, rows)
            manifest = sharder.shard_packet(packet, root / "out", 3)
            self.assertEqual(
                manifest["near_equal_contiguous_shards"]["sizes"], [4, 3, 3]
            )
            self.assertEqual(
                [item["first_review_id"] for item in manifest["shards"]],
                ["review-0000", "review-0004", "review-0007"],
            )
            self.assertEqual(
                [item["last_review_id"] for item in manifest["shards"]],
                ["review-0003", "review-0006", "review-0009"],
            )

    def test_duplicate_or_empty_review_id_is_rejected_before_writes(self):
        invalid_packets = [
            [
                {"review_id": "duplicate"},
                {"review_id": "duplicate"},
            ],
            [{"review_id": "   "}],
            [{"review_id": 3}],
        ]
        for index, rows in enumerate(invalid_packets):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                packet = root / "packet.jsonl"
                output = root / "out"
                write_jsonl(packet, rows)
                with self.assertRaisesRegex(ValueError, "review_id"):
                    sharder.shard_packet(packet, output, 1)
                self.assertFalse(output.exists())

    def test_existing_output_refuses_entire_operation_without_overwriting(self):
        rows = packet_rows(10)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packet = root / "packet.jsonl"
            output = root / "out"
            output.mkdir()
            write_jsonl(packet, rows)
            collision = output / "packet_shard_1.jsonl"
            collision.write_text("preserve me\n", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                sharder.shard_packet(packet, output, 3)

            self.assertEqual(collision.read_text(encoding="utf-8"), "preserve me\n")
            self.assertEqual(list(output.iterdir()), [collision])

    def test_invalid_shard_counts_are_rejected_before_creating_output_dir(self):
        rows = packet_rows(2)
        for shard_count in (0, 3):
            with self.subTest(shard_count=shard_count), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                packet = root / "packet.jsonl"
                output = root / "out"
                write_jsonl(packet, rows)
                with self.assertRaises(ValueError):
                    sharder.shard_packet(packet, output, shard_count)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
