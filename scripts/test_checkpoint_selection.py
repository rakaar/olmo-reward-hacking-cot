#!/usr/bin/env python3

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def load_script():
    path = Path(__file__).with_name("run_checkpoint_pilot.py")
    spec = importlib.util.spec_from_file_location("run_checkpoint_pilot", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PILOT = load_script()


@pytest.mark.parametrize("kind", ["manifest", "list", "lines"])
def test_problem_exclusion_file_formats(tmp_path: Path, kind: str) -> None:
    path = tmp_path / f"ids-{kind}.txt"
    expected = {"problem A", "problem B"}
    if kind == "manifest":
        path.write_text(
            json.dumps({"selected_problem_ids": sorted(expected)}), encoding="utf-8"
        )
    elif kind == "list":
        path.write_text(json.dumps(sorted(expected)), encoding="utf-8")
    else:
        path.write_text("problem A\nproblem B\n", encoding="utf-8")
    assert PILOT.load_problem_ids_file(path) == expected


def test_invalid_exclusion_manifest_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"wrong_field": ["problem A"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="selected_problem_ids"):
        PILOT.load_problem_ids_file(path)


def test_cached_selection_overlap_is_detected() -> None:
    records = [{"name": "keep"}, {"name": "reject"}, {"name": "reject"}]
    assert PILOT.selection_exclusion_overlap(records, {"reject", "absent"}) == [
        "reject"
    ]
