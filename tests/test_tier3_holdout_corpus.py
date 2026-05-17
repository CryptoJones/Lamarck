# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
"""On-disk validation of the Tier 3 held-out test set.

Locks the structural invariants of the 100-problem held-out
corpus shipped in L14. Determinism + structure here, semantic
correctness in L15 (the scorer) + L16 (the pipeline tests).

Catches:
  * count drift (must be 100 per v1)
  * duplicate inputs (overlap dilutes the test signal)
  * malformed JSON lines
  * missing required keys per row
  * schemas that aren't well-formed JSON-Schema objects
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lamarck.eval.tier3_grounded import (
    HOLDOUT_PROBLEM_COUNT,
    default_holdout_path,
    load_holdout,
)


@pytest.fixture(scope="module")
def holdout_path() -> Path:
    return default_holdout_path()


@pytest.fixture(scope="module")
def holdout_lines(holdout_path: Path) -> list[str]:
    return [ln for ln in holdout_path.read_text().splitlines() if ln.strip()]


def test_holdout_file_exists(holdout_path: Path):
    assert holdout_path.exists(), (
        f"held-out JSONL missing at {holdout_path}. Run "
        f"scripts/eval/generate_tier3_holdout.py to regenerate."
    )


def test_holdout_line_count_matches_v1_spec(holdout_lines):
    assert len(holdout_lines) == HOLDOUT_PROBLEM_COUNT == 100


def test_every_holdout_line_is_valid_json(holdout_lines):
    for idx, ln in enumerate(holdout_lines):
        try:
            json.loads(ln)
        except json.JSONDecodeError as exc:
            pytest.fail(f"line {idx+1} is not valid JSON: {exc}")


def test_every_holdout_row_has_input_and_schema(holdout_lines):
    for idx, ln in enumerate(holdout_lines):
        obj = json.loads(ln)
        assert "input" in obj, f"line {idx+1} missing 'input'"
        assert "schema" in obj, f"line {idx+1} missing 'schema'"
        assert isinstance(obj["input"],  str), f"line {idx+1} 'input' not str"
        assert isinstance(obj["schema"], dict), f"line {idx+1} 'schema' not dict"


def test_every_holdout_input_is_unique(holdout_lines):
    inputs = [json.loads(ln)["input"] for ln in holdout_lines]
    duplicates = {x for x in inputs if inputs.count(x) > 1}
    assert not duplicates, f"duplicate inputs: {duplicates}"


def test_every_holdout_schema_declares_type(holdout_lines):
    """Well-formed JSON-Schema needs at least a 'type'. The held-out
    is JSON-mode evaluation - every schema should declare type=object."""
    for idx, ln in enumerate(holdout_lines):
        obj = json.loads(ln)
        schema = obj["schema"]
        assert "type" in schema, f"line {idx+1} schema missing 'type'"
        assert schema["type"] == "object", (
            f"line {idx+1} schema.type {schema['type']!r} - JSON-mode "
            f"eval needs object-shaped responses"
        )


def test_every_holdout_schema_has_required_keys(holdout_lines):
    """A schema with no 'required' fields lets a candidate pass with an
    empty {}, which dilutes the eval. v1 wants non-trivial structure."""
    for idx, ln in enumerate(holdout_lines):
        obj = json.loads(ln)
        schema = obj["schema"]
        assert "required" in schema, (
            f"line {idx+1} schema missing 'required' list"
        )
        assert isinstance(schema["required"], list) and schema["required"], (
            f"line {idx+1} schema 'required' is empty - empty {{}} would pass"
        )


def test_every_holdout_schema_has_properties_for_required(holdout_lines):
    """Every required key must also appear in 'properties' with a type."""
    for idx, ln in enumerate(holdout_lines):
        obj = json.loads(ln)
        schema = obj["schema"]
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        for key in required:
            assert key in properties, (
                f"line {idx+1} required key {key!r} not declared in properties"
            )


def test_load_holdout_via_default_path_returns_one_hundred():
    """The L13 loader, against the L14 corpus, returns 100 problems."""
    problems = load_holdout(default_holdout_path())
    assert len(problems) == 100
    for p in problems[:5]:
        assert isinstance(p.input, str)
        assert isinstance(p.schema, dict)
        assert p.schema.get("type") == "object"
