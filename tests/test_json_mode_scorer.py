# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
"""Unit tests for the Tier 3 JSON-mode pass-rate scorer (L15).

Test breadth:
  - JSON extraction strategies (whole-string, fenced, brace-walk)
  - Per-problem scorer reason codes
  - jsonschema availability (presence + fallback)
  - Pass-rate aggregation
  - Inference-error isolation
  - Default stub raises requires-gpu-runtime
  - Real on-disk held-out integration with mocked inference

The scorer is fully dependency-injected; no network or torch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from lamarck.eval.rubrics import json_mode
from lamarck.eval.tier3_grounded import (
    HoldoutProblem,
    default_holdout_path,
    load_holdout,
)


# ---- JSON extraction -------------------------------------------------------

def test_extract_whole_string_json():
    src = '{"a": 1}'
    assert json_mode._extract_first_json_object(src) == src


def test_extract_fenced_json():
    src = '```json\n{"a": 1, "b": [1, 2]}\n```'
    extracted = json_mode._extract_first_json_object(src)
    assert extracted == '{"a": 1, "b": [1, 2]}'


def test_extract_brace_balanced_skips_prose_preamble():
    src = 'Sure! Here is the response:\n{"name": "test", "ok": true}\nDone.'
    extracted = json_mode._extract_first_json_object(src)
    assert extracted == '{"name": "test", "ok": true}'


def test_extract_balanced_handles_nested_braces():
    src = 'preamble {"outer": {"inner": {"deep": 1}}} suffix'
    extracted = json_mode._extract_first_json_object(src)
    assert extracted == '{"outer": {"inner": {"deep": 1}}}'


def test_extract_balanced_respects_strings_with_braces():
    """A closing-brace inside a string shouldn't end the object."""
    src = '{"sentence": "this has } braces { in it", "ok": true}'
    extracted = json_mode._extract_first_json_object(src)
    assert extracted == src


def test_extract_returns_none_on_no_json():
    assert json_mode._extract_first_json_object("just prose, no braces") is None
    assert json_mode._extract_first_json_object("") is None


def test_extract_returns_none_on_unbalanced_json():
    src = '{"a": 1, "b": '  # truncated
    assert json_mode._extract_first_json_object(src) is None


# ---- Per-problem scoring ---------------------------------------------------

def test_score_one_passes_valid_match():
    schema = {"type": "object", "required": ["name"],
              "properties": {"name": {"type": "string"}}}
    passed, reason = json_mode.score_one_output('{"name": "Aaron"}', schema)
    assert passed is True
    assert reason == json_mode.PASS_REASON


def test_score_one_no_json_found():
    passed, reason = json_mode.score_one_output("no JSON here", {})
    assert passed is False
    assert reason == json_mode.REASON_NO_JSON


def test_score_one_parse_error_inside_extracted_braces():
    """The extractor returns a string with a brace pair but the contents
    aren't valid JSON (trailing comma)."""
    src = '{"a": 1,}'  # extractor returns the whole thing; json.loads fails
    passed, reason = json_mode.score_one_output(
        src, {"type": "object", "required": []},
    )
    assert passed is False
    assert json_mode.REASON_PARSE in reason


def test_score_one_schema_validation_failure():
    """jsonschema rejects: required key missing."""
    schema = {"type": "object", "required": ["name", "email"],
              "properties": {"name": {"type": "string"},
                             "email": {"type": "string"}}}
    passed, reason = json_mode.score_one_output('{"name": "Aaron"}', schema)
    assert passed is False
    assert json_mode.REASON_SCHEMA in reason


# ---- Fallback path (jsonschema absent) ------------------------------------

def test_fallback_validator_accepts_object_with_required_keys():
    """When jsonschema isn't importable, the structural fallback runs."""
    schema = {"type": "object", "required": ["a", "b"]}
    with patch.dict("sys.modules", {"jsonschema": None}):
        # Use a clean import to ensure the import inside the function
        # actually hits the patched stub.
        passed, why = json_mode._validate_against_schema(
            {"a": 1, "b": 2}, schema,
        )
    assert passed is True


def test_fallback_validator_rejects_missing_required():
    schema = {"type": "object", "required": ["a", "b"]}
    with patch.dict("sys.modules", {"jsonschema": None}):
        passed, why = json_mode._validate_against_schema(
            {"a": 1}, schema,  # missing 'b'
        )
    assert passed is False
    assert "fallback: missing required keys ['b']" in why


def test_fallback_validator_rejects_non_object_when_type_is_object():
    schema = {"type": "object", "required": []}
    with patch.dict("sys.modules", {"jsonschema": None}):
        passed, why = json_mode._validate_against_schema([1, 2, 3], schema)
    assert passed is False
    assert "fallback: expected object" in why


# ---- score_json_mode end-to-end -------------------------------------------

def _holdout(n: int) -> list[HoldoutProblem]:
    return [
        HoldoutProblem(
            input=f"prompt {i}",
            schema={
                "type": "object",
                "required": ["k"],
                "properties": {"k": {"type": "string"}},
            },
        )
        for i in range(n)
    ]


def test_score_json_mode_all_pass():
    result = json_mode.score_json_mode(
        Path("/tmp/adapter"),
        _holdout(5),
        inference_fn=lambda adapter, prompt: '{"k": "v"}',
    )
    assert result["pass_count"] == 5
    assert result["total"] == 5
    assert result["fraction"] == 1.0
    assert result["score"] == 100.0
    assert result["errors"] == []
    assert all(p["passed"] for p in result["per_problem"])


def test_score_json_mode_all_fail_no_json():
    result = json_mode.score_json_mode(
        Path("/tmp/adapter"),
        _holdout(5),
        inference_fn=lambda *a: "I cannot answer",
    )
    assert result["pass_count"] == 0
    assert result["score"] == 0.0
    for p in result["per_problem"]:
        assert p["passed"] is False
        assert p["reason"] == json_mode.REASON_NO_JSON


def test_score_json_mode_partial_pass():
    """Pass on even indices, fail on odd. 5 of 10 -> 50.0."""
    call_idx = [0]

    def alternating(adapter, prompt):
        i = call_idx[0]
        call_idx[0] += 1
        return '{"k": "v"}' if i % 2 == 0 else "no json"

    result = json_mode.score_json_mode(
        Path("/tmp/adapter"), _holdout(10),
        inference_fn=alternating,
    )
    assert result["pass_count"] == 5
    assert result["score"] == 50.0


def test_score_json_mode_isolates_inference_errors():
    """Inference raising on one problem doesn't poison the rest."""

    def buggy(adapter, prompt):
        if prompt == "prompt 2":
            raise RuntimeError("boom")
        return '{"k": "v"}'

    result = json_mode.score_json_mode(
        Path("/tmp/adapter"), _holdout(5),
        inference_fn=buggy,
    )
    # 4 of 5 pass (idx 0, 1, 3, 4); idx 2 raises.
    assert result["pass_count"] == 4
    assert result["score"] == 80.0
    assert len(result["errors"]) == 1
    assert "RuntimeError" in result["errors"][0]
    # Per-problem still has 5 entries; idx 2 marked failed with inference_error.
    assert len(result["per_problem"]) == 5
    failed_problem = next(p for p in result["per_problem"] if p["idx"] == 2)
    assert failed_problem["passed"] is False
    assert json_mode.REASON_INFER in failed_problem["reason"]


def test_score_json_mode_truncates_raw_excerpt():
    long_output = "X" * 5000 + '{"k": "v"}'
    result = json_mode.score_json_mode(
        Path("/tmp/adapter"), _holdout(1),
        inference_fn=lambda *a: long_output,
        output_truncate_chars=100,
    )
    assert len(result["per_problem"][0]["raw_excerpt"]) == 100


def test_default_inference_fn_raises_requires_gpu():
    """When no inference_fn is injected, the default stub raises -
    which lands as a per-problem inference_error (not an aggregate crash)."""
    result = json_mode.score_json_mode(
        Path("/tmp/adapter"), _holdout(3),
        inference_fn=None,
    )
    assert result["pass_count"] == 0
    # Every problem errored via the default stub.
    assert len(result["errors"]) == 3
    for line in result["errors"]:
        assert "RuntimeError" in line and "requires-gpu-runtime" in line


def test_empty_holdout_returns_zero_safely():
    result = json_mode.score_json_mode(
        Path("/tmp/adapter"), [],
        inference_fn=lambda *a: '{"k": "v"}',
    )
    assert result["pass_count"] == 0
    assert result["total"] == 0
    assert result["fraction"] == 0.0
    assert result["score"] == 0.0


# ---- Real on-disk held-out integration ------------------------------------

def test_real_holdout_perfect_inference_scores_one_hundred():
    """Wire the actual 100-problem held-out set through the scorer with
    a mocked inference that always returns a valid response for the
    given schema. Locks the end-to-end pass-rate math."""
    holdout = load_holdout(default_holdout_path())
    assert len(holdout) == 100  # sanity

    def schema_satisfying(adapter, prompt) -> str:
        # Build a minimal valid object for the LAST schema we saw.
        # The scorer fires problems in order, so we track which one
        # we're on by parsing the schema from the prompt - simpler:
        # just look at the schema attribute via a closure on holdout.
        idx = schema_satisfying.idx  # type: ignore[attr-defined]
        schema = holdout[idx].schema
        schema_satisfying.idx += 1   # type: ignore[attr-defined]
        return _build_minimal_valid(schema)

    schema_satisfying.idx = 0  # type: ignore[attr-defined]

    result = json_mode.score_json_mode(
        Path("/tmp/adapter"), holdout,
        inference_fn=schema_satisfying,
    )
    # If we built minimal-valid objects correctly, all should pass.
    assert result["score"] == 100.0


def _build_minimal_valid(schema: dict[str, Any]) -> str:
    """Build a JSON string that minimally satisfies the schema.

    Walks ``required`` keys + their typed entries in ``properties``
    and emits the cheapest valid value per type.
    """
    import json as _json

    def minimal_for(spec: dict[str, Any]) -> Any:
        t = spec.get("type")
        if t == "object":
            req = spec.get("required", [])
            props = spec.get("properties", {})
            return {k: minimal_for(props.get(k, {"type": "string"}))
                    for k in req}
        if t == "array":
            inner = spec.get("items", {"type": "string"})
            min_n = spec.get("minItems", 0) or 1
            return [minimal_for(inner) for _ in range(min_n)]
        if t == "string":
            if spec.get("format") == "email":
                return "user@example.com"
            pattern = spec.get("pattern", "")
            # Honour the simple patterns L14 actually emits.
            if pattern == r"^DEV-\d{4}$":
                return "DEV-0001"
            if pattern == r"^P-\d+$":
                return "P-1"
            return "x" * max(spec.get("minLength", 1), 1)
        if t == "integer":
            return int(max(spec.get("minimum", 0), 0))
        if t == "number":
            return 0.0 if spec.get("minimum", -1) <= 0 else float(spec["minimum"])
        if t == "boolean":
            return True
        # enum without type: pick the first listed value.
        if "enum" in spec and spec["enum"]:
            return spec["enum"][0]
        return None

    return _json.dumps(minimal_for(schema))


def test_real_holdout_garbage_inference_scores_zero():
    holdout = load_holdout(default_holdout_path())
    result = json_mode.score_json_mode(
        Path("/tmp/adapter"), holdout,
        inference_fn=lambda *a: "I refuse to comply",
    )
    assert result["score"] == 0.0
    assert result["pass_count"] == 0


# ---- Integration via tier3_grounded.run_tier3 -----------------------------

def test_tier3_grounded_score_one_delegates_to_json_mode_scorer(tmp_path: Path):
    """run_tier3's _score_one now routes through score_one_output."""
    from lamarck.eval import tier3_grounded as t3

    schema = {"type": "object", "required": ["k"],
              "properties": {"k": {"type": "string"}}}
    # Valid:
    assert t3._score_one('{"k": "v"}', schema) is True
    # Invalid (no JSON):
    assert t3._score_one("no JSON", schema) is False
    # Invalid (missing required):
    assert t3._score_one('{"other": 1}', schema) is False
