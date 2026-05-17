# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
"""Locked-invariant + mock-injected tests for the Tier 3 harness.

Full GPU-driven integration belongs on the pod; the L16 mock-based
tests will exercise more pathways once L14/L15 are wired. This file
locks the v1 numeric contract and end-to-end scoring math.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lamarck.eval import tier3_grounded as t3


# ---- Locked v1 contract ----------------------------------------------------

def test_student_model_id_is_locked():
    """v1 ships Llama-3.2-1B-Instruct. Any change is a v2 event."""
    assert t3.STUDENT_MODEL_ID == "meta-llama/Llama-3.2-1B-Instruct"


def test_holdout_problem_count_is_locked_to_one_hundred():
    assert t3.HOLDOUT_PROBLEM_COUNT == 100


def test_recipe_is_locked_to_v1_values():
    r = t3.TIER3_RECIPE
    assert r.epochs           == 1
    assert r.per_device_batch == 4
    assert r.learning_rate    == 2e-4
    assert r.lora_rank        == 16
    assert r.lora_alpha       == 32
    assert r.lora_dropout     == 0.05
    assert r.seed             == 42


def test_recipe_is_frozen_dataclass():
    """Locked recipes shouldn't be mutated mid-run."""
    with pytest.raises((AttributeError, Exception)):
        t3.TIER3_RECIPE.learning_rate = 1e-3  # type: ignore[misc]


def test_recipe_as_dict_round_trip():
    d = t3.TIER3_RECIPE.as_dict()
    assert set(d) == {"epochs", "per_device_batch", "learning_rate",
                      "lora_rank", "lora_alpha", "lora_dropout", "seed"}


# ---- Default stubs require GPU --------------------------------------------

def test_default_fine_tune_fn_raises_requires_gpu():
    with pytest.raises(RuntimeError, match="requires-gpu-runtime"):
        t3._requires_gpu_fine_tune(
            t3.StudentSpec(), t3.TIER3_RECIPE,
            Path("/tmp/no.jsonl"), Path("/tmp/out"),
        )


def test_default_inference_fn_raises_requires_gpu():
    with pytest.raises(RuntimeError, match="requires-gpu-runtime"):
        t3._requires_gpu_inference(Path("/tmp/adapter"), "prompt")


# ---- Held-out loader -------------------------------------------------------

def test_load_holdout_missing_file_returns_empty(tmp_path: Path):
    assert t3.load_holdout(tmp_path / "does-not-exist.jsonl") == []


def test_load_holdout_parses_valid_jsonl(tmp_path: Path):
    p = tmp_path / "h.jsonl"
    p.write_text(
        '{"input": "Q1", "schema": {"type": "object"}}\n'
        '{"input": "Q2", "schema": {"type": "array"}}\n'
    )
    problems = t3.load_holdout(p)
    assert len(problems) == 2
    assert problems[0].input == "Q1"
    assert problems[0].schema == {"type": "object"}


def test_load_holdout_rejects_missing_keys(tmp_path: Path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"input": "Q1"}\n')  # no schema
    with pytest.raises(ValueError, match="missing keys"):
        t3.load_holdout(p)


def test_load_holdout_skips_blank_lines(tmp_path: Path):
    p = tmp_path / "h.jsonl"
    p.write_text(
        '{"input": "Q1", "schema": {"type": "object"}}\n'
        '\n'
        '   \n'
        '{"input": "Q2", "schema": {"type": "object"}}\n'
    )
    assert len(t3.load_holdout(p)) == 2


# ---- Per-problem scorer ---------------------------------------------------

def test_score_one_passes_valid_json_matching_schema():
    assert t3._score_one('{"name": "x"}', {"type": "object"}) is True


def test_score_one_fails_invalid_json():
    assert t3._score_one("not json", {"type": "object"}) is False


def test_score_one_fails_wrong_top_level_type():
    """Fallback path: array schema expected, object given."""
    assert t3._score_one('{"name": "x"}', {"type": "array"}) is False


# ---- run_tier3 end-to-end --------------------------------------------------

def _write_curriculum(tmp_path: Path) -> Path:
    cur = tmp_path / "curriculum.jsonl"
    cur.write_text('{"prompt": "p", "completion": "c"}\n')
    return cur


def _write_holdout(tmp_path: Path, n_problems: int = 4) -> Path:
    h = tmp_path / "holdout.jsonl"
    lines = [
        json.dumps({"input": f"Q{i}", "schema": {"type": "object"}})
        for i in range(n_problems)
    ]
    h.write_text("\n".join(lines) + "\n")
    return h


def test_no_curriculum_supplied_returns_partial(tmp_path: Path):
    result = t3.run_tier3(
        model_id="m", curriculum_jsonl=None,
    )
    assert result["tier"] == 3
    assert result["score"] == 0.0
    assert "curriculum" in result["components"]["errors"]


def test_curriculum_path_missing_returns_partial(tmp_path: Path):
    result = t3.run_tier3(
        model_id="m",
        curriculum_jsonl=tmp_path / "does-not-exist.jsonl",
    )
    assert result["score"] == 0.0
    assert "curriculum" in result["components"]["errors"]


def test_holdout_missing_returns_partial(tmp_path: Path):
    cur = _write_curriculum(tmp_path)
    result = t3.run_tier3(
        model_id="m", curriculum_jsonl=cur,
        holdout_path=tmp_path / "no-holdout.jsonl",
    )
    assert result["score"] == 0.0
    assert "holdout" in result["components"]["errors"]


def test_full_pass_pipeline_scores_one_hundred(tmp_path: Path):
    """Mocked fine-tune + always-passing inference -> 100.0."""
    cur = _write_curriculum(tmp_path)
    hold = _write_holdout(tmp_path, n_problems=4)

    def fake_fine_tune(student, recipe, curriculum, output_dir):
        assert student.model_id == t3.STUDENT_MODEL_ID
        assert recipe is t3.TIER3_RECIPE
        return tmp_path / "fake-adapter"

    def always_pass(adapter, prompt):
        return '{"ok": true}'

    result = t3.run_tier3(
        model_id="m", curriculum_jsonl=cur, holdout_path=hold,
        fine_tune_fn=fake_fine_tune, inference_fn=always_pass,
    )
    assert result["score"] == 100.0
    assert result["components"]["pass_count"] == 4
    assert result["components"]["holdout_size"] == 4


def test_full_fail_pipeline_scores_zero(tmp_path: Path):
    cur = _write_curriculum(tmp_path)
    hold = _write_holdout(tmp_path, n_problems=4)

    def fake_fine_tune(*a, **kw):
        return tmp_path / "adapter"

    def always_fail(adapter, prompt):
        return "not json"

    result = t3.run_tier3(
        model_id="m", curriculum_jsonl=cur, holdout_path=hold,
        fine_tune_fn=fake_fine_tune, inference_fn=always_fail,
    )
    assert result["score"] == 0.0
    assert result["components"]["pass_count"] == 0


def test_partial_pass_pipeline_scores_correctly(tmp_path: Path):
    """2 of 4 problems pass -> 50.0."""
    cur = _write_curriculum(tmp_path)
    hold = _write_holdout(tmp_path, n_problems=4)

    def fake_fine_tune(*a, **kw):
        return tmp_path / "adapter"

    call_count = [0]
    def alternating(adapter, prompt):
        call_count[0] += 1
        return '{"ok": true}' if call_count[0] % 2 == 0 else "bad"

    result = t3.run_tier3(
        model_id="m", curriculum_jsonl=cur, holdout_path=hold,
        fine_tune_fn=fake_fine_tune, inference_fn=alternating,
    )
    assert result["score"] == 50.0
    assert result["components"]["pass_count"] == 2


def test_fine_tune_exception_returns_partial_with_error(tmp_path: Path):
    cur = _write_curriculum(tmp_path)
    hold = _write_holdout(tmp_path, n_problems=4)

    def bad_fine_tune(*a, **kw):
        raise RuntimeError("CUDA OOM")

    result = t3.run_tier3(
        model_id="m", curriculum_jsonl=cur, holdout_path=hold,
        fine_tune_fn=bad_fine_tune,
        inference_fn=lambda *a: "should not be called",
    )
    assert result["score"] == 0.0
    assert "fine_tune" in result["components"]["errors"]
    assert "RuntimeError" in result["components"]["errors"]["fine_tune"]
    assert "CUDA OOM" in result["components"]["errors"]["fine_tune"]


def test_inference_exception_isolated_to_failing_problem(tmp_path: Path):
    """If inference raises on one problem, others still run."""
    cur = _write_curriculum(tmp_path)
    hold = _write_holdout(tmp_path, n_problems=4)

    def fake_fine_tune(*a, **kw):
        return tmp_path / "adapter"

    def buggy_inference(adapter, prompt):
        if prompt == "Q2":
            raise RuntimeError("inference failure")
        return '{"ok": true}'

    result = t3.run_tier3(
        model_id="m", curriculum_jsonl=cur, holdout_path=hold,
        fine_tune_fn=fake_fine_tune, inference_fn=buggy_inference,
    )
    # 3 problems pass (Q0, Q1, Q3); Q2 errored -> 3/4 = 75.0.
    assert result["score"] == 75.0
    assert result["components"]["pass_count"] == 3
    errs = result["components"]["errors"]["inference_errors"]
    assert any("RuntimeError" in e and "inference failure" in e for e in errs)


def test_run_tier3_exposes_locked_constants_in_components(tmp_path: Path):
    """The TierResult must include the student model ID + recipe so a
    consumer can audit which v1 spec the run was against."""
    cur = _write_curriculum(tmp_path)
    hold = _write_holdout(tmp_path, n_problems=2)
    result = t3.run_tier3(
        model_id="m", curriculum_jsonl=cur, holdout_path=hold,
        fine_tune_fn=lambda *a, **kw: tmp_path / "a",
        inference_fn=lambda a, p: '{"ok": true}',
    )
    comps = result["components"]
    assert comps["student_model_id"] == t3.STUDENT_MODEL_ID
    assert comps["recipe"] == t3.TIER3_RECIPE.as_dict()
