# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
"""End-to-end integration tests for Tier 3.

Wires the full Tier 3 pipeline against the on-disk 100-problem
held-out corpus shipped in L14. Mocks fine_tune_fn + inference_fn
(both GPU-only) so the tests run cleanly in a CPU-only CI venv.

Companion files:
  - test_tier3_smoke.py:        harness invariants, mock-injected
                                 small-holdout math.
  - test_tier3_holdout_corpus.py: corpus-validation (count, schemas).
  - test_json_mode_scorer.py:    scorer details + tier3 _score_one
                                 delegation check.
  - test_tier3.py (this file):  full end-to-end against the real
                                 on-disk 100-problem holdout.

The shared minimal-valid-document helper lives in
``tests/_minimal_valid.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest

from lamarck.eval import tier3_grounded as t3
from lamarck.eval.tier3_grounded import (
    HOLDOUT_PROBLEM_COUNT,
    STUDENT_MODEL_ID,
    TIER3_RECIPE,
    default_holdout_path,
    load_holdout,
    run_tier3,
)

from _minimal_valid import minimal_valid_json


# ---- Fixtures --------------------------------------------------------------

@pytest.fixture(scope="module")
def real_holdout():
    return load_holdout(default_holdout_path())


@pytest.fixture
def real_curriculum(tmp_path: Path) -> Path:
    """A non-empty curriculum file. Contents don't matter for these tests;
    the mocked fine_tune_fn ignores them."""
    p = tmp_path / "curriculum.jsonl"
    p.write_text(
        '{"prompt": "respond in JSON", "completion": "{}"}\n'
        '{"prompt": "another example", "completion": "{}"}\n'
    )
    return p


def _ok_fine_tune(adapter: Path) -> Callable:
    """A fake fine-tune that just returns a stub adapter path."""

    def fn(student, recipe, curriculum, output_dir):
        return adapter

    return fn


def _perfect_inference(holdout) -> Callable[[Path, str], str]:
    """Build an inference_fn that returns a minimal-valid response per
    held-out problem. Track which problem we're on by prompt-to-index
    lookup (each held-out 'input' is unique - locked by the L14 test)."""
    by_input = {p.input: p.schema for p in holdout}

    def fn(adapter: Path, prompt: str) -> str:
        schema = by_input[prompt]
        return minimal_valid_json(schema)

    return fn


# ---- (a) Perfect inference: real corpus, score 100.0 ---------------------

def test_tier3_perfect_inference_against_real_holdout(real_holdout,
                                                      real_curriculum,
                                                      tmp_path: Path):
    result = run_tier3(
        model_id="lamarck-g1",
        curriculum_jsonl=real_curriculum,
        fine_tune_fn=_ok_fine_tune(tmp_path / "adapter"),
        inference_fn=_perfect_inference(real_holdout),
    )
    assert result["tier"] == 3
    assert result["v"]    == 1
    assert result["model_id"] == "lamarck-g1"
    assert result["score"] == 100.0
    assert result["components"]["pass_count"]   == HOLDOUT_PROBLEM_COUNT
    assert result["components"]["holdout_size"] == HOLDOUT_PROBLEM_COUNT


# ---- (b) All-fail inference: score 0.0 -----------------------------------

def test_tier3_garbage_inference_against_real_holdout(real_holdout,
                                                      real_curriculum,
                                                      tmp_path: Path):
    result = run_tier3(
        model_id="lamarck-g1",
        curriculum_jsonl=real_curriculum,
        fine_tune_fn=_ok_fine_tune(tmp_path / "adapter"),
        inference_fn=lambda adapter, prompt: "I cannot comply.",
    )
    assert result["score"] == 0.0
    assert result["components"]["pass_count"] == 0
    assert result["components"]["holdout_size"] == HOLDOUT_PROBLEM_COUNT


# ---- (c) Partial pass: every 4th problem passes -> score 25.0 ------------

def test_tier3_every_fourth_passes_real_holdout(real_holdout,
                                                real_curriculum,
                                                tmp_path: Path):
    by_input = {p.input: p.schema for p in real_holdout}
    inputs_in_order = [p.input for p in real_holdout]
    pass_indices = set(range(0, HOLDOUT_PROBLEM_COUNT, 4))  # 25 passes

    def alternating(adapter, prompt):
        idx = inputs_in_order.index(prompt)
        if idx in pass_indices:
            return minimal_valid_json(by_input[prompt])
        return "no JSON here"

    result = run_tier3(
        model_id="lamarck-g1",
        curriculum_jsonl=real_curriculum,
        fine_tune_fn=_ok_fine_tune(tmp_path / "adapter"),
        inference_fn=alternating,
    )
    assert result["score"] == 25.0
    assert result["components"]["pass_count"] == 25


# ---- (d) Inference raises on subset: isolated per-problem failures -------

def test_tier3_inference_exceptions_are_per_problem_isolated(real_holdout,
                                                              real_curriculum,
                                                              tmp_path: Path):
    """Inference raises on problems 10..14 (5 problems). The other 95
    score normally. The 5 failing problems get inference_errors entries
    in components.errors but the run completes."""
    by_input = {p.input: p.schema for p in real_holdout}
    inputs_in_order = [p.input for p in real_holdout]
    raising_indices = set(range(10, 15))

    def buggy(adapter, prompt):
        idx = inputs_in_order.index(prompt)
        if idx in raising_indices:
            raise RuntimeError(f"transient at idx {idx}")
        return minimal_valid_json(by_input[prompt])

    result = run_tier3(
        model_id="lamarck-g1",
        curriculum_jsonl=real_curriculum,
        fine_tune_fn=_ok_fine_tune(tmp_path / "adapter"),
        inference_fn=buggy,
    )
    # 95 of 100 pass; the 5 raised.
    assert result["components"]["pass_count"] == 95
    assert result["score"] == 95.0
    errs = result["components"]["errors"]["inference_errors"]
    assert len(errs) == 5
    for line in errs:
        assert "RuntimeError" in line and "transient at idx" in line


# ---- (e) Locked-spec exposure in components ------------------------------

def test_tier3_components_carry_locked_student_and_recipe(real_holdout,
                                                          real_curriculum,
                                                          tmp_path: Path):
    """Auditing a Tier 3 result must reveal which v1 student model and
    recipe the run was against. This is what catches a stealth
    student-model swap mid-experiment."""
    result = run_tier3(
        model_id="lamarck-g1",
        curriculum_jsonl=real_curriculum,
        fine_tune_fn=_ok_fine_tune(tmp_path / "adapter"),
        inference_fn=_perfect_inference(real_holdout),
    )
    comps = result["components"]
    assert comps["student_model_id"] == STUDENT_MODEL_ID
    assert comps["recipe"] == TIER3_RECIPE.as_dict()
    assert comps["holdout_size"] == HOLDOUT_PROBLEM_COUNT
    # adapter_path was returned by the fine-tune mock.
    assert "adapter" in comps["adapter_path"]


# ---- (f) Curriculum-missing path is partial -------------------------------

def test_tier3_missing_curriculum_returns_partial(tmp_path: Path):
    """No curriculum supplied -> partial result with the curriculum
    error named in components.errors. Fine-tune is NOT attempted."""
    fine_tune_called = []

    def must_not_run(*a, **kw):
        fine_tune_called.append(1)
        return tmp_path / "adapter"

    result = run_tier3(
        model_id="lamarck-g1", curriculum_jsonl=None,
        fine_tune_fn=must_not_run,
        inference_fn=lambda *a: '{"k": "v"}',
    )
    assert result["score"] == 0.0
    assert "curriculum" in result["components"]["errors"]
    assert fine_tune_called == []  # fine-tune was not invoked


def test_tier3_curriculum_path_present_but_file_missing_returns_partial(
    tmp_path: Path,
):
    fake_curriculum = tmp_path / "does-not-exist.jsonl"
    result = run_tier3(
        model_id="lamarck-g1", curriculum_jsonl=fake_curriculum,
        fine_tune_fn=lambda *a, **kw: tmp_path / "adapter",
        inference_fn=lambda *a: '{"k": "v"}',
    )
    assert result["score"] == 0.0
    assert "curriculum" in result["components"]["errors"]


# ---- Tier-result schema round trip ----------------------------------------

def test_tier3_result_matches_TierResult_keys(real_holdout, real_curriculum,
                                              tmp_path: Path):
    result = run_tier3(
        model_id="m", curriculum_jsonl=real_curriculum,
        fine_tune_fn=_ok_fine_tune(tmp_path / "adapter"),
        inference_fn=_perfect_inference(real_holdout),
    )
    for key in ("tier", "score", "components", "ran_at", "model_id", "v"):
        assert key in result, f"TierResult missing key: {key!r}"
    assert isinstance(result["score"], float)
    assert isinstance(result["components"], dict)
    assert result["v"] == 1  # SUITE_VERSION


# ---- Sanity: minimal_valid_json builds something jsonschema accepts ------

def test_minimal_valid_helper_builds_passing_responses(real_holdout):
    """Verify the test helper itself produces responses that pass the
    L15 scorer for every held-out schema. If this fails, the helper is
    buggy - the integration tests above depend on it."""
    from lamarck.eval.rubrics.json_mode import score_one_output

    for idx, problem in enumerate(real_holdout):
        response = minimal_valid_json(problem.schema)
        passed, reason = score_one_output(response, problem.schema)
        assert passed, (
            f"problem {idx} minimal-valid response did not pass scorer: "
            f"reason={reason!r} response={response!r}"
        )
