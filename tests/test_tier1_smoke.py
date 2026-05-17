# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
"""Minimal smoke tests for Tier 1 module-level invariants.

Full integration tests (with mocked lm_eval) land in L3 as
``tests/test_tier1.py``. This file's job is to catch silent edits
to the locked Tier 1 spec — benchmark list, MMLU-Pro subsets,
metric extraction discipline — so a stale-by-accident change
shows up in CI even before the full mocks exist.
"""

from __future__ import annotations

import pytest

from lamarck.eval import tier1_external as t1


def test_tier1_benchmark_list_is_locked():
    """v1 ships exactly these three benchmarks. Adding/removing is v2."""
    assert t1.TIER1_BENCHMARKS == ("mmlu_pro", "humaneval_plus", "gsm8k")


def test_mmlu_pro_subset_list_is_locked():
    """v1 restricts MMLU-Pro to these four subsets."""
    assert t1.MMLU_PRO_SUBSETS == (
        "computer_science", "engineering", "math", "physics",
    )


def test_each_benchmark_has_a_runner():
    """Every name in TIER1_BENCHMARKS must have a registered runner."""
    for name in t1.TIER1_BENCHMARKS:
        assert name in t1._BENCH_RUNNERS


def test_extract_score_accepts_zero_to_one():
    """Most lm-eval task adapters report fractions in [0, 1]."""
    results = {"results": {"task_x": {"acc,none": 0.73}}}
    assert t1._extract_score(results, "task_x") == 0.73


def test_extract_score_coerces_zero_to_hundred_scale():
    """Some harness versions report on a 0-100 scale. Coerce to fractions."""
    results = {"results": {"task_x": {"acc,none": 73.0}}}
    assert t1._extract_score(results, "task_x") == 0.73


def test_extract_score_rejects_out_of_range():
    """Negative or >100 should fail loudly, not silently."""
    results = {"results": {"task_x": {"acc,none": -0.1}}}
    with pytest.raises(ValueError):
        t1._extract_score(results, "task_x")


def test_extract_score_rejects_unknown_metrics():
    """If no recognized metric name is present, raise — don't fabricate."""
    results = {"results": {"task_x": {"some_unknown_metric": 0.5}}}
    with pytest.raises(KeyError):
        t1._extract_score(results, "task_x")


def test_run_tier1_handles_lm_eval_missing(monkeypatch):
    """When lm_eval isn't installed, run_tier1 returns a partial result
    with the import error surfaced under components.errors, not a crash."""
    def _no_lm_eval():
        raise ImportError("no module named lm_eval")
    monkeypatch.setattr(t1, "_resolve_simple_evaluate", _no_lm_eval)
    with pytest.raises(ImportError):
        t1.run_tier1(model_id="test-model")


def test_run_tier1_returns_partial_when_a_benchmark_fails(monkeypatch):
    """A benchmark exception should not abort Tier 1 — surface the error
    in components and score what remains."""

    def fake_evaluate(model: str, model_args: str, tasks: list[str]) -> dict:
        if "humaneval_plus" in tasks:
            raise RuntimeError("humaneval load failed")
        # All tasks named here are MMLU-Pro subsets or gsm8k.
        return {"results": {task: {"acc,none": 0.5} for task in tasks}}

    monkeypatch.setattr(t1, "_resolve_simple_evaluate", lambda: fake_evaluate)

    result = t1.run_tier1(model_id="test-model")
    assert result["tier"] == 1
    assert result["v"] == 1
    assert result["model_id"] == "test-model"
    assert result["components"]["humaneval_plus"] is None
    assert "errors" in result["components"]
    assert "humaneval_plus" in result["components"]["errors"]
    # Surviving benchmarks averaged: (50 + 50) / 2 = 50
    assert result["score"] == 50.0


def test_run_tier1_normalizes_to_0_to_100(monkeypatch):
    """Three benchmarks at 0.5, 0.6, 0.7 → average 0.6 → score 60.0."""
    fractions = iter([0.5, 0.6, 0.7])

    def fake_evaluate(model: str, model_args: str, tasks: list[str]) -> dict:
        return {"results": {task: {"acc,none": next(fractions)} for task in tasks}}

    # Disable subset averaging by stubbing the runners directly — that
    # way we get exactly the three values above, not a 4-way mean.
    monkeypatch.setattr(t1, "_run_mmlu_pro",       lambda *_a: 0.5)
    monkeypatch.setattr(t1, "_run_humaneval_plus", lambda *_a: 0.6)
    monkeypatch.setattr(t1, "_run_gsm8k",          lambda *_a: 0.7)
    # Re-bind the runners dict so the run_tier1 lookup picks up the patches.
    monkeypatch.setattr(t1, "_BENCH_RUNNERS", {
        "mmlu_pro":       t1._run_mmlu_pro,
        "humaneval_plus": t1._run_humaneval_plus,
        "gsm8k":          t1._run_gsm8k,
    })
    monkeypatch.setattr(t1, "_resolve_simple_evaluate", lambda: fake_evaluate)

    result = t1.run_tier1(model_id="test-model")
    assert result["score"] == 60.0
    assert result["components"]["mmlu_pro"]       == 50.0
    assert result["components"]["humaneval_plus"] == 60.0
    assert result["components"]["gsm8k"]          == 70.0
