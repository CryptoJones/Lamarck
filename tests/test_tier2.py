# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
"""End-to-end integration tests for Tier 2.

Wires all four rubric runners (peft_loop, custom_layer, llm_judge x2)
into the ``tier2_engineering`` dispatcher and verifies the full
scoring pipeline against the real on-disk task corpus.

Test matrix:
  (a) all 50 tasks pass with mocked-positive runners ->
      score == 100.0, raw_points == 200, no errors.
  (b) all categories registered but each runner returns 0/max ->
      score == 0, by-category breakdown all-zero.
  (c) mixed: peft + custom-layers at max (90 + 10 = 100), curriculum
      + diagnostics at half (20 + 30 = 50) -> raw 150/200 = 75.0.
  (d) one runner unregistered -> no-runner stub fires, errors entry
      surfaces, other categories still score.
  (e) one runner raises -> per-task error in errors.task_errors,
      sibling categories unaffected.

The query_model is also mocked so no network is touched.
"""

from __future__ import annotations

from typing import Callable
from unittest.mock import patch

import pytest

from lamarck.eval.rubrics import custom_layer, llm_judge, peft_loop
from lamarck.eval.tier2_engineering import (
    TIER2_CATEGORIES,
    TIER2_MAX_POINTS_PER_TASK,
    TIER2_RAW_POINTS_PER_CATEGORY,
    TIER2_TASK_COUNTS,
    TIER2_TOTAL_RAW_POINTS,
    RubricResult,
    default_tasks_dir,
    run_tier2,
)


# ---- Test runners (closure-based, return constant results) ----------------

def _make_constant_runner(score: int, max_score: int) -> Callable:
    def runner(task, model_output):
        return RubricResult(score=score, max_score=max_score,
                            rationale=f"constant {score}/{max_score}")
    return runner


def _stub_query_model(prompt: str, model_id: str, base_url: str) -> str:
    """Stand-in for the model under test. Returns a placeholder string;
    actual scoring is decided by the injected runners, not the output."""
    return "STUB_MODEL_RESPONSE"


# ---- (a) Full-positive integration -----------------------------------------

def test_full_corpus_with_positive_runners_scores_one_hundred():
    """All 50 tasks pass every rubric -> 200/200 raw -> 100.0 normalized."""
    runners = {
        "peft-loops":        _make_constant_runner(6, 6),
        "curriculum-design": _make_constant_runner(4, 4),
        "custom-layers":     _make_constant_runner(1, 1),
        "diagnostics":       _make_constant_runner(4, 4),
    }
    result = run_tier2(
        model_id="test-model",
        tasks_dir=default_tasks_dir(),
        rubric_runners=runners,
        query_model=_stub_query_model,
    )
    assert result["tier"] == 2
    assert result["v"] == 1
    assert result["model_id"] == "test-model"
    assert result["score"] == 100.0
    assert result["components"]["raw_points"] == TIER2_TOTAL_RAW_POINTS
    assert result["components"]["max_raw_points"] == TIER2_TOTAL_RAW_POINTS
    assert "errors" not in result["components"], (
        f"unexpected errors: {result['components'].get('errors')}"
    )
    for cat in TIER2_CATEGORIES:
        cat_block = result["components"]["by_category"][cat]
        assert cat_block["score"]     == TIER2_RAW_POINTS_PER_CATEGORY[cat]
        assert cat_block["max_score"] == TIER2_RAW_POINTS_PER_CATEGORY[cat]
        assert cat_block["tasks_run"] == TIER2_TASK_COUNTS[cat]


# ---- (b) Full-zero integration --------------------------------------------

def test_full_corpus_with_zero_runners_scores_zero():
    """Every task scores 0; final score == 0.0, by-category all zero."""
    runners = {cat: _make_constant_runner(0, TIER2_MAX_POINTS_PER_TASK[cat])
               for cat in TIER2_CATEGORIES}
    result = run_tier2(
        model_id="test",
        tasks_dir=default_tasks_dir(),
        rubric_runners=runners,
        query_model=_stub_query_model,
    )
    assert result["score"] == 0.0
    assert result["components"]["raw_points"] == 0
    assert "errors" not in result["components"]
    for cat in TIER2_CATEGORIES:
        cat_block = result["components"]["by_category"][cat]
        assert cat_block["score"] == 0
        assert cat_block["tasks_run"] == TIER2_TASK_COUNTS[cat]


# ---- (c) Mixed-credit integration -----------------------------------------

def test_mixed_runners_produce_weighted_breakdown():
    """peft-loops + custom-layers at full credit (90 + 10 = 100 raw);
    curriculum-design + diagnostics at half credit (20 + 30 = 50 raw).
    Total raw 150 / 200 = 75.0 normalized."""
    runners = {
        "peft-loops":        _make_constant_runner(6, 6),  # 15 x 6 = 90
        "custom-layers":     _make_constant_runner(1, 1),  # 10 x 1 = 10
        "curriculum-design": _make_constant_runner(2, 4),  # 10 x 2 = 20
        "diagnostics":       _make_constant_runner(2, 4),  # 15 x 2 = 30
    }
    result = run_tier2(
        model_id="m", tasks_dir=default_tasks_dir(),
        rubric_runners=runners, query_model=_stub_query_model,
    )
    assert result["score"] == 75.0
    assert result["components"]["raw_points"] == 150
    by_cat = result["components"]["by_category"]
    assert by_cat["peft-loops"]["score"]        == 90
    assert by_cat["custom-layers"]["score"]     == 10
    assert by_cat["curriculum-design"]["score"] == 20
    assert by_cat["diagnostics"]["score"]       == 30


# ---- (d) Missing-runner fallback ------------------------------------------

def test_missing_runner_falls_back_to_no_runner_stub_and_surfaces_error():
    """Drop the diagnostics runner; the no-runner stub fires (score 0 with
    explanatory rationale per task), other categories still score normally."""
    runners = {
        "peft-loops":        _make_constant_runner(6, 6),
        "curriculum-design": _make_constant_runner(4, 4),
        "custom-layers":     _make_constant_runner(1, 1),
        # diagnostics intentionally omitted.
    }
    result = run_tier2(
        model_id="m", tasks_dir=default_tasks_dir(),
        rubric_runners=runners, query_model=_stub_query_model,
    )
    # Diagnostics scored zero across all 15 tasks; other categories at max.
    by_cat = result["components"]["by_category"]
    assert by_cat["peft-loops"]["score"]        == 90
    assert by_cat["curriculum-design"]["score"] == 40
    assert by_cat["custom-layers"]["score"]     == 10
    assert by_cat["diagnostics"]["score"]       == 0
    # Diagnostics tasks all RAN (against the no-op stub), they just scored 0.
    assert by_cat["diagnostics"]["tasks_run"] == TIER2_TASK_COUNTS["diagnostics"]
    # Raw 140 / 200 = 70.0.
    assert result["score"] == 70.0
    # The rationale on the diagnostics task results names the missing runner.
    diag_results = by_cat["diagnostics"]["task_results"]
    assert all("no rubric runner registered" in r["rationale"]
               for r in diag_results)


# ---- (e) Runner exception isolation ---------------------------------------

def test_runner_exception_lands_in_task_errors_without_poisoning_siblings():
    """A custom-layers runner that raises should not abort the run;
    its tasks become task_errors entries but other categories score normally."""
    def crash_runner(task, model_output):
        raise RuntimeError(f"boom on {task['task_id']}")

    runners = {
        "peft-loops":        _make_constant_runner(6, 6),
        "curriculum-design": _make_constant_runner(4, 4),
        "custom-layers":     crash_runner,
        "diagnostics":       _make_constant_runner(4, 4),
    }
    result = run_tier2(
        model_id="m", tasks_dir=default_tasks_dir(),
        rubric_runners=runners, query_model=_stub_query_model,
    )
    # custom-layers scored 0 (every task raised); siblings normal.
    by_cat = result["components"]["by_category"]
    assert by_cat["peft-loops"]["score"]        == 90
    assert by_cat["curriculum-design"]["score"] == 40
    assert by_cat["custom-layers"]["score"]     == 0
    assert by_cat["custom-layers"]["tasks_run"] == 0  # all 10 errored, none ran
    assert by_cat["diagnostics"]["score"]       == 60

    # Errors surfaced under task_errors.custom-layers.
    errors = result["components"]["errors"]
    assert "task_errors" in errors
    assert "custom-layers" in errors["task_errors"]
    cl_errors = errors["task_errors"]["custom-layers"]
    assert len(cl_errors) == TIER2_TASK_COUNTS["custom-layers"]
    for line in cl_errors:
        assert "RuntimeError" in line and "boom on" in line

    # Raw 190 / 200 = 95.0.
    assert result["score"] == 95.0


# ---- Real runners wired through the dispatcher (smoke) --------------------

def test_real_runners_wired_with_mocked_dependencies():
    """Wire the actual L6/L8/L11 runners into the dispatcher with their
    dependencies mocked. This is a wiring smoke - verifies the runner
    signatures match the dispatcher's expectation without exercising
    torch / subprocesses / network."""

    # peft_loop in static mode scores 0 per task (rungs 5-6 skipped),
    # but we can mock the score_peft_loop directly to return max.
    def peft_runner(task, output):
        return RubricResult(score=6, max_score=6, rationale="mocked")

    def cl_runner(task, output):
        return RubricResult(score=1, max_score=1, rationale="mocked")

    def positive_judge(prompt, model_id, base_url, timeout):
        # All four axes -> 1
        axes_set = {
            "curriculum-design": ('"completeness": 1, "ordering": 1, '
                                  '"specificity": 1, "plausibility": 1'),
            "diagnostics":       ('"correct_root_cause": 1, "viable_fix": 1, '
                                  '"no_new_bug": 1, '
                                  '"mechanistic_explanation": 1'),
        }
        # The judge prompt includes the task prompt; we can sniff it to
        # decide which set to return. Simpler: return BOTH so both work.
        return ('{"completeness": 1, "ordering": 1, "specificity": 1, '
                '"plausibility": 1, "correct_root_cause": 1, '
                '"viable_fix": 1, "no_new_bug": 1, '
                '"mechanistic_explanation": 1}')

    def cd_runner(task, output):
        return llm_judge.score_llm_judge(task, output, judge_query=positive_judge)

    def diag_runner(task, output):
        return llm_judge.score_llm_judge(task, output, judge_query=positive_judge)

    runners = {
        "peft-loops":        peft_runner,
        "custom-layers":     cl_runner,
        "curriculum-design": cd_runner,
        "diagnostics":       diag_runner,
    }
    result = run_tier2(
        model_id="m", tasks_dir=default_tasks_dir(),
        rubric_runners=runners, query_model=_stub_query_model,
    )
    assert result["score"] == 100.0
    assert "errors" not in result["components"]


def test_real_peft_runner_static_mode_through_dispatcher():
    """Wire the actual peft_loop runner (static mode) into the dispatcher.
    Each L5 reference solution scores 4/6 in static mode -> 60 raw points
    for peft-loops. Other categories use stub runners at max."""

    def peft_static_runner(task, output):
        return peft_loop.score_peft_loop(task, output, sandbox=False)

    # query_model returns the reference solution so the runner has good code.
    def query_returning_reference(prompt, model_id, base_url):
        # We don't have task context here, so reconstruct by stashing.
        # Simpler approach: stash the last task's reference via a closure.
        return query_returning_reference.last_reference

    query_returning_reference.last_reference = ""

    # Instead, wire it via a tiny wrapper that runs the runner against
    # the task's OWN reference - integration check that runner+dispatcher
    # agree on the task-format contract.
    def peft_against_reference(task, output):
        # Use the reference solution as the candidate, ignore stub output.
        return peft_loop.score_peft_loop(
            task, task["reference_solution"], sandbox=False,
        )

    runners = {
        "peft-loops":        peft_against_reference,
        "curriculum-design": _make_constant_runner(4, 4),
        "custom-layers":     _make_constant_runner(1, 1),
        "diagnostics":       _make_constant_runner(4, 4),
    }
    result = run_tier2(
        model_id="m", tasks_dir=default_tasks_dir(),
        rubric_runners=runners, query_model=_stub_query_model,
    )
    # peft-loops static -> 4/6 per task x 15 tasks = 60.
    # Other categories at max: 40 + 10 + 60 = 110.
    # Total: 170/200 = 85.0.
    by_cat = result["components"]["by_category"]
    assert by_cat["peft-loops"]["score"] == 60
    assert result["score"] == 85.0
