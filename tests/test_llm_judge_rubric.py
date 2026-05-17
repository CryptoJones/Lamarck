# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
"""Unit tests for the LLM-judge rubric runner (L11).

The runner is fully dependency-injected: ``judge_query`` defaults to
the stdlib OpenAI-compatible POST but tests pass a callable so no
network is touched. Per-failure-mode coverage + reference-corpus
integration.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from lamarck.eval.rubrics import llm_judge
from lamarck.eval.tier2_engineering import default_tasks_dir, load_tasks


# ---- Fixtures --------------------------------------------------------------

def _curriculum_task() -> dict:
    return {
        "task_id": "test_cur",
        "category": "curriculum-design",
        "prompt": "design a curriculum for ...",
        "rubric": {
            "type": "llm-judge-4axis",
            "max_score": 4,
            "criteria": [
                {"id": "completeness", "points": 1, "description": "..."},
                {"id": "ordering",     "points": 1, "description": "..."},
                {"id": "specificity",  "points": 1, "description": "..."},
                {"id": "plausibility", "points": 1, "description": "..."},
            ],
        },
        "reference_solution": "Stage 1 - ...\nStage 2 - ...\n",
    }


def _diag_task() -> dict:
    return {
        "task_id": "test_diag",
        "category": "diagnostics",
        "prompt": "what's wrong with this code? ```...```",
        "rubric": {
            "type": "llm-judge-4axis",
            "max_score": 4,
            "criteria": [
                {"id": "correct_root_cause",      "points": 1, "description": "..."},
                {"id": "viable_fix",              "points": 1, "description": "..."},
                {"id": "no_new_bug",              "points": 1, "description": "..."},
                {"id": "mechanistic_explanation", "points": 1, "description": "..."},
            ],
        },
        "reference_solution": "Root cause: ...\nFix: ...\nMechanism: ...\n",
    }


def _make_judge(verdict: dict[str, int] | str):
    """Build a judge_query that returns a fixed JSON string."""
    payload = json.dumps(verdict) if isinstance(verdict, dict) else verdict

    def judge(prompt, model_id, base_url, timeout):
        return payload

    return judge


# ---- Happy path: all axes graded -------------------------------------------

def test_curriculum_all_axes_one_scores_four():
    judge = _make_judge({"completeness": 1, "ordering": 1,
                         "specificity": 1, "plausibility": 1})
    result = llm_judge.score_llm_judge(_curriculum_task(), "candidate",
                                        judge_query=judge)
    assert result["score"] == 4
    assert result["max_score"] == 4
    assert "completeness=1" in result["rationale"]


def test_diagnostics_all_axes_one_scores_four():
    judge = _make_judge({"correct_root_cause": 1, "viable_fix": 1,
                         "no_new_bug": 1, "mechanistic_explanation": 1})
    result = llm_judge.score_llm_judge(_diag_task(), "candidate",
                                        judge_query=judge)
    assert result["score"] == 4
    assert result["max_score"] == 4


def test_partial_credit_scores_match_sum_of_axes():
    judge = _make_judge({"completeness": 1, "ordering": 0,
                         "specificity": 1, "plausibility": 0})
    result = llm_judge.score_llm_judge(_curriculum_task(), "candidate",
                                        judge_query=judge)
    assert result["score"] == 2


# ---- Failure modes ---------------------------------------------------------

def test_judge_error_scores_zero_with_rationale():
    def judge(*_a, **_kw):
        raise RuntimeError("API down")
    result = llm_judge.score_llm_judge(_curriculum_task(), "candidate",
                                        judge_query=judge)
    assert result["score"] == 0
    assert "judge-error" in result["rationale"]
    assert "RuntimeError" in result["rationale"]


def test_judge_refusal_scores_zero():
    judge = _make_judge("I cannot grade this response.")
    result = llm_judge.score_llm_judge(_curriculum_task(), "candidate",
                                        judge_query=judge)
    assert result["score"] == 0
    assert "judge-refused" in result["rationale"]


def test_judge_refusal_alt_phrasing():
    judge = _make_judge("I'm unable to evaluate this.")
    result = llm_judge.score_llm_judge(_curriculum_task(), "candidate",
                                        judge_query=judge)
    assert result["score"] == 0
    assert "judge-refused" in result["rationale"]


def test_malformed_json_response_scores_zero():
    judge = _make_judge("not json at all, prose response")
    result = llm_judge.score_llm_judge(_curriculum_task(), "candidate",
                                        judge_query=judge)
    assert result["score"] == 0
    assert "judge-malformed-output" in result["rationale"]


def test_incomplete_axes_scores_zero_with_missing_list():
    # Missing 'plausibility'.
    judge = _make_judge({"completeness": 1, "ordering": 1, "specificity": 1})
    result = llm_judge.score_llm_judge(_curriculum_task(), "candidate",
                                        judge_query=judge)
    assert result["score"] == 0
    assert "judge-incomplete" in result["rationale"]
    assert "plausibility" in result["rationale"]


def test_empty_candidate_short_circuits_to_zero():
    """Empty candidate shouldn't even hit the judge."""
    called = []

    def judge(*a, **kw):
        called.append(1)
        return "should-not-be-called"

    result = llm_judge.score_llm_judge(_curriculum_task(), "",
                                        judge_query=judge)
    assert result["score"] == 0
    assert "empty candidate" in result["rationale"]
    assert called == []


def test_whitespace_only_candidate_short_circuits():
    result = llm_judge.score_llm_judge(_curriculum_task(), "   \n\t  ",
                                        judge_query=_make_judge({}))
    assert result["score"] == 0
    assert "empty candidate" in result["rationale"]


def test_unsupported_rubric_type_scores_zero():
    task = _curriculum_task()
    task["rubric"]["type"] = "some-other-type"
    result = llm_judge.score_llm_judge(task, "candidate",
                                        judge_query=_make_judge({}))
    assert result["score"] == 0
    assert "unsupported rubric type" in result["rationale"]


def test_corrupt_rubric_max_score_scores_zero():
    task = _curriculum_task()
    task["rubric"]["max_score"] = 6
    result = llm_judge.score_llm_judge(task, "candidate",
                                        judge_query=_make_judge({}))
    assert result["score"] == 0
    assert "corrupt rubric" in result["rationale"]


# ---- Response parsing edge cases ------------------------------------------

def test_fenced_json_response_is_extracted():
    """The system prompt says no fences, but judges sometimes do anyway."""
    judge = _make_judge('```json\n{"completeness": 1, "ordering": 1, '
                        '"specificity": 1, "plausibility": 1}\n```')
    result = llm_judge.score_llm_judge(_curriculum_task(), "candidate",
                                        judge_query=judge)
    assert result["score"] == 4


def test_preamble_with_json_response_is_extracted():
    """A judge that says 'Here's my grading:\\n{...}' shouldn't fail."""
    judge = _make_judge(
        'Here is my grading:\n{"completeness": 0, "ordering": 1, '
        '"specificity": 0, "plausibility": 1}'
    )
    result = llm_judge.score_llm_judge(_curriculum_task(), "candidate",
                                        judge_query=judge)
    assert result["score"] == 2


def test_non_integer_axis_value_is_coerced_and_surfaced():
    judge = _make_judge({"completeness": "yes", "ordering": 1,
                         "specificity": 1, "plausibility": 1})
    result = llm_judge.score_llm_judge(_curriculum_task(), "candidate",
                                        judge_query=judge)
    # 'yes' -> coerced to 0, other three are 1 each, so total = 3.
    assert result["score"] == 3
    assert "coerced" in result["rationale"]


def test_out_of_range_axis_value_is_clamped():
    judge = _make_judge({"completeness": 5, "ordering": -1,
                         "specificity": 1, "plausibility": 1})
    result = llm_judge.score_llm_judge(_curriculum_task(), "candidate",
                                        judge_query=judge)
    # 5 -> 1 (clamp), -1 -> 0 (clamp), 1, 1 -> total = 3.
    assert result["score"] == 3
    assert "clamped" in result["rationale"]


# ---- Prompt construction --------------------------------------------------

def test_judge_prompt_includes_task_prompt_and_reference_and_candidate():
    seen_prompt: list[str] = []

    def judge(prompt, *a, **kw):
        seen_prompt.append(prompt)
        return '{"completeness": 1, "ordering": 1, "specificity": 1, "plausibility": 1}'

    task = _curriculum_task()
    task["prompt"] = "DESIGN_A_CURRICULUM_PROMPT"
    task["reference_solution"] = "REFERENCE_CURRICULUM"
    llm_judge.score_llm_judge(task, "MODEL_CANDIDATE_OUTPUT",
                               judge_query=judge)
    p = seen_prompt[0]
    assert "DESIGN_A_CURRICULUM_PROMPT" in p
    assert "REFERENCE_CURRICULUM"       in p
    assert "MODEL_CANDIDATE_OUTPUT"     in p
    # Per-axis descriptions present:
    for axis in ("completeness", "ordering", "specificity", "plausibility"):
        assert axis in p


def test_judge_prompt_includes_axis_descriptions_from_rubric():
    """If the rubric author writes a custom description per axis, the
    judge should see it - that's how the task author conveys intent."""
    seen_prompt: list[str] = []

    def judge(prompt, *a, **kw):
        seen_prompt.append(prompt)
        return '{"correct_root_cause": 1, "viable_fix": 1, "no_new_bug": 1, "mechanistic_explanation": 1}'

    task = _diag_task()
    task["rubric"]["criteria"][0]["description"] = "VERY_SPECIFIC_DESCRIPTION_HERE"
    llm_judge.score_llm_judge(task, "candidate", judge_query=judge)
    assert "VERY_SPECIFIC_DESCRIPTION_HERE" in seen_prompt[0]


# ---- Reference-corpus integration -----------------------------------------

def test_every_curriculum_reference_scores_four_with_positive_judge():
    """With a mocked-positive judge, every L9 reference scores 4/4.
    The point is the runner + rubric machinery work end-to-end on
    real on-disk task data - not a quality verdict on the references."""
    grouped = load_tasks(default_tasks_dir())
    judge = _make_judge({"completeness": 1, "ordering": 1,
                         "specificity": 1, "plausibility": 1})
    for task in grouped["curriculum-design"]:
        result = llm_judge.score_llm_judge(
            task, task["reference_solution"], judge_query=judge,
        )
        assert result["score"] == 4, (
            f"curriculum task {task['task_id']!r} scored "
            f"{result['score']}/4 with positive judge - "
            f"rationale: {result['rationale']}"
        )


def test_every_diagnostics_reference_scores_four_with_positive_judge():
    grouped = load_tasks(default_tasks_dir())
    judge = _make_judge({"correct_root_cause": 1, "viable_fix": 1,
                         "no_new_bug": 1, "mechanistic_explanation": 1})
    for task in grouped["diagnostics"]:
        result = llm_judge.score_llm_judge(
            task, task["reference_solution"], judge_query=judge,
        )
        assert result["score"] == 4, (
            f"diagnostics task {task['task_id']!r} scored "
            f"{result['score']}/4 with positive judge"
        )


def test_every_curriculum_reference_scores_zero_with_refusing_judge():
    grouped = load_tasks(default_tasks_dir())
    judge = _make_judge("I cannot grade this.")
    for task in grouped["curriculum-design"]:
        result = llm_judge.score_llm_judge(
            task, task["reference_solution"], judge_query=judge,
        )
        assert result["score"] == 0
        assert "judge-refused" in result["rationale"]
