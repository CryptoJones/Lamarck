# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
"""On-disk validation of the diagnostics Tier 2 task corpus.

Mirrors the L5 / L7 / L9 corpus tests. Locks the 15-task count and
the structural invariants the L11 LLM-judge runner relies on.

Catches:
  * count drift (must be 15 per v1)
  * duplicate task_id
  * malformed rubric (wrong type, max_score != 4, missing axes)
  * prompt that lacks a script or a traceback / failure description
  * reference_solution that doesn't name a root cause AND a fix
"""

from __future__ import annotations

import pytest

from lamarck.eval.tier2_engineering import (
    TIER2_MAX_POINTS_PER_TASK,
    TIER2_TASK_COUNTS,
    default_tasks_dir,
    load_tasks,
)

CATEGORY = "diagnostics"
EXPECTED_AXES = {
    "correct_root_cause",
    "viable_fix",
    "no_new_bug",
    "mechanistic_explanation",
}


@pytest.fixture(scope="module")
def diag_tasks():
    return load_tasks(default_tasks_dir())[CATEGORY]


def test_diag_corpus_count_matches_v1_spec(diag_tasks):
    """v1 locks this at 15. Any drift is a corpus bug."""
    assert len(diag_tasks) == TIER2_TASK_COUNTS[CATEGORY] == 15


def test_diag_task_ids_are_unique(diag_tasks):
    ids = [t["task_id"] for t in diag_tasks]
    assert len(set(ids)) == len(ids), f"duplicate task_id: {ids}"


def test_diag_task_ids_follow_naming_convention(diag_tasks):
    """task_id must start with diag_NNN_ for stable sort + grep."""
    for t in diag_tasks:
        assert t["task_id"].startswith("diag_"), (
            f"task_id {t['task_id']!r} doesn't follow diag_NNN_ convention"
        )


def test_diag_categories_are_uniform(diag_tasks):
    for t in diag_tasks:
        assert t["category"] == CATEGORY


def test_diag_prompts_are_substantive(diag_tasks):
    """Diagnostic prompts ship a script + a traceback / failure mode.
    Bump the lower bound to >= 250 chars."""
    for t in diag_tasks:
        assert len(t["prompt"]) >= 250, (
            f"task {t['task_id']!r} prompt only {len(t['prompt'])} chars"
        )


def test_diag_prompts_contain_a_code_block(diag_tasks):
    """Every diagnostic prompt must show the failing code in a fenced
    block - 'find the bug in this idea' is not a diagnostic task."""
    for t in diag_tasks:
        prompt = t["prompt"]
        assert "```" in prompt, (
            f"task {t['task_id']!r} prompt has no fenced code block"
        )


def test_diag_prompts_describe_a_failure_mode(diag_tasks):
    """Either a traceback line or a quantitative failure description
    must be present in the prompt - otherwise there's nothing to diagnose."""
    failure_markers = (
        "traceback", "error", "loss", "oom", "nan", "crash", "fail",
        "slow", "longer", "no progress", "regress",
    )
    for t in diag_tasks:
        prompt_lower = t["prompt"].lower()
        assert any(m in prompt_lower for m in failure_markers), (
            f"task {t['task_id']!r} prompt has no recognisable failure "
            f"description ({failure_markers})"
        )


def test_diag_rubric_is_llm_judge_4axis(diag_tasks):
    expected_max = TIER2_MAX_POINTS_PER_TASK[CATEGORY]  # 4
    for t in diag_tasks:
        rubric = t["rubric"]
        assert rubric["type"] == "llm-judge-4axis", (
            f"task {t['task_id']!r} rubric.type {rubric['type']!r}"
        )
        assert rubric["max_score"] == expected_max
        criteria = rubric["criteria"]
        assert len(criteria) == 4
        axis_ids = {c["id"] for c in criteria}
        assert axis_ids == EXPECTED_AXES, (
            f"task {t['task_id']!r} axis ids {axis_ids} != {EXPECTED_AXES}"
        )
        for c in criteria:
            assert c["points"] == 1


def test_diag_reference_solutions_are_substantive(diag_tasks):
    """A reference root-cause + fix needs real substance; reject any
    reference under 400 chars."""
    for t in diag_tasks:
        ref = t["reference_solution"]
        assert len(ref) >= 400, (
            f"task {t['task_id']!r} reference is only {len(ref)} chars"
        )


def test_diag_reference_names_root_cause_and_fix(diag_tasks):
    """Lock the two-section structure: every reference must label a
    root cause AND a fix block."""
    for t in diag_tasks:
        ref_lower = t["reference_solution"].lower()
        assert "root cause" in ref_lower, (
            f"task {t['task_id']!r} reference doesn't name a root cause"
        )
        assert "fix" in ref_lower, (
            f"task {t['task_id']!r} reference doesn't propose a fix"
        )


def test_diag_reference_has_mechanism_explanation(diag_tasks):
    """The 4th rubric axis is mechanistic_explanation - the reference
    must include WHY the bug happens, not just WHAT to change."""
    for t in diag_tasks:
        ref_lower = t["reference_solution"].lower()
        assert "mechanism" in ref_lower, (
            f"task {t['task_id']!r} reference lacks a Mechanism section"
        )
