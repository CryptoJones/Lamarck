# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
"""On-disk validation of the curriculum-design Tier 2 task corpus.

Mirrors the L5 / L7 corpus tests: validates the structural invariants
the v1 spec and the L11 LLM-judge runner will rely on, WITHOUT running
a real LLM judge (that's wired in L11 + L12).

Catches:
  * count drift (must be 10 per v1)
  * duplicate task_id
  * malformed rubric (wrong type, max_score != 4, missing axes)
  * reference_solution that's empty or trivially short
"""

from __future__ import annotations

import pytest

from lamarck.eval.tier2_engineering import (
    TIER2_MAX_POINTS_PER_TASK,
    TIER2_TASK_COUNTS,
    default_tasks_dir,
    load_tasks,
)

CATEGORY = "curriculum-design"
EXPECTED_AXES = {"completeness", "ordering", "specificity", "plausibility"}


@pytest.fixture(scope="module")
def cd_tasks():
    return load_tasks(default_tasks_dir())[CATEGORY]


def test_cd_corpus_count_matches_v1_spec(cd_tasks):
    """v1 locks this at 10. Any drift is a corpus bug."""
    assert len(cd_tasks) == TIER2_TASK_COUNTS[CATEGORY] == 10


def test_cd_task_ids_are_unique(cd_tasks):
    ids = [t["task_id"] for t in cd_tasks]
    assert len(set(ids)) == len(ids), f"duplicate task_id: {ids}"


def test_cd_task_ids_follow_naming_convention(cd_tasks):
    """task_id must start with cd_NNN_ for stable sort + grep."""
    for t in cd_tasks:
        assert t["task_id"].startswith("cd_"), (
            f"task_id {t['task_id']!r} doesn't follow cd_NNN_ convention"
        )


def test_cd_categories_are_uniform(cd_tasks):
    for t in cd_tasks:
        assert t["category"] == CATEGORY


def test_cd_prompts_are_substantive(cd_tasks):
    """Curriculum prompts need to specify what's being trained; bump
    to >= 200 chars so a one-liner doesn't slip in."""
    for t in cd_tasks:
        assert len(t["prompt"]) >= 200, (
            f"task {t['task_id']!r} prompt only {len(t['prompt'])} chars"
        )


def test_cd_rubric_is_llm_judge_4axis(cd_tasks):
    expected_max = TIER2_MAX_POINTS_PER_TASK[CATEGORY]  # 4
    for t in cd_tasks:
        rubric = t["rubric"]
        assert rubric["type"] == "llm-judge-4axis", (
            f"task {t['task_id']!r} rubric.type {rubric['type']!r}"
        )
        assert rubric["max_score"] == expected_max, (
            f"task {t['task_id']!r} rubric.max_score "
            f"{rubric['max_score']} != {expected_max}"
        )
        criteria = rubric["criteria"]
        assert len(criteria) == 4, (
            f"task {t['task_id']!r} has {len(criteria)} criteria, "
            f"expected 4"
        )
        axis_ids = {c["id"] for c in criteria}
        assert axis_ids == EXPECTED_AXES, (
            f"task {t['task_id']!r} axis ids {axis_ids} != {EXPECTED_AXES}"
        )
        for c in criteria:
            assert c["points"] == 1, (
                f"task {t['task_id']!r} axis {c['id']!r} points "
                f"{c['points']} != 1"
            )
        total = sum(c["points"] for c in criteria)
        assert total == expected_max


def test_cd_reference_solutions_are_substantive(cd_tasks):
    """Curricula need multi-stage substance; flag any reference that's
    too short to plausibly contain prereqs + target + eval."""
    for t in cd_tasks:
        ref = t["reference_solution"]
        assert len(ref) >= 500, (
            f"task {t['task_id']!r} reference is only {len(ref)} chars - "
            f"probably not a real multi-stage curriculum"
        )


def test_cd_reference_solutions_mention_dataset_and_success(cd_tasks):
    """The v1 spec says each stage names a dataset + success criterion.
    These are textual references, but we can sanity-check both keywords
    appear somewhere in the reference."""
    for t in cd_tasks:
        ref_lower = t["reference_solution"].lower()
        assert "dataset" in ref_lower or "source:" in ref_lower, (
            f"task {t['task_id']!r} reference mentions neither dataset "
            f"nor source"
        )
        assert "success" in ref_lower or "eval" in ref_lower, (
            f"task {t['task_id']!r} reference mentions neither success "
            f"criterion nor eval"
        )


def test_cd_reference_solutions_have_multiple_stages(cd_tasks):
    """A 1-stage curriculum is not a curriculum. Require >= 3 stage
    markers in each reference."""
    for t in cd_tasks:
        ref = t["reference_solution"]
        stage_markers = ref.lower().count("stage ")
        assert stage_markers >= 3, (
            f"task {t['task_id']!r} reference has only {stage_markers} "
            f"'Stage N' markers"
        )
