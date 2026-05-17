# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
"""On-disk validation of the custom-layers Tier 2 task corpus.

Mirrors ``tests/test_tier2_peft_corpus.py``: validates the structural
invariants the v1 spec and the L8 runner will rely on, WITHOUT
executing the references (that requires torch + pytest in a
subprocess - which the L8 runner does on the pod).

Catches:
  * count drift (must be 10 per v1)
  * duplicate task_id
  * malformed rubric (wrong type, max_score != 1, criterion id mismatch)
  * reference_solution / unit_tests that don't parse with ast.parse
  * unit_tests that don't reference ``from solution import`` (the L8
    runner writes the candidate code as ``solution.py``; tests must
    import from there)
  * reference_solution that doesn't import torch (sanity check that
    these are PyTorch tasks, not Python-only stubs)
"""

from __future__ import annotations

import ast

import pytest

from lamarck.eval.tier2_engineering import (
    TIER2_MAX_POINTS_PER_TASK,
    TIER2_TASK_COUNTS,
    default_tasks_dir,
    load_tasks,
)

CATEGORY = "custom-layers"


@pytest.fixture(scope="module")
def cl_tasks():
    return load_tasks(default_tasks_dir())[CATEGORY]


def test_cl_corpus_count_matches_v1_spec(cl_tasks):
    """v1 locks this at 10. Any drift is a corpus bug."""
    assert len(cl_tasks) == TIER2_TASK_COUNTS[CATEGORY] == 10


def test_cl_task_ids_are_unique(cl_tasks):
    ids = [t["task_id"] for t in cl_tasks]
    assert len(set(ids)) == len(ids), f"duplicate task_id: {ids}"


def test_cl_task_ids_follow_naming_convention(cl_tasks):
    """task_id must start with cl_NNN_ for stable sort + grep."""
    for t in cl_tasks:
        assert t["task_id"].startswith("cl_"), (
            f"task_id {t['task_id']!r} doesn't follow cl_NNN_ convention"
        )


def test_cl_categories_are_uniform(cl_tasks):
    for t in cl_tasks:
        assert t["category"] == CATEGORY


def test_cl_prompts_are_substantive(cl_tasks):
    for t in cl_tasks:
        assert len(t["prompt"]) >= 100, (
            f"task {t['task_id']!r} prompt only {len(t['prompt'])} chars"
        )


def test_cl_rubric_is_binary(cl_tasks):
    expected_max = TIER2_MAX_POINTS_PER_TASK[CATEGORY]  # 1
    for t in cl_tasks:
        rubric = t["rubric"]
        assert rubric["type"] == "unit-tests-binary", (
            f"task {t['task_id']!r} rubric.type {rubric['type']!r}"
        )
        assert rubric["max_score"] == expected_max, (
            f"task {t['task_id']!r} rubric.max_score "
            f"{rubric['max_score']} != {expected_max}"
        )
        criteria = rubric["criteria"]
        assert len(criteria) == 1, (
            f"task {t['task_id']!r} has {len(criteria)} criteria, "
            f"expected 1"
        )
        crit = criteria[0]
        assert crit["id"] == "tests_pass", (
            f"task {t['task_id']!r} criterion id {crit['id']!r}"
        )
        assert crit["points"] == 1


def test_cl_reference_solutions_parse(cl_tasks):
    for t in cl_tasks:
        try:
            ast.parse(t["reference_solution"])
        except SyntaxError as exc:
            pytest.fail(
                f"task {t['task_id']!r} reference_solution does not "
                f"parse: {exc}"
            )


def test_cl_reference_solutions_import_torch(cl_tasks):
    """Sanity: these are PyTorch tasks, not Python-only stubs."""
    for t in cl_tasks:
        assert "torch" in t["reference_solution"], (
            f"task {t['task_id']!r} reference does not mention torch"
        )


def test_cl_unit_tests_are_present_and_parse(cl_tasks):
    """Every custom-layers task ships a `unit_tests` field of pytest
    source. The L8 runner writes this to test_solution.py."""
    for t in cl_tasks:
        assert "unit_tests" in t, (
            f"task {t['task_id']!r} missing the unit_tests field"
        )
        unit_tests = t["unit_tests"]
        assert isinstance(unit_tests, str) and unit_tests.strip(), (
            f"task {t['task_id']!r} has empty unit_tests"
        )
        try:
            ast.parse(unit_tests)
        except SyntaxError as exc:
            pytest.fail(
                f"task {t['task_id']!r} unit_tests does not parse: {exc}"
            )


def test_cl_unit_tests_import_from_solution(cl_tasks):
    """L8 runner writes candidate code as solution.py - the tests must
    import from there or the harness can't reach the candidate."""
    for t in cl_tasks:
        assert "from solution import" in t["unit_tests"], (
            f"task {t['task_id']!r} unit_tests does not "
            f"`from solution import ...`"
        )


def test_cl_unit_tests_define_test_functions(cl_tasks):
    """pytest discovers def test_*() - bare assert scripts don't count."""
    for t in cl_tasks:
        tree = ast.parse(t["unit_tests"])
        test_fns = [
            n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
        ]
        assert test_fns, (
            f"task {t['task_id']!r} unit_tests has no def test_*() functions"
        )
