# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
"""On-disk validation of the PEFT/LoRA Tier 2 task corpus.

Loads the actual ``src/lamarck/eval/tier2_tasks/peft-loops/`` corpus
shipped in L5 and asserts the structural invariants the v1 spec and
the L6 runner will rely on. If any of these fail, the corpus is
broken and Tier 2 scoring will mis-grade silently.

Catches:
  * count drift (must be 15 per v1 spec, including future "we added
    a task!" accidents that would corrupt the locked weighting)
  * duplicate task_id
  * malformed rubric block (wrong type, wrong max_score, wrong
    number of criteria)
  * reference_solution that doesn't parse with ast.parse
  * reference_solution that's missing the imports / config /
    trainer call the L6 6-point ladder will look for
"""

from __future__ import annotations

import ast

import pytest

from lamarck.eval.tier2_engineering import (
    TIER2_TASK_COUNTS,
    TIER2_MAX_POINTS_PER_TASK,
    default_tasks_dir,
    load_tasks,
)

CATEGORY = "peft-loops"


@pytest.fixture(scope="module")
def peft_tasks():
    grouped = load_tasks(default_tasks_dir())
    return grouped[CATEGORY]


def test_peft_corpus_count_matches_v1_spec(peft_tasks):
    """v1 locks this at 15. Any drift is a corpus bug."""
    assert len(peft_tasks) == TIER2_TASK_COUNTS[CATEGORY] == 15


def test_peft_task_ids_are_unique(peft_tasks):
    ids = [t["task_id"] for t in peft_tasks]
    assert len(set(ids)) == len(ids), f"duplicate task_id present: {ids}"


def test_peft_task_ids_follow_naming_convention(peft_tasks):
    """task_id must start with peft_NNN_ for stable sort + grep."""
    for t in peft_tasks:
        assert t["task_id"].startswith("peft_"), (
            f"task_id {t['task_id']!r} doesn't follow peft_NNN_ convention"
        )


def test_peft_categories_are_uniform(peft_tasks):
    for t in peft_tasks:
        assert t["category"] == CATEGORY


def test_peft_prompts_are_substantive(peft_tasks):
    """Prompts must be at least 100 chars - one-liners are too vague to grade."""
    for t in peft_tasks:
        assert len(t["prompt"]) >= 100, (
            f"task {t['task_id']!r} prompt is only {len(t['prompt'])} chars"
        )


def test_peft_rubric_is_six_point_ladder(peft_tasks):
    expected_max = TIER2_MAX_POINTS_PER_TASK[CATEGORY]  # 6
    for t in peft_tasks:
        rubric = t["rubric"]
        assert rubric["type"] == "programmatic-6pt-ladder", (
            f"task {t['task_id']!r} has rubric.type {rubric['type']!r}"
        )
        assert rubric["max_score"] == expected_max, (
            f"task {t['task_id']!r} rubric.max_score "
            f"{rubric['max_score']} != {expected_max}"
        )
        criteria = rubric["criteria"]
        assert len(criteria) == 6, (
            f"task {t['task_id']!r} has {len(criteria)} criteria, "
            f"expected 6"
        )
        # Sum of points across criteria must equal max_score.
        total_pts = sum(c["points"] for c in criteria)
        assert total_pts == expected_max, (
            f"task {t['task_id']!r} criteria sum to {total_pts}, "
            f"expected {expected_max}"
        )
        # Each rung of the ladder has a stable id the L6 runner
        # will key off; locking them here means the corpus and
        # the runner can't drift apart silently.
        ladder_ids = {c["id"] for c in criteria}
        assert ladder_ids == {
            "parses", "imports", "lora_config",
            "trainer_train_called", "runs_to_completion",
            "adapter_loads",
        }, f"task {t['task_id']!r} ladder ids: {ladder_ids}"


def test_peft_reference_solutions_parse(peft_tasks):
    """If a reference solution doesn't parse, the rubric runner can't
    use it as a known-good baseline."""
    for t in peft_tasks:
        try:
            ast.parse(t["reference_solution"])
        except SyntaxError as exc:
            pytest.fail(
                f"task {t['task_id']!r} reference_solution does not "
                f"parse: {exc}"
            )


def test_peft_reference_solutions_hit_all_six_rungs(peft_tasks):
    """Every reference solution must satisfy each of the 6 rungs
    the L6 programmatic rubric checks for. A reference that doesn't
    score 6/6 is not a reference."""
    required = {
        # rung 2: imports
        "imports_peft":         "peft",
        "imports_transformers": "transformers",
        "imports_trl_or_trainer": "trl",  # trl OR transformers.Trainer
        # rung 3: LoraConfig instantiated
        "lora_config_call":     "LoraConfig(",
        # rung 4: Trainer.train() / SFTTrainer.train() call
        "trainer_train":        ".train()",
        # rung 6: adapter loads via PeftModel.from_pretrained
        "peft_model_from_pretrained": "PeftModel.from_pretrained",
    }
    for t in peft_tasks:
        src = t["reference_solution"]
        for label, needle in required.items():
            assert needle in src, (
                f"task {t['task_id']!r} reference_solution missing "
                f"{label!r} marker ({needle!r})"
            )
