# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
"""Locked-invariant smoke tests for the Tier 2 framework.

Full mocked integration tests (with synthetic tasks and stubbed
runners) land in L12 as ``tests/test_tier2.py``. This file's job
is to lock the v1 numeric contract so a silent edit to category
counts, per-task ceilings, or aggregate raw points fails CI
loudly - before any of the L5-L11 task corpora and runners ship.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lamarck.eval import tier2_engineering as t2


# ---- Locked v1 numeric contract --------------------------------------------

def test_tier2_categories_are_locked():
    """v1 ships exactly these four categories, in this order."""
    assert t2.TIER2_CATEGORIES == (
        "peft-loops", "curriculum-design", "custom-layers", "diagnostics",
    )


def test_tier2_task_counts_match_v1_spec():
    """15 + 10 + 10 + 15 = 50 hand-curated tasks."""
    assert t2.TIER2_TASK_COUNTS == {
        "peft-loops":        15,
        "curriculum-design": 10,
        "custom-layers":     10,
        "diagnostics":       15,
    }


def test_tier2_max_points_per_task_match_v1_spec():
    """6-pt programmatic ladder, 4-pt LLM-judge, 1-pt binary."""
    assert t2.TIER2_MAX_POINTS_PER_TASK == {
        "peft-loops":        6,
        "curriculum-design": 4,
        "custom-layers":     1,
        "diagnostics":       4,
    }


def test_tier2_raw_points_per_category_match_task_count_times_max():
    """The hard-coded per-category ceilings must match count x max.

    The lookups are deliberately separate so this test exists - any
    drift between them is the bug we want to catch.
    """
    for cat in t2.TIER2_CATEGORIES:
        expected = t2.TIER2_TASK_COUNTS[cat] * t2.TIER2_MAX_POINTS_PER_TASK[cat]
        assert t2.TIER2_RAW_POINTS_PER_CATEGORY[cat] == expected, (
            f"category {cat!r} raw-points mismatch: "
            f"{t2.TIER2_RAW_POINTS_PER_CATEGORY[cat]} != "
            f"{t2.TIER2_TASK_COUNTS[cat]} x {t2.TIER2_MAX_POINTS_PER_TASK[cat]}"
        )


def test_tier2_total_raw_points_is_two_hundred():
    """v1 spec: 90 + 40 + 10 + 60 = 200 raw points before /100 norm."""
    assert t2.TIER2_TOTAL_RAW_POINTS == 200
    assert sum(t2.TIER2_RAW_POINTS_PER_CATEGORY.values()) == 200


def test_programmatic_vs_llm_judge_split_is_locked():
    """v1 spec: 2a + 2c are programmatic; 2b + 2d are LLM-judge."""
    assert set(t2.PROGRAMMATIC_CATEGORIES) == {"peft-loops", "custom-layers"}
    assert set(t2.LLM_JUDGE_CATEGORIES)   == {"curriculum-design", "diagnostics"}
    # Every category must be in exactly one bucket.
    assert (
        set(t2.PROGRAMMATIC_CATEGORIES) | set(t2.LLM_JUDGE_CATEGORIES)
    ) == set(t2.TIER2_CATEGORIES)
    assert (
        set(t2.PROGRAMMATIC_CATEGORIES) & set(t2.LLM_JUDGE_CATEGORIES)
    ) == set()


# ---- Runner dispatch -------------------------------------------------------

def test_resolve_runner_returns_injected_runner_when_registered():
    sentinel = lambda task, output: t2.RubricResult(
        score=1, max_score=1, rationale="sentinel"
    )
    runner = t2.resolve_runner("custom-layers", {"custom-layers": sentinel})
    assert runner is sentinel


def test_resolve_runner_falls_back_to_no_op_when_unregistered():
    runner = t2.resolve_runner("peft-loops", rubric_runners=None)
    result = runner({"task_id": "x", "category": "peft-loops",
                     "prompt": "", "rubric": {}, "reference_solution": ""},
                    model_output="anything")
    assert result["score"] == 0
    assert result["max_score"] == t2.TIER2_MAX_POINTS_PER_TASK["peft-loops"]
    assert "no rubric runner registered" in result["rationale"]


def test_resolve_runner_rejects_unknown_category():
    """Mis-named category -> immediate KeyError, no silent acceptance."""
    with pytest.raises(KeyError):
        t2.resolve_runner("not-a-real-category", rubric_runners=None)


# ---- Task loading ----------------------------------------------------------

def test_load_tasks_missing_dir_returns_empty_lists(tmp_path: Path):
    grouped = t2.load_tasks(tmp_path / "does-not-exist")
    assert grouped == {cat: [] for cat in t2.TIER2_CATEGORIES}


def test_load_tasks_partial_corpus_yields_partial_groups(tmp_path: Path):
    cat_dir = tmp_path / "peft-loops"
    cat_dir.mkdir()
    (cat_dir / "task_001.json").write_text(json.dumps({
        "task_id": "peft_001", "category": "peft-loops",
        "prompt": "Train a LoRA on alpaca", "rubric": {"type": "programmatic"},
        "reference_solution": "from peft import LoraConfig",
    }))
    grouped = t2.load_tasks(tmp_path)
    assert len(grouped["peft-loops"]) == 1
    assert grouped["peft-loops"][0]["task_id"] == "peft_001"
    assert grouped["curriculum-design"] == []
    assert grouped["custom-layers"]     == []
    assert grouped["diagnostics"]       == []


def test_load_tasks_rejects_missing_required_keys(tmp_path: Path):
    cat_dir = tmp_path / "custom-layers"
    cat_dir.mkdir()
    (cat_dir / "bad.json").write_text(json.dumps({"task_id": "x"}))
    with pytest.raises(ValueError, match="missing required keys"):
        t2.load_tasks(tmp_path)


def test_load_tasks_rejects_category_mismatch(tmp_path: Path):
    cat_dir = tmp_path / "diagnostics"
    cat_dir.mkdir()
    (cat_dir / "wrong_home.json").write_text(json.dumps({
        "task_id": "x", "category": "peft-loops",
        "prompt": "", "rubric": {}, "reference_solution": "",
    }))
    with pytest.raises(ValueError, match="declares category"):
        t2.load_tasks(tmp_path)


# ---- run_tier2 contract ----------------------------------------------------

def test_run_tier2_with_empty_corpus_reports_per_category_errors(tmp_path: Path):
    """No tasks on disk -> partial result with one error per category."""
    result = t2.run_tier2(
        model_id="m", tasks_dir=tmp_path,
        query_model=lambda *_a: "should not be called",
    )
    assert result["tier"] == 2
    assert result["v"] == 1
    assert result["model_id"] == "m"
    assert result["score"] == 0.0
    assert result["components"]["raw_points"] == 0
    assert result["components"]["max_raw_points"] == 200
    # An entry per category in errors.
    errors = result["components"]["errors"]
    for cat in t2.TIER2_CATEGORIES:
        assert cat in errors, f"no error entry for {cat!r}"
        assert "no tasks loaded" in errors[cat]


def test_run_tier2_with_full_corpus_and_perfect_runner_scores_one_hundred(
    tmp_path: Path,
):
    """Inject the full v1 task count + a perfect-score runner.

    With 50 tasks scoring max points each, raw points = 200, final
    score = 100.0. This locks the score normalization end-to-end.
    """
    for cat in t2.TIER2_CATEGORIES:
        cat_dir = tmp_path / cat
        cat_dir.mkdir()
        for i in range(t2.TIER2_TASK_COUNTS[cat]):
            (cat_dir / f"task_{i:03d}.json").write_text(json.dumps({
                "task_id":           f"{cat}_{i:03d}",
                "category":          cat,
                "prompt":            f"prompt {i}",
                "rubric":            {"type": "stub"},
                "reference_solution": "",
            }))

    def perfect(task, output):
        max_pts = t2.TIER2_MAX_POINTS_PER_TASK[task["category"]]
        return t2.RubricResult(score=max_pts, max_score=max_pts, rationale="ok")

    runners = {cat: perfect for cat in t2.TIER2_CATEGORIES}

    result = t2.run_tier2(
        model_id="m", tasks_dir=tmp_path,
        rubric_runners=runners,
        query_model=lambda prompt, model_id, base_url: "model output",
    )

    assert result["score"] == 100.0
    assert result["components"]["raw_points"]     == 200
    assert result["components"]["max_raw_points"] == 200
    assert "errors" not in result["components"]
    for cat in t2.TIER2_CATEGORIES:
        cat_block = result["components"]["by_category"][cat]
        assert cat_block["score"]     == t2.TIER2_RAW_POINTS_PER_CATEGORY[cat]
        assert cat_block["max_score"] == t2.TIER2_RAW_POINTS_PER_CATEGORY[cat]
        assert cat_block["tasks_run"] == t2.TIER2_TASK_COUNTS[cat]


def test_run_tier2_query_model_receives_prompt_model_id_and_base_url(tmp_path: Path):
    """The injected query function must see the same (prompt, model_id, base_url)."""
    cat_dir = tmp_path / "custom-layers"
    cat_dir.mkdir()
    (cat_dir / "t.json").write_text(json.dumps({
        "task_id": "cl_001", "category": "custom-layers",
        "prompt": "Implement RMSNorm without nn.Module",
        "rubric": {"type": "unit-tests"}, "reference_solution": "",
    }))

    seen: list[tuple[str, str, str]] = []

    def query(prompt, model_id, base_url):
        seen.append((prompt, model_id, base_url))
        return "stub-output"

    def runner(task, output):
        return t2.RubricResult(score=1, max_score=1, rationale="ok")

    t2.run_tier2(
        model_id="lamarck-g7",
        base_url="http://10.0.0.5:8000/v1",
        tasks_dir=tmp_path,
        rubric_runners={"custom-layers": runner},
        query_model=query,
    )

    assert seen == [(
        "Implement RMSNorm without nn.Module",
        "lamarck-g7",
        "http://10.0.0.5:8000/v1",
    )]


def test_run_tier2_runner_exception_lands_in_task_errors(tmp_path: Path):
    """A runner that raises shouldn't poison the whole tier."""
    cat_dir = tmp_path / "custom-layers"
    cat_dir.mkdir()
    (cat_dir / "t.json").write_text(json.dumps({
        "task_id": "cl_001", "category": "custom-layers",
        "prompt": "", "rubric": {}, "reference_solution": "",
    }))

    def runner(task, output):
        raise RuntimeError("sandbox blew up")

    result = t2.run_tier2(
        model_id="m", tasks_dir=tmp_path,
        rubric_runners={"custom-layers": runner},
        query_model=lambda *_a: "out",
    )

    cl = result["components"]["by_category"]["custom-layers"]
    assert cl["score"] == 0
    assert cl["tasks_run"] == 0
    assert "task_errors" in result["components"]["errors"]
    err = result["components"]["errors"]["task_errors"]["custom-layers"]
    assert any("RuntimeError" in e and "sandbox blew up" in e for e in err)


def test_run_tier2_partial_corpus_flags_task_count_mismatch(tmp_path: Path):
    """Only one peft task on disk (vs. v1's 15) - still scores but flags."""
    cat_dir = tmp_path / "peft-loops"
    cat_dir.mkdir()
    (cat_dir / "t.json").write_text(json.dumps({
        "task_id": "p", "category": "peft-loops",
        "prompt": "", "rubric": {}, "reference_solution": "",
    }))

    def runner(task, output):
        return t2.RubricResult(score=6, max_score=6, rationale="ok")

    result = t2.run_tier2(
        model_id="m", tasks_dir=tmp_path,
        rubric_runners={"peft-loops": runner},
        query_model=lambda *_a: "out",
    )

    assert "task_count_mismatch" in result["components"]["errors"]
    mismatch = result["components"]["errors"]["task_count_mismatch"]
    assert mismatch["peft-loops"] == {"expected": 15, "found": 1}
    # Score is 6 raw / 200 max = 3.0.
    assert result["score"] == 3.0
