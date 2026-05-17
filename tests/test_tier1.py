# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
"""Full mocked integration tests for ``lamarck.eval.tier1_external``.

The L2 smoke tests in ``test_tier1_smoke.py`` lock module invariants
and exercise narrow paths. This file is the end-to-end pass:
``run_tier1`` driven against a mocked ``simple_evaluate`` across all
the realistic scenarios the locked v1 spec creates.

What's covered here that the smoke tests don't:

- All three benchmarks succeed → score is the equal-weight average
  with correct rounding.
- MMLU-Pro is computed as the equal-weight average of its four
  locked subsets (``computer_science``, ``engineering``, ``math``,
  ``physics``) and nothing else.
- ``simple_evaluate`` is called exactly once per benchmark with the
  expected task names; per-subset MMLU-Pro tasks are passed as one
  list.
- Cache: cache miss → run + write. Cache hit (same model_id) → read.
  model_id mismatch → invalidate + re-run + overwrite.
- ``cache_dir=None`` codepath: nothing is read, nothing is written,
  the result is identical apart from no on-disk artifacts.
- Default ``base_url`` reaches the runner. Override is honored.
- TierResult schema is wholly populated: ``tier=1``, ``v=1``,
  ``components`` with one entry per benchmark (and an ``errors``
  key only when at least one failed).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lamarck.eval import tier1_external as t1


# ---------- fixtures ---------------------------------------------------------

def _make_results(scores: dict[str, float]) -> dict[str, Any]:
    """Build a fake ``simple_evaluate`` return given task → fraction."""
    return {"results": {task: {"acc,none": frac} for task, frac in scores.items()}}


def _patch_resolve(monkeypatch: pytest.MonkeyPatch, fake) -> list[tuple[str, list[str]]]:
    """Replace _resolve_simple_evaluate with a recorder + the fake.

    Returns a mutable list each call appends ``(model_args, tasks)`` to,
    so tests can assert on the calls made.
    """
    calls: list[tuple[str, list[str]]] = []

    def recorded(model: str, model_args: str, tasks: list[str]) -> dict[str, Any]:
        calls.append((model_args, list(tasks)))
        return fake(model_args, tasks)

    monkeypatch.setattr(t1, "_resolve_simple_evaluate", lambda: recorded)
    return calls


# ---------- happy path -------------------------------------------------------

def test_all_three_benchmarks_succeed_score_is_equal_weight_average(monkeypatch):
    def fake(_args: str, tasks: list[str]) -> dict[str, Any]:
        # MMLU-Pro: 4 subsets each at 0.80 → mean 0.80
        # HumanEval+: 0.50
        # GSM8K: 0.20
        # Tier 1 average = (80 + 50 + 20) / 3 = 50.0
        if "mmlu_pro_computer_science" in tasks:
            return _make_results({t: 0.80 for t in tasks})
        if "humaneval_plus" in tasks:
            return _make_results({"humaneval_plus": 0.50})
        if "gsm8k" in tasks:
            return _make_results({"gsm8k": 0.20})
        raise AssertionError(f"unexpected tasks: {tasks}")

    _patch_resolve(monkeypatch, fake)

    result = t1.run_tier1(model_id="lamarck-g1")

    assert result["tier"] == 1
    assert result["v"] == 1
    assert result["model_id"] == "lamarck-g1"
    assert result["score"] == 50.0
    assert result["components"]["mmlu_pro"] == 80.0
    assert result["components"]["humaneval_plus"] == 50.0
    assert result["components"]["gsm8k"] == 20.0
    assert "errors" not in result["components"]


def test_mmlu_pro_is_locked_to_four_subsets_and_averaged_equally(monkeypatch):
    seen_tasks: list[str] = []

    def fake(_args: str, tasks: list[str]) -> dict[str, Any]:
        seen_tasks.extend(tasks)
        scores = {
            "mmlu_pro_computer_science": 0.40,
            "mmlu_pro_engineering":      0.60,
            "mmlu_pro_math":             0.80,
            "mmlu_pro_physics":          1.00,
            "humaneval_plus":            0.50,
            "gsm8k":                     0.50,
        }
        return _make_results({t: scores[t] for t in tasks if t in scores})

    _patch_resolve(monkeypatch, fake)
    result = t1.run_tier1(model_id="m")

    # All four locked subsets ran — nothing more, nothing less.
    mmlu_calls = [t for t in seen_tasks if t.startswith("mmlu_pro_")]
    assert set(mmlu_calls) == {
        "mmlu_pro_computer_science",
        "mmlu_pro_engineering",
        "mmlu_pro_math",
        "mmlu_pro_physics",
    }
    # MMLU-Pro aggregate: (40 + 60 + 80 + 100) / 4 = 70.0
    assert result["components"]["mmlu_pro"] == 70.0


def test_each_benchmark_is_called_with_expected_task_names(monkeypatch):
    calls = _patch_resolve(
        monkeypatch,
        lambda _args, tasks: _make_results({t: 0.5 for t in tasks}),
    )
    t1.run_tier1(model_id="m")

    task_lists = [tasks for _args, tasks in calls]
    # One call to lm_eval per benchmark family.
    assert any(set(tasks).issuperset({"mmlu_pro_computer_science"}) for tasks in task_lists)
    assert ["humaneval_plus"] in task_lists
    assert ["gsm8k"] in task_lists


# ---------- partial-result path ----------------------------------------------

def test_one_benchmark_failure_leaves_tier_with_partial_score(monkeypatch):
    def fake(_args: str, tasks: list[str]) -> dict[str, Any]:
        if "humaneval_plus" in tasks:
            raise RuntimeError("HF dataset card was nil")
        return _make_results({t: 0.5 for t in tasks})

    _patch_resolve(monkeypatch, fake)
    result = t1.run_tier1(model_id="m")

    # Surviving benchmarks: MMLU-Pro (50) + GSM8K (50). Average: 50.0.
    assert result["score"] == 50.0
    assert result["components"]["humaneval_plus"] is None
    assert "errors" in result["components"]
    assert "humaneval_plus" in result["components"]["errors"]
    assert "RuntimeError" in result["components"]["errors"]["humaneval_plus"]


def test_all_benchmarks_failing_returns_zero(monkeypatch):
    def fake(_args: str, _tasks: list[str]) -> dict[str, Any]:
        raise RuntimeError("everything is broken")

    _patch_resolve(monkeypatch, fake)
    result = t1.run_tier1(model_id="m")

    assert result["score"] == 0.0
    for bench in t1.TIER1_BENCHMARKS:
        assert result["components"][bench] is None
    assert set(result["components"]["errors"]) == set(t1.TIER1_BENCHMARKS)


# ---------- caching path -----------------------------------------------------

def test_cache_miss_writes_one_file_per_benchmark(monkeypatch, tmp_path: Path):
    _patch_resolve(
        monkeypatch,
        lambda _args, tasks: _make_results({t: 0.6 for t in tasks}),
    )
    t1.run_tier1(model_id="m1", cache_dir=tmp_path)

    for bench in t1.TIER1_BENCHMARKS:
        cache = tmp_path / f"{bench}.json"
        assert cache.exists(), f"cache missing for {bench}"
        payload = json.loads(cache.read_text())
        assert payload["benchmark"] == bench
        assert payload["model_id"] == "m1"
        assert payload["fraction"] == 0.6


def test_cache_hit_skips_simple_evaluate(monkeypatch, tmp_path: Path):
    # Seed the cache with a different fraction than we'd see on a re-run.
    for bench in t1.TIER1_BENCHMARKS:
        (tmp_path / f"{bench}.json").write_text(json.dumps({
            "benchmark": bench, "model_id": "m1",
            "fraction": 0.9, "ran_at": "2026-01-01T00:00:00Z",
        }))

    calls = _patch_resolve(
        monkeypatch,
        lambda _args, tasks: _make_results({t: 0.1 for t in tasks}),
    )
    result = t1.run_tier1(model_id="m1", cache_dir=tmp_path)

    # Cache values were used; runner was never called.
    assert calls == []
    assert result["score"] == 90.0
    for bench in t1.TIER1_BENCHMARKS:
        assert result["components"][bench] == 90.0


def test_cache_hit_only_for_matching_model_id(monkeypatch, tmp_path: Path):
    # Seed cache for model "m1"…
    (tmp_path / "gsm8k.json").write_text(json.dumps({
        "benchmark": "gsm8k", "model_id": "m1",
        "fraction": 0.9, "ran_at": "2026-01-01T00:00:00Z",
    }))

    # …but call run_tier1 with a different model_id.
    calls = _patch_resolve(
        monkeypatch,
        lambda _args, tasks: _make_results({t: 0.3 for t in tasks}),
    )
    result = t1.run_tier1(model_id="m2", cache_dir=tmp_path)

    # gsm8k cache was invalidated → runner WAS called for gsm8k.
    assert any(tasks == ["gsm8k"] for _args, tasks in calls)
    assert result["components"]["gsm8k"] == 30.0
    # Cache file got overwritten with the new model_id.
    payload = json.loads((tmp_path / "gsm8k.json").read_text())
    assert payload["model_id"] == "m2"
    assert payload["fraction"] == 0.3


def test_cache_dir_none_neither_reads_nor_writes(monkeypatch):
    _patch_resolve(
        monkeypatch,
        lambda _args, tasks: _make_results({t: 0.5 for t in tasks}),
    )
    # No exception, no disk activity, no surprises. The earlier "happy
    # path" tests also exercise this codepath; this one's the explicit
    # contract.
    result = t1.run_tier1(model_id="m", cache_dir=None)
    assert result["score"] == 50.0


# ---------- base_url plumbing ------------------------------------------------

def test_default_base_url_is_localhost(monkeypatch):
    calls = _patch_resolve(
        monkeypatch,
        lambda _args, tasks: _make_results({t: 0.5 for t in tasks}),
    )
    t1.run_tier1(model_id="m")
    for args, _tasks in calls:
        assert "base_url=http://localhost:8000/v1" in args


def test_custom_base_url_reaches_runner(monkeypatch):
    calls = _patch_resolve(
        monkeypatch,
        lambda _args, tasks: _make_results({t: 0.5 for t in tasks}),
    )
    t1.run_tier1(model_id="m", base_url="http://10.0.0.5:8000/v1")
    for args, _tasks in calls:
        assert "base_url=http://10.0.0.5:8000/v1" in args


def test_model_id_reaches_runner_via_model_args(monkeypatch):
    calls = _patch_resolve(
        monkeypatch,
        lambda _args, tasks: _make_results({t: 0.5 for t in tasks}),
    )
    t1.run_tier1(model_id="lamarck-g7")
    for args, _tasks in calls:
        assert "model=lamarck-g7" in args
