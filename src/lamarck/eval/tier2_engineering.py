# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
"""Tier 2 - ML-engineering tasks.

The framework for the ~50-task hand-curated suite that does the
heavy lifting for Lamarck's reward signal. Per the locked v1 spec
(``docs/eval-suite-v1.md``), Tier 2 has four categories:

  * ``peft-loops``        - 15 tasks x 6 pts = 90 raw points
                            (programmatic 6-point ladder rubric)
  * ``curriculum-design`` - 10 tasks x 4 pts = 40 raw points
                            (LLM-judge 4-axis rubric)
  * ``custom-layers``     - 10 tasks x 1 pt  = 10 raw points
                            (programmatic binary unit-test rubric)
  * ``diagnostics``       - 15 tasks x 4 pts = 60 raw points
                            (LLM-judge 4-axis rubric)

Total: 200 raw points, normalized to 0-100 for aggregation.

This module is the *framework* - the task loader, the category
dispatcher, and the runner protocols. The actual rubric runners
live under ``lamarck.eval.rubrics.*`` and are wired in via the
``rubric_runners`` argument. That separation lets L6/L8/L11 ship
the runner implementations without touching this dispatcher.

Mock seams for tests:
  * ``rubric_runners`` - a ``{category: RubricRunner}`` dict.
  * ``query_model``    - a callable ``(prompt, model_id, base_url)
                         -> str`` used to fetch the model's
                         response to each task. Default uses
                         ``urllib.request`` against an
                         OpenAI-compatible ``/chat/completions``
                         endpoint, so no third-party dep needed.
  * ``tasks_dir``      - override the on-disk task corpus location.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, TypedDict

from . import SUITE_VERSION, TierResult


# ---- Locked v1 spec constants ----------------------------------------------
# These four numbers are the load-bearing v1 contract. Changing any of
# them is a v2 event and breaks generation-to-generation comparability.

TIER2_CATEGORIES: tuple[str, ...] = (
    "peft-loops",
    "curriculum-design",
    "custom-layers",
    "diagnostics",
)

TIER2_TASK_COUNTS: dict[str, int] = {
    "peft-loops":        15,
    "curriculum-design": 10,
    "custom-layers":     10,
    "diagnostics":       15,
}

TIER2_MAX_POINTS_PER_TASK: dict[str, int] = {
    "peft-loops":        6,
    "curriculum-design": 4,
    "custom-layers":     1,
    "diagnostics":       4,
}

# Per-category raw point ceilings - the product of task count x max
# points per task. Hard-coded here (not derived) so a silent edit to
# the lookup tables above fails the locked-invariant smoke tests.
TIER2_RAW_POINTS_PER_CATEGORY: dict[str, int] = {
    "peft-loops":        90,   # 15 x 6
    "curriculum-design": 40,   # 10 x 4
    "custom-layers":     10,   # 10 x 1
    "diagnostics":       60,   # 15 x 4
}

TIER2_TOTAL_RAW_POINTS: int = 200  # sum of the four above

# Categories whose rubric is graded programmatically (deterministic
# code analysis / sandboxed execution) vs. LLM-judge graded.
# L6 ships the peft-loops runner, L8 ships custom-layers, L11 ships
# the LLM-judge runner used by the other two.
PROGRAMMATIC_CATEGORIES: tuple[str, ...] = ("peft-loops", "custom-layers")
LLM_JUDGE_CATEGORIES:   tuple[str, ...] = ("curriculum-design", "diagnostics")


# ---- Type contracts --------------------------------------------------------

class Task(TypedDict):
    """A single Tier 2 task as it lives on disk under tier2_tasks/."""

    task_id: str
    category: str
    prompt: str
    rubric: dict[str, Any]
    reference_solution: str


class RubricResult(TypedDict):
    """A rubric runner's verdict on one (task, model_output) pair."""

    score:     int        # >= 0
    max_score: int        # category's per-task ceiling
    rationale: str        # short human-readable explanation


class RubricRunner(Protocol):
    """Callable signature every rubric runner implements."""

    def __call__(self, task: Task, model_output: str) -> RubricResult: ...


QueryModel = Callable[[str, str, str], str]
"""(prompt, model_id, base_url) -> model_output."""


# ---- I/O helpers -----------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_tasks_dir() -> Path:
    """The canonical on-disk location of the Tier 2 task corpus.

    Tasks live next to this module so they're installed alongside
    the code (no separate data package). L5/L7/L9/L10 populate the
    subdirectories; until those land this directory may be empty
    or partial, and ``load_tasks`` reports the missing categories
    via the error channel instead of raising.
    """
    return Path(__file__).parent / "tier2_tasks"


def _read_task_file(path: Path) -> Task:
    """Parse one task JSON file with the keys the v1 spec mandates."""
    payload = json.loads(path.read_text())
    required = ("task_id", "category", "prompt", "rubric", "reference_solution")
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"task file {path} missing required keys: {missing}")
    return Task(
        task_id=payload["task_id"],
        category=payload["category"],
        prompt=payload["prompt"],
        rubric=payload["rubric"],
        reference_solution=payload["reference_solution"],
    )


def load_tasks(tasks_dir: Path | None = None) -> dict[str, list[Task]]:
    """Load all Tier 2 tasks, grouped by category.

    A category whose directory is missing or empty yields an empty
    list - the runner reports that as an error in ``components``
    rather than crashing, so a partial corpus still produces a
    diagnostic result.
    """
    root = tasks_dir if tasks_dir is not None else default_tasks_dir()
    grouped: dict[str, list[Task]] = {cat: [] for cat in TIER2_CATEGORIES}
    if not root.exists():
        return grouped
    for category in TIER2_CATEGORIES:
        cat_dir = root / category
        if not cat_dir.is_dir():
            continue
        for task_path in sorted(cat_dir.glob("*.json")):
            task = _read_task_file(task_path)
            # Enforce the on-disk category matches the directory it
            # lives in - silently mis-categorized tasks break scoring.
            if task["category"] != category:
                raise ValueError(
                    f"task {task['task_id']} declares category "
                    f"{task['category']!r} but lives under {category!r}"
                )
            grouped[category].append(task)
    return grouped


# ---- Model querying --------------------------------------------------------

def _default_query_model(prompt: str, model_id: str, base_url: str) -> str:
    """POST to an OpenAI-compatible ``/chat/completions`` endpoint.

    Uses only the stdlib so this module imports cleanly on a fresh
    pod. Tests inject a fake instead of monkeypatching this.
    """
    import urllib.request  # noqa: WPS433 - stdlib lazy import is cheap

    body = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        # Deterministic-ish settings keep scoring noise down; v1 does
        # not pin these, so they're free to tune without invalidating
        # the suite.
        "temperature": 0.0,
        "max_tokens": 2048,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:  # noqa: S310 - local pod
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


# ---- Default rubric dispatchers --------------------------------------------

def _no_runner_registered(category: str) -> RubricRunner:
    """Build a runner that reports the category has no implementation.

    Until L6/L8/L11 land their runners, ``run_tier2`` returns a partial
    result where every task in an unimplemented category contributes 0
    and an explanatory error - rather than crashing or silently scoring 0.
    """
    max_pts = TIER2_MAX_POINTS_PER_TASK[category]

    def _runner(task: Task, model_output: str) -> RubricResult:
        return RubricResult(
            score=0,
            max_score=max_pts,
            rationale=f"no rubric runner registered for category {category!r}",
        )

    return _runner


def resolve_runner(
    category: str,
    rubric_runners: dict[str, RubricRunner] | None,
) -> RubricRunner:
    """Pick the runner for ``category`` from the injected registry.

    Falls back to a "no runner registered" stub that scores zero and
    reports the gap. Categories outside the v1 list raise immediately
    - silent acceptance there would mask a misspelled task category.
    """
    if category not in TIER2_CATEGORIES:
        raise KeyError(f"unknown Tier 2 category: {category!r}")
    if rubric_runners and category in rubric_runners:
        return rubric_runners[category]
    return _no_runner_registered(category)


# ---- Public entry point ----------------------------------------------------

def run_tier2(
    model_id: str,
    base_url: str = "http://localhost:8000/v1",
    *,
    tasks_dir: Path | None = None,
    rubric_runners: dict[str, RubricRunner] | None = None,
    query_model: QueryModel | None = None,
) -> TierResult:
    """Run Tier 2's four task categories against a served model.

    Per the locked v1 spec, the four categories together yield
    ``TIER2_TOTAL_RAW_POINTS`` (= 200) raw points, normalized to a
    0-100 score that the aggregator multiplies by ``TIER_WEIGHTS[2]``
    (= 0.50).

    Returns a partial ``TierResult`` if any category cannot run
    (missing tasks, unregistered runner, runner exception, model
    query failure). Partial results carry per-category breakdowns
    and an ``errors`` dict in ``components`` so the caller can see
    exactly which slice was lost.
    """
    query = query_model if query_model is not None else _default_query_model
    tasks_by_category = load_tasks(tasks_dir)

    per_category: dict[str, dict[str, Any]] = {}
    errors: dict[str, Any] = {}
    total_score = 0

    for category in TIER2_CATEGORIES:
        tasks = tasks_by_category.get(category, [])
        expected = TIER2_TASK_COUNTS[category]
        max_pts = TIER2_RAW_POINTS_PER_CATEGORY[category]
        runner = resolve_runner(category, rubric_runners)

        if not tasks:
            errors[category] = (
                f"no tasks loaded for {category!r} "
                f"(expected {expected}, found 0)"
            )
            per_category[category] = {
                "score": 0, "max_score": max_pts,
                "tasks_run": 0, "task_results": [],
            }
            continue

        if len(tasks) != expected:
            # Not fatal - v1 lets a partial corpus produce a diagnostic
            # result - but the discrepancy belongs in the errors channel.
            errors.setdefault("task_count_mismatch", {})[category] = {
                "expected": expected, "found": len(tasks),
            }

        cat_score = 0
        task_results: list[dict[str, Any]] = []
        cat_errors: list[str] = []
        for task in tasks:
            try:
                model_output = query(task["prompt"], model_id, base_url)
                result = runner(task, model_output)
            except Exception as exc:  # noqa: BLE001 - surface all modes
                cat_errors.append(
                    f"{task['task_id']}: {type(exc).__name__}: {exc}"
                )
                continue
            cat_score += int(result["score"])
            task_results.append({
                "task_id":   task["task_id"],
                "score":     int(result["score"]),
                "max_score": int(result["max_score"]),
                "rationale": result.get("rationale", ""),
            })

        per_category[category] = {
            "score":        cat_score,
            "max_score":    max_pts,
            "tasks_run":    len(task_results),
            "task_results": task_results,
        }
        if cat_errors:
            errors.setdefault("task_errors", {})[category] = cat_errors
        total_score += cat_score

    score_0_100 = (total_score / TIER2_TOTAL_RAW_POINTS) * 100.0
    components: dict[str, Any] = {
        "raw_points":       total_score,
        "max_raw_points":   TIER2_TOTAL_RAW_POINTS,
        "by_category":      per_category,
    }
    if errors:
        components["errors"] = errors

    return TierResult(
        tier=2,
        score=round(score_0_100, 2),
        components=components,
        ran_at=_utc_now(),
        model_id=model_id,
        v=SUITE_VERSION,
    )
