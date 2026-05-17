# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
"""Unit tests for the custom-layers unit-test rubric runner (L8).

Mock seam: ``_run_pytest_against_candidate`` and ``_torch_available``
are both isolated functions so we can monkeypatch them per test. The
pre-pytest syntax check is real (cheap, deterministic).
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from lamarck.eval.rubrics import custom_layer
from lamarck.eval.tier2_engineering import default_tasks_dir, load_tasks


def _task(unit_tests: str = "def test_x():\n    assert True\n") -> dict:
    """A minimal custom-layers Task dict carrying a unit_tests payload."""
    return {
        "task_id": "test_task",
        "category": "custom-layers",
        "prompt": "test prompt",
        "rubric": {"type": "unit-tests-binary", "max_score": 1,
                   "criteria": [{"id": "tests_pass", "points": 1}]},
        "reference_solution": "",
        "unit_tests": unit_tests,
    }


# ---- Rung 0: syntax check --------------------------------------------------

def test_syntax_error_scores_zero_immediately():
    """A non-parsing candidate doesn't even hit the subprocess."""
    result = custom_layer.score_custom_layer(_task(), "def f(:\n    pass\n")
    assert result["score"] == 0
    assert result["max_score"] == 1
    assert "SyntaxError" in result["rationale"]


def test_syntax_error_in_sandbox_mode_also_short_circuits():
    """Even in sandbox mode, a parse error returns before any subprocess."""
    with patch.object(custom_layer, "_run_pytest_against_candidate") as run, \
         patch.object(custom_layer, "_torch_available") as probe:
        result = custom_layer.score_custom_layer(
            _task(), "def f(:\n    pass\n", sandbox=True,
        )
    assert result["score"] == 0
    run.assert_not_called()
    probe.assert_not_called()


# ---- Static (CI default) mode ----------------------------------------------

def test_static_mode_skips_pytest_even_for_valid_code():
    """sandbox=False never runs the subprocess - it's static-analysis-only."""
    with patch.object(custom_layer, "_run_pytest_against_candidate") as run, \
         patch.object(custom_layer, "_torch_available") as probe:
        result = custom_layer.score_custom_layer(
            _task(), "x = 1\n", sandbox=False,
        )
    assert result["score"] == 0
    assert "static-analysis-mode" in result["rationale"]
    run.assert_not_called()
    probe.assert_not_called()


# ---- Sandbox: torch pre-probe ---------------------------------------------

def test_sandbox_skipped_when_torch_missing():
    fake_probe = {"status": "missing-torch", "error": "ImportError: No module named 'torch'"}
    with patch.object(custom_layer, "_torch_available", return_value=fake_probe), \
         patch.object(custom_layer, "_run_pytest_against_candidate") as run:
        result = custom_layer.score_custom_layer(
            _task(), "x = 1\n", sandbox=True,
        )
    assert result["score"] == 0
    assert "requires-torch-runtime" in result["rationale"]
    run.assert_not_called()


def test_sandbox_probe_failure_surfaced_in_rationale():
    fake_probe = {"status": "probe-timeout", "timeout_seconds": 10}
    with patch.object(custom_layer, "_torch_available", return_value=fake_probe):
        result = custom_layer.score_custom_layer(
            _task(), "x = 1\n", sandbox=True,
        )
    assert result["score"] == 0
    assert "probe-timeout" in result["rationale"]


# ---- Sandbox: pytest pass/fail --------------------------------------------

_OK_PROBE = {"status": "ok"}


def test_sandbox_pytest_zero_exit_scores_one():
    fake_verdict = {"status": "ran", "returncode": 0,
                    "stdout_tail": "1 passed", "stderr_tail": ""}
    with patch.object(custom_layer, "_torch_available", return_value=_OK_PROBE), \
         patch.object(custom_layer, "_run_pytest_against_candidate",
                      return_value=fake_verdict):
        result = custom_layer.score_custom_layer(
            _task(), "x = 1\n", sandbox=True,
        )
    assert result["score"] == 1
    assert "pytest passed" in result["rationale"]


def test_sandbox_pytest_nonzero_exit_scores_zero_with_tail():
    fake_verdict = {"status": "ran", "returncode": 1,
                    "stdout_tail": "test_x FAILED\nE   AssertionError",
                    "stderr_tail": ""}
    with patch.object(custom_layer, "_torch_available", return_value=_OK_PROBE), \
         patch.object(custom_layer, "_run_pytest_against_candidate",
                      return_value=fake_verdict):
        result = custom_layer.score_custom_layer(
            _task(), "x = 1\n", sandbox=True,
        )
    assert result["score"] == 0
    assert "exit 1" in result["rationale"]
    assert "AssertionError" in result["rationale"]


def test_sandbox_pytest_timeout_scores_zero():
    fake_verdict = {"status": "timeout", "timeout_seconds": 30}
    with patch.object(custom_layer, "_torch_available", return_value=_OK_PROBE), \
         patch.object(custom_layer, "_run_pytest_against_candidate",
                      return_value=fake_verdict):
        result = custom_layer.score_custom_layer(
            _task(), "x = 1\n", sandbox=True, timeout=30,
        )
    assert result["score"] == 0
    assert "pytest timeout after 30s" in result["rationale"]


def test_sandbox_subprocess_crash_scores_zero():
    fake_verdict = {"status": "crash", "error": "OSError: [Errno 2]"}
    with patch.object(custom_layer, "_torch_available", return_value=_OK_PROBE), \
         patch.object(custom_layer, "_run_pytest_against_candidate",
                      return_value=fake_verdict):
        result = custom_layer.score_custom_layer(
            _task(), "x = 1\n", sandbox=True,
        )
    assert result["score"] == 0
    assert "subprocess error: crash" in result["rationale"]


# ---- Corpus guard: missing/empty unit_tests -------------------------------

def test_sandbox_missing_unit_tests_field_is_corpus_error():
    """If the task somehow loaded without unit_tests, we surface it as
    a corpus-bug rather than silently scoring 0 with no explanation."""
    task = _task()
    del task["unit_tests"]
    with patch.object(custom_layer, "_torch_available", return_value=_OK_PROBE), \
         patch.object(custom_layer, "_run_pytest_against_candidate") as run:
        result = custom_layer.score_custom_layer(
            task, "x = 1\n", sandbox=True,
        )
    assert result["score"] == 0
    assert "task corpus error" in result["rationale"]
    run.assert_not_called()


def test_sandbox_empty_unit_tests_string_is_corpus_error():
    task = _task(unit_tests="   \n")
    with patch.object(custom_layer, "_torch_available", return_value=_OK_PROBE), \
         patch.object(custom_layer, "_run_pytest_against_candidate") as run:
        result = custom_layer.score_custom_layer(
            task, "x = 1\n", sandbox=True,
        )
    assert result["score"] == 0
    assert "task corpus error" in result["rationale"]
    run.assert_not_called()


# ---- _torch_available probe behaviour -------------------------------------

def test_torch_probe_parses_json_status():
    """Probe handler reads the subprocess's JSON status verbatim."""
    fake = MagicMock(stdout='{"status": "ok"}', stderr="", returncode=0)
    with patch.object(custom_layer.subprocess, "run", return_value=fake):
        assert custom_layer._torch_available() == {"status": "ok"}


def test_torch_probe_handles_timeout():
    from subprocess import TimeoutExpired

    def _raise(*_a, **_kw):
        raise TimeoutExpired(cmd="x", timeout=5)

    with patch.object(custom_layer.subprocess, "run", side_effect=_raise):
        verdict = custom_layer._torch_available(timeout=5)
    assert verdict["status"] == "probe-timeout"


def test_torch_probe_handles_malformed_output():
    fake = MagicMock(stdout="not json", stderr="", returncode=0)
    with patch.object(custom_layer.subprocess, "run", return_value=fake):
        verdict = custom_layer._torch_available()
    assert verdict["status"] == "malformed-probe-output"


# ---- _run_pytest_against_candidate end-to-end ----------------------------

def test_runner_passes_when_reference_meets_a_python_only_test(tmp_path):
    """End-to-end with REAL pytest but a torch-free reference + test pair.

    This exercises the on-disk staging + PYTHONPATH plumbing without
    requiring torch in the CI venv.
    """
    candidate = "def add(a, b):\n    return a + b\n"
    unit_tests = (
        "from solution import add\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n"
    )
    verdict = custom_layer._run_pytest_against_candidate(
        candidate, unit_tests, timeout=30,
    )
    assert verdict["status"] == "ran"
    assert verdict["returncode"] == 0


def test_runner_fails_when_test_assertion_fails(tmp_path):
    candidate = "def add(a, b):\n    return a - b\n"  # wrong
    unit_tests = (
        "from solution import add\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n"
    )
    verdict = custom_layer._run_pytest_against_candidate(
        candidate, unit_tests, timeout=30,
    )
    assert verdict["status"] == "ran"
    assert verdict["returncode"] != 0
    assert "FAILED" in verdict["stdout_tail"] or "assert" in verdict["stdout_tail"]


# ---- Reference-corpus integration -----------------------------------------

def test_every_reference_solution_parses_under_static_mode():
    """In static mode, every L7 reference parses cleanly - so it gets a
    0/1 with the static-analysis rationale (not a parse error)."""
    grouped = load_tasks(default_tasks_dir())
    for task in grouped["custom-layers"]:
        result = custom_layer.score_custom_layer(
            task, task["reference_solution"], sandbox=False,
        )
        assert "static-analysis-mode" in result["rationale"], (
            f"reference for {task['task_id']!r} did not reach static "
            f"skip - rationale: {result['rationale']}"
        )


def test_every_reference_solution_scores_one_in_mocked_sandbox():
    """With a mocked-ok torch probe + mocked-passing pytest, every L7
    reference scores 1/1. Real execution happens on the pod (torch
    isn't in this venv)."""
    grouped = load_tasks(default_tasks_dir())
    ok_pytest = {"status": "ran", "returncode": 0,
                 "stdout_tail": "10 passed", "stderr_tail": ""}
    with patch.object(custom_layer, "_torch_available",
                      return_value={"status": "ok"}), \
         patch.object(custom_layer, "_run_pytest_against_candidate",
                      return_value=ok_pytest):
        for task in grouped["custom-layers"]:
            result = custom_layer.score_custom_layer(
                task, task["reference_solution"], sandbox=True,
            )
            assert result["score"] == 1, (
                f"reference for {task['task_id']!r} scored "
                f"{result['score']}/1 in mocked sandbox\n"
                f"{result['rationale']}"
            )
