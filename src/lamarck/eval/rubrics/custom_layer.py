# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
"""Tier 2 unit-test rubric runner for the custom-layers category.

Per the locked v1 spec, custom-layers tasks score on a single binary
criterion: does the candidate's code pass the task's hand-curated
unit-test suite? Each L7 task ships:

  * ``reference_solution`` - a working implementation
  * ``unit_tests``         - pytest source that does ``from solution
                             import ...; def test_*()`` and asserts
                             on the resulting object's behaviour

This runner reconstructs the L8 contract on disk: it writes the
candidate to ``solution.py`` and the tests to ``test_solution.py``
in a tmpdir, then runs ``pytest`` in a subprocess. Exit code 0
means all tests passed -> score 1; anything else -> score 0.

## Two execution modes (same shape as the L6 PEFT/LoRA runner)

``score_custom_layer(task, model_output, sandbox=False, timeout=120)``:

- **``sandbox=False`` (default, CI mode):** the rung emits
  ``"skipped: static-analysis-mode"`` and scores 0. Static-only
  guardrail (rung 0 below) still runs - if the candidate is a
  syntax error, we surface that fast without bothering with
  pytest.

- **``sandbox=True`` (G2 eval mode):** pre-probe checks that
  ``torch`` imports in the subprocess. If not, the rung degrades
  to ``"skipped: requires-torch-runtime"`` and scores 0. Otherwise
  pytest runs against the candidate; pass/fail maps to 1/0.

## Rung 0: pre-pytest syntax check

If the candidate doesn't parse with ``ast.parse`` we don't bother
writing it to disk - the rationale immediately reports the syntax
error and the rung scores 0. This is the same "no parse, no
further evaluation" guardrail L6 uses for rung 1.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

from ..tier2_engineering import RubricResult, Task


# The v1 custom-layers rubric has a single criterion id; lock it here
# so the corpus and runner can't drift.
RUNG_ID: str = "tests_pass"

# Pre-probe that confirms torch is importable in the sandbox subprocess
# environment. Kept tiny so failure costs nothing.
_TORCH_PROBE_SOURCE = textwrap.dedent("""\
    import json, sys
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        print(json.dumps({"status": "missing-torch",
                          "error": f"{type(exc).__name__}: {exc}"}))
        sys.exit(0)
    print(json.dumps({"status": "ok"}))
""")


def _check_parses(source: str) -> tuple[bool, str]:
    """Pre-pytest syntax check. No point shelling out for SyntaxError."""
    try:
        ast.parse(source)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc.msg} at line {exc.lineno}"
    return True, "parses cleanly"


def _torch_available(timeout: int = 10) -> dict[str, Any]:
    """Probe the subprocess environment for a working torch import."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", _TORCH_PROBE_SOURCE],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"status": "probe-timeout", "timeout_seconds": timeout}
    except OSError as exc:
        return {"status": "probe-crash", "error": f"OSError: {exc}"}
    try:
        return json.loads(result.stdout.strip())
    except (json.JSONDecodeError, ValueError):
        return {"status": "malformed-probe-output",
                "raw_stdout": result.stdout[:512]}


def _run_pytest_against_candidate(
    candidate_src: str,
    unit_tests_src: str,
    timeout: int,
) -> dict[str, Any]:
    """Stage candidate + tests on disk and shell out to pytest.

    Always passes ``-q`` for quiet output and ``-p no:cacheprovider``
    so we don't litter the tmpdir with .pytest_cache. ``-x`` stops at
    first failure - we only need pass/fail, not a per-test breakdown.
    """
    with tempfile.TemporaryDirectory(prefix="lamarck-cl-") as tmp:
        td = Path(tmp)
        (td / "solution.py").write_text(candidate_src)
        (td / "test_solution.py").write_text(unit_tests_src)
        # Ensure the staging dir is on sys.path so ``from solution
        # import ...`` resolves - pytest adds rootdir to sys.path by
        # default, but we belt-and-suspenders with PYTHONPATH too.
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{td}{os.pathsep}{existing}" if existing else str(td)
        )
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-x",
                 "-p", "no:cacheprovider", "test_solution.py"],
                cwd=str(td), env=env,
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "timeout_seconds": timeout}
        except OSError as exc:
            return {"status": "crash", "error": f"OSError: {exc}"}

        return {
            "status": "ran",
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-1024:],
            "stderr_tail": result.stderr[-512:],
        }


def score_custom_layer(
    task: Task,
    model_output: str,
    *,
    sandbox: bool = False,
    timeout: int = 120,
) -> RubricResult:
    """Score ``model_output`` against the v1 custom-layers binary rubric.

    Static syntax check first (always). Then either skip in static
    mode, gracefully degrade if torch is missing, or actually run
    pytest against (model_output, task['unit_tests']).
    """
    score = 0
    rationale_parts: list[str] = []

    parses_ok, parse_msg = _check_parses(model_output)
    rationale_parts.append(f"parses: {parse_msg}")
    if not parses_ok:
        return RubricResult(
            score=0, max_score=1,
            rationale=f"{RUNG_ID}=0: {parse_msg}",
        )

    if not sandbox:
        return RubricResult(
            score=0, max_score=1,
            rationale=f"{RUNG_ID}=0: skipped: static-analysis-mode "
                      f"(rung 0 parse {parse_msg})",
        )

    # Sandbox mode: confirm torch is available, then actually run pytest.
    probe = _torch_available()
    if probe.get("status") != "ok":
        if probe.get("status") == "missing-torch":
            return RubricResult(
                score=0, max_score=1,
                rationale=(f"{RUNG_ID}=0: skipped: requires-torch-runtime "
                           f"({probe.get('error', '')})"),
            )
        return RubricResult(
            score=0, max_score=1,
            rationale=(f"{RUNG_ID}=0: torch pre-probe failed: "
                       f"{probe.get('status')}"),
        )

    unit_tests = task.get("unit_tests")
    if not isinstance(unit_tests, str) or not unit_tests.strip():
        return RubricResult(
            score=0, max_score=1,
            rationale=(f"{RUNG_ID}=0: task corpus error - "
                       f"unit_tests missing or empty"),
        )

    verdict = _run_pytest_against_candidate(
        candidate_src=model_output,
        unit_tests_src=unit_tests,
        timeout=timeout,
    )
    status = verdict.get("status", "unknown")
    if status == "ran":
        rc = verdict.get("returncode", 1)
        if rc == 0:
            return RubricResult(
                score=1, max_score=1,
                rationale=f"{RUNG_ID}=1: pytest passed",
            )
        return RubricResult(
            score=0, max_score=1,
            rationale=(f"{RUNG_ID}=0: pytest failed "
                       f"(exit {rc})\n"
                       f"stdout tail: {verdict.get('stdout_tail', '')[-256:]}"),
        )
    if status == "timeout":
        return RubricResult(
            score=0, max_score=1,
            rationale=(f"{RUNG_ID}=0: pytest timeout after "
                       f"{verdict.get('timeout_seconds')}s"),
        )
    return RubricResult(
        score=0, max_score=1,
        rationale=f"{RUNG_ID}=0: subprocess error: {status}",
    )
