# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
"""Subprocess tests for ``scripts/eval/run-all.sh``.

These tests shell out to the actual bash script and assert on:
  - Exit codes for usage errors + tier failures
  - Output-file presence and shape after a successful --mock run
  - The end-to-end mock score is exactly 57.0 (the locked value
    L18 documents).
  - The orchestrator wires aggregate_cli correctly.

Why subprocess: the script's job is to glue Python CLIs together;
the only honest test of that glue is to actually run it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_ALL = REPO_ROOT / "scripts" / "eval" / "run-all.sh"


def _run(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Invoke run-all.sh with $PYTHON pointed at the test interpreter.

    The script honours $PYTHON; pointing it at sys.executable means
    the orchestrator finds the same Python the tests run under
    (which has lamarck.eval on its path).
    """
    env = os.environ.copy()
    env["PYTHON"] = sys.executable
    return subprocess.run(
        [str(RUN_ALL), *args],
        env=env,
        cwd=str(cwd) if cwd else str(REPO_ROOT),
        capture_output=True, text=True, timeout=60,
    )


# ---- Usage / argument errors ----------------------------------------------

def test_no_args_prints_usage_and_exits_two():
    r = _run([])
    assert r.returncode == 2
    assert "usage:" in r.stderr.lower()


def test_help_flag_prints_usage_and_exits_two():
    r = _run(["-h"])
    assert r.returncode == 2  # script exits 2 from the early-help branch
    assert "usage:" in r.stderr.lower()


def test_two_positional_args_is_an_error():
    r = _run(["only-model-id", "only-base-url"])
    assert r.returncode == 2
    assert "need <model_id> <base_url> <curriculum_jsonl>" in r.stderr


def test_unknown_flag_is_an_error(tmp_path: Path):
    r = _run([
        "lamarck-g1", "http://stub", "/dev/null", "--bogus-flag",
        "--out-dir", str(tmp_path),
    ])
    assert r.returncode == 2
    assert "unknown flag" in r.stderr


def test_out_dir_without_value_is_an_error():
    r = _run([
        "lamarck-g1", "http://stub", "/dev/null", "--mock", "--out-dir",
    ])
    assert r.returncode == 2
    assert "--out-dir requires a value" in r.stderr


# ---- Happy path: mock pipeline writes all four artifacts ------------------

def test_mock_pipeline_writes_all_four_artifacts(tmp_path: Path):
    r = _run([
        "lamarck-g1", "http://stub", "/dev/null", "--mock",
        "--out-dir", str(tmp_path),
    ])
    assert r.returncode == 0, f"stderr:\n{r.stderr}\nstdout:\n{r.stdout}"
    for name in ("tier1.json", "tier2.json", "tier3.json", "final.json"):
        assert (tmp_path / name).exists(), f"missing {name}"


def test_mock_pipeline_final_score_is_locked_at_57(tmp_path: Path):
    """0.20 * 75 + 0.50 * 60 + 0.30 * 40 = 15 + 30 + 12 = 57.0.
    L18 documents this; L19 locks it in CI."""
    r = _run([
        "lamarck-g1", "http://stub", "/dev/null", "--mock",
        "--out-dir", str(tmp_path),
    ])
    assert r.returncode == 0
    final = json.loads((tmp_path / "final.json").read_text())
    assert final["final_score"] == 57.0
    assert final["partial"] is False
    assert final["v"] == 1


def test_mock_pipeline_per_tier_scores_are_locked(tmp_path: Path):
    _run([
        "lamarck-g1", "http://stub", "/dev/null", "--mock",
        "--out-dir", str(tmp_path),
    ])
    t1 = json.loads((tmp_path / "tier1.json").read_text())
    t2 = json.loads((tmp_path / "tier2.json").read_text())
    t3 = json.loads((tmp_path / "tier3.json").read_text())
    assert t1["tier"] == 1 and t1["score"] == 75.0
    assert t2["tier"] == 2 and t2["score"] == 60.0
    assert t3["tier"] == 3 and t3["score"] == 40.0
    for t in (t1, t2, t3):
        assert t["model_id"] == "lamarck-g1"
        assert t["v"] == 1
        assert t["components"]["mock"] is True


def test_mock_pipeline_default_out_dir_is_eval_out(tmp_path: Path):
    """Without --out-dir the script writes to ./eval-out in cwd."""
    r = _run(
        ["lamarck-g1", "http://stub", "/dev/null", "--mock"],
        cwd=tmp_path,
    )
    assert r.returncode == 0
    expected = tmp_path / "eval-out" / "final.json"
    assert expected.exists()


def test_mock_pipeline_prints_final_score_to_stdout(tmp_path: Path):
    r = _run([
        "lamarck-g1", "http://stub", "/dev/null", "--mock",
        "--out-dir", str(tmp_path),
    ])
    assert "Final score: 57.0" in r.stdout


# ---- Failing tier surfacing -----------------------------------------------

def test_invalid_curriculum_path_for_real_run_fails_tier3(tmp_path: Path):
    """Without --mock and with no real-runner deps installed in CI,
    Tier 1 will fail first because lm_eval is missing. The orchestrator
    surfaces that as exit code 10 (Tier 1)."""
    r = _run([
        "lamarck-g1", "http://stub", str(tmp_path / "no.jsonl"),
        "--out-dir", str(tmp_path),
    ])
    # In a CI env with no deps the first failing tier wins. Either way,
    # exit code is one of the tier-failure codes (10/20/30), not 0.
    assert r.returncode in {10, 20, 30}
    assert "FAILED" in r.stderr
