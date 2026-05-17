# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
"""Unit + smoke tests for the aggregation CLI (L17).

Tests cover:
  - load_tier_result happy + every validation-failure mode
  - CLI argument parsing (build_parser)
  - main() end-to-end with on-disk JSON inputs
  - Partial-result paths when tier files are omitted
  - --out - writes to stdout
  - Score-math round-trip matching aggregate() in __init__
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

from lamarck.eval import SUITE_VERSION
from lamarck.eval.aggregate_cli import (
    TIER_RESULT_REQUIRED_KEYS,
    TierResultError,
    build_parser,
    load_tier_result,
    main,
)


# ---- Fixture: build well-formed tier-result JSON files --------------------

def _tier_result(tier: int, score: float, model_id: str = "m") -> dict[str, Any]:
    return {
        "tier":      tier,
        "score":     score,
        "components": {},
        "ran_at":    "2026-05-17T00:00:00+00:00",
        "model_id":  model_id,
        "v":         SUITE_VERSION,
    }


def _write_json(tmp_path: Path, name: str, payload: dict) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(payload))
    return p


# ---- load_tier_result validation ------------------------------------------

def test_load_tier_result_happy_path(tmp_path: Path):
    p = _write_json(tmp_path, "t1.json", _tier_result(1, 80.0))
    result = load_tier_result(p, expected_tier=1)
    assert result["tier"] == 1
    assert result["score"] == 80.0


def test_load_tier_result_file_missing_raises(tmp_path: Path):
    with pytest.raises(TierResultError, match="not found"):
        load_tier_result(tmp_path / "no.json", expected_tier=1)


def test_load_tier_result_invalid_json_raises(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("not json at all")
    with pytest.raises(TierResultError, match="not valid JSON"):
        load_tier_result(p, expected_tier=1)


def test_load_tier_result_top_level_must_be_object(tmp_path: Path):
    p = tmp_path / "list.json"
    p.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(TierResultError, match="top-level must be a JSON object"):
        load_tier_result(p, expected_tier=1)


def test_load_tier_result_missing_required_keys_raises(tmp_path: Path):
    payload = _tier_result(1, 80.0)
    del payload["ran_at"]
    p = _write_json(tmp_path, "t1.json", payload)
    with pytest.raises(TierResultError, match="missing required keys"):
        load_tier_result(p, expected_tier=1)


def test_load_tier_result_tier_mismatch_raises(tmp_path: Path):
    p = _write_json(tmp_path, "t2.json", _tier_result(2, 50.0))
    with pytest.raises(TierResultError, match="mismatches expected"):
        load_tier_result(p, expected_tier=1)


def test_load_tier_result_score_out_of_range_raises(tmp_path: Path):
    p = _write_json(tmp_path, "t1.json", _tier_result(1, 250.0))
    with pytest.raises(TierResultError, match="not in"):
        load_tier_result(p, expected_tier=1)


def test_load_tier_result_score_non_numeric_raises(tmp_path: Path):
    payload = _tier_result(1, 50.0)
    payload["score"] = "perfect"
    p = _write_json(tmp_path, "t1.json", payload)
    with pytest.raises(TierResultError, match="must be a number"):
        load_tier_result(p, expected_tier=1)


def test_load_tier_result_wrong_suite_version_raises(tmp_path: Path):
    payload = _tier_result(1, 80.0)
    payload["v"] = 2
    p = _write_json(tmp_path, "t1.json", payload)
    with pytest.raises(TierResultError, match="not 1"):
        load_tier_result(p, expected_tier=1)


# ---- CLI argument parser --------------------------------------------------

def test_build_parser_requires_out():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--tier1", "x.json"])  # no --out


def test_build_parser_accepts_partial_tier_args(tmp_path: Path):
    parser = build_parser()
    ns = parser.parse_args(["--tier1", "x.json", "--out", "y.json"])
    assert ns.tier1 == "x.json"
    assert ns.tier2 is None
    assert ns.tier3 is None
    assert ns.out == "y.json"


# ---- main() end-to-end ----------------------------------------------------

def test_main_three_tiers_writes_expected_aggregate(tmp_path: Path):
    """0.20 * 80 + 0.50 * 60 + 0.30 * 40 = 16 + 30 + 12 = 58."""
    t1 = _write_json(tmp_path, "t1.json", _tier_result(1, 80.0))
    t2 = _write_json(tmp_path, "t2.json", _tier_result(2, 60.0))
    t3 = _write_json(tmp_path, "t3.json", _tier_result(3, 40.0))
    out = tmp_path / "final.json"

    rc = main([
        "--tier1", str(t1), "--tier2", str(t2), "--tier3", str(t3),
        "--out", str(out),
    ])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["final_score"] == 58.0
    assert payload["partial"] is False
    assert payload["v"] == SUITE_VERSION
    assert payload["tier1"]["score"] == 80.0
    assert payload["tier2"]["score"] == 60.0
    assert payload["tier3"]["score"] == 40.0


def test_main_missing_tier_marks_partial(tmp_path: Path):
    """Only Tier 1 supplied; final == 0.20 * 80 = 16, partial=True."""
    t1 = _write_json(tmp_path, "t1.json", _tier_result(1, 80.0))
    out = tmp_path / "final.json"
    rc = main(["--tier1", str(t1), "--out", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["final_score"] == 16.0
    assert payload["partial"] is True
    assert payload["tier2"] is None
    assert payload["tier3"] is None


def test_main_all_three_missing_returns_zero_partial(tmp_path: Path):
    out = tmp_path / "final.json"
    rc = main(["--out", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["final_score"] == 0.0
    assert payload["partial"] is True


def test_main_validation_error_writes_to_stderr_and_returns_two(
    tmp_path: Path, capsys: pytest.CaptureFixture,
):
    """Invalid tier file -> stderr message + exit code 2."""
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    out = tmp_path / "final.json"
    rc = main(["--tier1", str(bad), "--out", str(out)])
    assert rc == 2
    captured = capsys.readouterr()
    assert "aggregate:" in captured.err
    assert "not valid JSON" in captured.err
    # No output file written.
    assert not out.exists()


def test_main_writes_to_stdout_when_out_is_dash(
    tmp_path: Path, capsys: pytest.CaptureFixture,
):
    t1 = _write_json(tmp_path, "t1.json", _tier_result(1, 50.0))
    rc = main(["--tier1", str(t1), "--out", "-"])
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["final_score"] == 10.0  # 0.20 * 50
    assert payload["partial"] is True


def test_main_creates_parent_dirs_for_output(tmp_path: Path):
    """--out path with non-existent parent dir gets the dir created."""
    t1 = _write_json(tmp_path, "t1.json", _tier_result(1, 80.0))
    out = tmp_path / "nested" / "results" / "final.json"
    rc = main(["--tier1", str(t1), "--out", str(out)])
    assert rc == 0
    assert out.exists()


# ---- Math round-trip lock --------------------------------------------------

def test_aggregate_cli_round_trip_matches_locked_weights(tmp_path: Path):
    """Sweep score combos and verify the CLI math matches the locked v1
    aggregator (0.20/0.50/0.30)."""
    cases = [
        # (t1, t2, t3, expected_final)
        (100.0, 100.0, 100.0, 100.0),  # ceiling
        (  0.0,   0.0,   0.0,   0.0),  # floor
        ( 50.0,  50.0,  50.0,  50.0),  # all-50
        (100.0,   0.0,   0.0,  20.0),  # only T1 weights kick
        (  0.0, 100.0,   0.0,  50.0),  # only T2
        (  0.0,   0.0, 100.0,  30.0),  # only T3
    ]
    for t1_score, t2_score, t3_score, expected in cases:
        t1 = _write_json(tmp_path, "t1.json", _tier_result(1, t1_score))
        t2 = _write_json(tmp_path, "t2.json", _tier_result(2, t2_score))
        t3 = _write_json(tmp_path, "t3.json", _tier_result(3, t3_score))
        out = tmp_path / "f.json"
        main(["--tier1", str(t1), "--tier2", str(t2),
              "--tier3", str(t3), "--out", str(out)])
        payload = json.loads(out.read_text())
        assert abs(payload["final_score"] - expected) < 1e-9, (
            f"({t1_score}, {t2_score}, {t3_score}) -> "
            f"{payload['final_score']} != {expected}"
        )
