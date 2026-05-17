# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
"""Locked-invariant tests for the eval-suite v1 spec.

These tests don't validate model behavior — they validate that the
v1 design hasn't been quietly edited. The eval suite is supposed
to be locked at the start of the first generation; a change to
weights or to the suite version is supposed to be a v2 event.

These tests fail if someone tries to silently bump v1 numbers.
"""

from __future__ import annotations

from lamarck import eval as lamarck_eval


def test_suite_version_is_one():
    assert lamarck_eval.SUITE_VERSION == 1


def test_tier_weights_sum_to_one():
    assert sum(lamarck_eval.TIER_WEIGHTS.values()) == 1.0


def test_tier_weights_are_locked_v1_values():
    """v1 weights: 20/50/30. Any change is a v2 event + DESIGN.md update."""
    assert lamarck_eval.TIER_WEIGHTS == {1: 0.20, 2: 0.50, 3: 0.30}


def test_aggregate_combines_three_tiers_with_locked_weights():
    t1 = lamarck_eval.TierResult(
        tier=1, score=80.0, components={}, ran_at="2026-01-01T00:00:00Z",
        model_id="g1", v=1,
    )
    t2 = lamarck_eval.TierResult(
        tier=2, score=60.0, components={}, ran_at="2026-01-01T00:00:00Z",
        model_id="g1", v=1,
    )
    t3 = lamarck_eval.TierResult(
        tier=3, score=40.0, components={}, ran_at="2026-01-01T00:00:00Z",
        model_id="g1", v=1,
    )
    result = lamarck_eval.aggregate(t1, t2, t3)
    # 0.20 * 80 + 0.50 * 60 + 0.30 * 40 = 16 + 30 + 12 = 58
    assert result["final_score"] == 58.0
    assert result["partial"] is False


def test_aggregate_marks_partial_when_a_tier_is_missing():
    t1 = lamarck_eval.TierResult(
        tier=1, score=80.0, components={}, ran_at="2026-01-01T00:00:00Z",
        model_id="g1", v=1,
    )
    result = lamarck_eval.aggregate(t1, None, None)
    assert result["partial"] is True
    assert result["final_score"] == 0.20 * 80.0  # only tier1 contributes


def test_aggregate_zero_when_all_tiers_missing():
    result = lamarck_eval.aggregate(None, None, None)
    assert result["final_score"] == 0.0
    assert result["partial"] is True
