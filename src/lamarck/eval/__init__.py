# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
"""Lamarck eval suite — v1 API surface.

See `docs/eval-suite-v1.md` for the design + locked spec.

This module defines the API contract the three tiers + aggregator
implement. Each tier returns a structured dict with at minimum:

    {
        "tier": 1 | 2 | 3,
        "score": float,           # 0.0 to 100.0
        "components": {...},      # per-benchmark / per-task breakdown
        "ran_at": "ISO-8601 timestamp",
        "model_id": str,          # the served model identifier
        "v": 1,                   # suite version
    }

The aggregator combines three tier-results into a final score:

    final = 0.20 * tier1_score + 0.50 * tier2_score + 0.30 * tier3_score

Per the locked spec, `final_score` is what G0 sees as its reward
signal for G2.
"""

from __future__ import annotations

from typing import Any, TypedDict

SUITE_VERSION: int = 1

# The three tier weights MUST sum to 1.0. These are the locked
# v1 weights — changing them is a v2 event.
TIER_WEIGHTS: dict[int, float] = {1: 0.20, 2: 0.50, 3: 0.30}


class TierResult(TypedDict):
    """The structured result every tier produces."""

    tier: int            # 1, 2, or 3
    score: float         # 0.0 to 100.0 (normalized)
    components: dict[str, Any]
    ran_at: str          # ISO-8601 UTC
    model_id: str
    v: int               # SUITE_VERSION


class AggregateResult(TypedDict):
    """The final result the aggregator produces."""

    final_score: float
    tier1: TierResult | None
    tier2: TierResult | None
    tier3: TierResult | None
    partial: bool        # True if any tier is None
    v: int


def aggregate(
    tier1: TierResult | None,
    tier2: TierResult | None,
    tier3: TierResult | None,
) -> AggregateResult:
    """Combine tier results into a weighted final score.

    Missing tiers (None) contribute 0 and set ``partial=True``.
    A partial result is research data, not a comparable result —
    callers should not compare a partial score to a complete one.
    """
    partial = any(t is None for t in (tier1, tier2, tier3))
    components = [
        (tier1, TIER_WEIGHTS[1]),
        (tier2, TIER_WEIGHTS[2]),
        (tier3, TIER_WEIGHTS[3]),
    ]
    final = sum(t["score"] * w for t, w in components if t is not None)
    return AggregateResult(
        final_score=final,
        tier1=tier1,
        tier2=tier2,
        tier3=tier3,
        partial=partial,
        v=SUITE_VERSION,
    )


# Tier implementations are imported lazily — keeping the suite
# surface importable even when downstream deps (lm-eval-harness,
# peft, etc.) are missing. Use the explicit module paths to load:
#
#   from lamarck.eval.tier1_external import run_tier1
#   from lamarck.eval.tier2_engineering import run_tier2
#   from lamarck.eval.tier3_grounded import run_tier3
#
# Each returns a TierResult dict; combine via aggregate().
