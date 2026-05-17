#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
"""Run a single tier and emit its TierResult JSON.

Used by ``scripts/eval/run-all.sh`` to invoke each tier in a
uniform way; can also be run directly for debugging.

Usage::

  python scripts/eval/run_tier.py --tier 1 --model-id lamarck-g1 \\
      --base-url http://localhost:8000/v1 --out tier1.json

  python scripts/eval/run_tier.py --tier 3 --model-id lamarck-g1 \\
      --base-url http://localhost:8000/v1 \\
      --curriculum-jsonl ./curriculum.jsonl --out tier3.json

  python scripts/eval/run_tier.py --tier 2 --mock --out tier2.json

``--mock`` emits a fixed TierResult instead of invoking the real
runner. The mock scores (T1=75.0, T2=60.0, T3=40.0) are chosen so
the aggregated final score in mock mode is:
  0.20 * 75 + 0.50 * 60 + 0.30 * 40 = 15 + 30 + 12 = 57.0

The exit code is 0 on success, 2 on argument / IO errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Mock scores per tier. Sum to 57.0 once weighted - locks the run-all
# smoke test (L19) to a stable expected value.
MOCK_SCORES: dict[int, float] = {1: 75.0, 2: 60.0, 3: 40.0}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _mock_tier_result(tier: int, model_id: str) -> dict[str, Any]:
    return {
        "tier":       tier,
        "score":      MOCK_SCORES[tier],
        "components": {"mock": True},
        "ran_at":     _utc_now(),
        "model_id":   model_id,
        "v":          1,
    }


def _run_real(tier: int, args: argparse.Namespace) -> dict[str, Any]:
    """Invoke the real tier runner with the standard signature."""
    if tier == 1:
        from lamarck.eval.tier1_external import run_tier1
        return run_tier1(model_id=args.model_id, base_url=args.base_url)
    if tier == 2:
        from lamarck.eval.tier2_engineering import run_tier2
        return run_tier2(model_id=args.model_id, base_url=args.base_url)
    if tier == 3:
        if not args.curriculum_jsonl:
            raise ValueError(
                "tier 3 requires --curriculum-jsonl"
            )
        from lamarck.eval.tier3_grounded import run_tier3
        return run_tier3(
            model_id=args.model_id, base_url=args.base_url,
            curriculum_jsonl=Path(args.curriculum_jsonl),
        )
    raise ValueError(f"unknown tier: {tier}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_tier.py",
        description="Run a single eval tier and emit its TierResult JSON.",
    )
    p.add_argument("--tier", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--model-id", default="lamarck-g0")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument(
        "--curriculum-jsonl",
        help="Tier 3 only: path to G_N's curriculum JSONL",
    )
    p.add_argument("--out", required=True,
                   help="Output path for the TierResult JSON, or '-'")
    p.add_argument(
        "--mock", action="store_true",
        help="Emit a fixed mock TierResult instead of invoking the runner",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.mock:
            result = _mock_tier_result(args.tier, args.model_id)
        else:
            result = _run_real(args.tier, args)
    except ValueError as exc:
        sys.stderr.write(f"run_tier: {exc}\n")
        return 2
    except ImportError as exc:
        # Real runner needs deps that aren't installed - surface clearly.
        sys.stderr.write(
            f"run_tier: real runner unavailable: {type(exc).__name__}: {exc}\n"
            f"  hint: pass --mock for CI-style runs without deps.\n"
        )
        return 2

    rendered = json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
    if args.out == "-":
        sys.stdout.write(rendered)
        sys.stdout.flush()
        return 0
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
