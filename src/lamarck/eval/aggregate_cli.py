# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
"""End-to-end aggregation CLI.

Reads the per-tier result JSON files emitted by the Tier 1 / 2 / 3
runners and produces the final aggregated score via the locked v1
``aggregate()`` function defined in ``lamarck.eval.__init__``.

Usage::

  python -m lamarck.eval.aggregate \\
      --tier1 t1.json --tier2 t2.json --tier3 t3.json \\
      --out   final.json

Any of ``--tier1``/``--tier2``/``--tier3`` may be omitted - the
aggregator marks the result as ``partial`` and the missing tier
contributes 0 to the weighted sum. A partial score is research
data, not a comparable result.

``--out -`` writes the AggregateResult to stdout.

The CLI is thin glue: validation lives in ``load_tier_result``,
arithmetic in the shared ``aggregate()`` function, IO here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import AggregateResult, TierResult, aggregate


# Required keys every tier-result JSON must declare. Locked under v1.
TIER_RESULT_REQUIRED_KEYS: tuple[str, ...] = (
    "tier", "score", "components", "ran_at", "model_id", "v",
)


class TierResultError(ValueError):
    """Raised by ``load_tier_result`` when a file fails validation."""


def load_tier_result(path: Path, expected_tier: int) -> TierResult:
    """Load + structurally validate a tier-result JSON file.

    Raises ``TierResultError`` (a ValueError subclass) on:
      - file not found
      - non-JSON contents
      - missing required keys
      - tier value mismatching ``expected_tier``
      - score not in [0, 100]
      - suite version (``v``) not 1

    Returns the loaded dict, typed as ``TierResult``. The caller can
    feed it directly into ``aggregate()``.
    """
    if not path.exists():
        raise TierResultError(f"tier-result file not found: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise TierResultError(f"{path}: not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise TierResultError(f"{path}: top-level must be a JSON object")
    missing = [k for k in TIER_RESULT_REQUIRED_KEYS if k not in data]
    if missing:
        raise TierResultError(
            f"{path}: missing required keys: {missing}"
        )
    if data["tier"] != expected_tier:
        raise TierResultError(
            f"{path}: tier {data['tier']} mismatches expected {expected_tier}"
        )
    if not isinstance(data["score"], (int, float)):
        raise TierResultError(f"{path}: score must be a number")
    if not (0.0 <= float(data["score"]) <= 100.0):
        raise TierResultError(
            f"{path}: score {data['score']} not in [0, 100]"
        )
    if data["v"] != 1:
        raise TierResultError(
            f"{path}: suite version v={data['v']} not 1 (run is v2+? "
            f"cannot aggregate across versions)"
        )
    return data  # type: ignore[return-value]


def _load_optional(path: str | None, expected_tier: int) -> TierResult | None:
    if not path:
        return None
    return load_tier_result(Path(path), expected_tier)


def _serialize(out: str, payload: AggregateResult) -> None:
    """Write the aggregate result either to a file path or stdout."""
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if out == "-":
        sys.stdout.write(rendered)
        sys.stdout.flush()
        return
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(rendered)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m lamarck.eval.aggregate",
        description=(
            "Aggregate per-tier eval results into the final v1 score. "
            "Any tier may be omitted - the result will be marked partial."
        ),
    )
    p.add_argument("--tier1", help="Path to the Tier 1 result JSON")
    p.add_argument("--tier2", help="Path to the Tier 2 result JSON")
    p.add_argument("--tier3", help="Path to the Tier 3 result JSON")
    p.add_argument(
        "--out", required=True,
        help="Output path for the aggregated JSON, or '-' for stdout",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        t1 = _load_optional(args.tier1, expected_tier=1)
        t2 = _load_optional(args.tier2, expected_tier=2)
        t3 = _load_optional(args.tier3, expected_tier=3)
    except TierResultError as exc:
        # CLI surfaces validation errors clearly on stderr + non-zero exit.
        sys.stderr.write(f"aggregate: {exc}\n")
        return 2

    result = aggregate(t1, t2, t3)
    _serialize(args.out, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
