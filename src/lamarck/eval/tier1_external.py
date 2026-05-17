# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
"""Tier 1 — external sanity benchmarks.

Per the locked v1 spec (``docs/eval-suite-v1.md``), Tier 1 runs three
off-the-shelf benchmarks via ``lm-evaluation-harness`` and reports the
mean as a 0-100 score:

  * MMLU-Pro (subsets: computer_science, engineering, math, physics).
  * HumanEval+ — 164 augmented Python coding problems.
  * GSM8K — 1319 grade-school math word problems.

Each benchmark contributes equally to the Tier 1 aggregate (1/3 weight
each inside Tier 1). Tier 1's aggregate then enters the suite-level
aggregation with weight 0.20.

The contract — the function ``run_tier1`` — returns a
``lamarck.eval.TierResult`` dict. Per-benchmark fractions are exposed
in ``components`` so a caller can diagnose which benchmark moved.

Caching: per-benchmark results land under ``cache_dir/<benchmark>.json``
(if ``cache_dir`` is provided) so re-runs against the same model_id
avoid re-evaluating an unchanged benchmark.

Optional dep: ``lm_eval`` (a.k.a. ``lm-evaluation-harness``).
Install on the pod via ``pip install --break-system-packages
lm_eval``. The import is lazy so this module remains importable
without the harness installed (e.g. in CI tests that mock it).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import SUITE_VERSION, TierResult


# Locked Tier 1 benchmark spec. Per the v1 lockdown, these strings,
# the subset list for MMLU-Pro, and the metric names are FIXED.
# Changes ship as v2 with a fresh generation lineage.
MMLU_PRO_SUBSETS: tuple[str, ...] = (
    "computer_science",
    "engineering",
    "math",
    "physics",
)

TIER1_BENCHMARKS: tuple[str, ...] = ("mmlu_pro", "humaneval_plus", "gsm8k")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _resolve_simple_evaluate() -> Callable[..., Any]:
    """Lazy import so the module loads in environments without lm_eval.

    Returns the ``lm_eval.simple_evaluate`` callable.

    Tests should monkeypatch this function (not the underlying
    ``lm_eval`` import) so they don't need lm_eval installed.
    """
    import lm_eval  # noqa: WPS433 (lazy import)

    return lm_eval.simple_evaluate


def _load_cached(cache_dir: Path | None, benchmark: str) -> dict[str, Any] | None:
    if cache_dir is None:
        return None
    path = cache_dir / f"{benchmark}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _write_cache(cache_dir: Path | None, benchmark: str, payload: dict[str, Any]) -> None:
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{benchmark}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def _model_args(model_id: str, base_url: str) -> str:
    """Build the lm_eval --model_args string for an OpenAI-compatible endpoint.

    lm-eval-harness's ``local-completions`` model adapter accepts
    ``base_url=...`` and ``model=...`` as comma-separated kwargs. The
    served vLLM endpoint advertises ``model=lamarck-g${GEN}`` (see
    ``scripts/runpod/serve.sh``); whatever the caller hands us as
    ``model_id`` is what we pass through.
    """
    return f"model={model_id},base_url={base_url}"


def _extract_score(results: dict[str, Any], task: str) -> float:
    """Pull a fraction-correct score (0.0-1.0) out of an lm_eval results dict.

    lm_eval's ``simple_evaluate`` returns ``{"results": {task: {metric:
    value}}}``. We accept any of these primary metric names in this
    order — different task configs report under different keys, and v1
    locks the conventional ones:

      * ``acc,none`` (MMLU-Pro and GSM8K)
      * ``exact_match,strict-match`` (GSM8K alt)
      * ``pass@1`` (HumanEval+)

    If none are present we raise ``KeyError`` — the harness output is
    malformed and we shouldn't silently emit a fake score.
    """
    task_results = results.get("results", {}).get(task)
    if not task_results:
        raise KeyError(f"no results for task: {task!r}")
    for key in ("acc,none", "acc", "pass@1,none", "pass@1",
                "exact_match,strict-match", "exact_match"):
        if key in task_results:
            value = float(task_results[key])
            # lm-eval reports 0-1 fractions; clamp + sanity-check.
            if 0.0 <= value <= 1.0:
                return value
            # Some harness versions report as 0-100; coerce that too.
            if 1.0 < value <= 100.0:
                return value / 100.0
            raise ValueError(f"unexpected score range for {task!r}: {value}")
    raise KeyError(f"no recognized metric in task results: {sorted(task_results)}")


def _run_mmlu_pro(model_args: str, simple_evaluate: Callable[..., Any]) -> float:
    """MMLU-Pro restricted to the v1-locked subsets only.

    We pass each subset as a separate task name to lm_eval (the harness
    exposes them as ``mmlu_pro_<subset>``) and average their accuracies
    equally. This gives us the locked-subset behaviour without depending
    on the harness's specific subset-selection API, which has shifted
    across versions.
    """
    tasks = [f"mmlu_pro_{s}" for s in MMLU_PRO_SUBSETS]
    out = simple_evaluate(model="local-completions", model_args=model_args, tasks=tasks)
    scores = [_extract_score(out, task) for task in tasks]
    return sum(scores) / len(scores)


def _run_humaneval_plus(model_args: str, simple_evaluate: Callable[..., Any]) -> float:
    out = simple_evaluate(
        model="local-completions", model_args=model_args, tasks=["humaneval_plus"],
    )
    return _extract_score(out, "humaneval_plus")


def _run_gsm8k(model_args: str, simple_evaluate: Callable[..., Any]) -> float:
    out = simple_evaluate(
        model="local-completions", model_args=model_args, tasks=["gsm8k"],
    )
    return _extract_score(out, "gsm8k")


_BENCH_RUNNERS: dict[str, Callable[[str, Callable[..., Any]], float]] = {
    "mmlu_pro":       _run_mmlu_pro,
    "humaneval_plus": _run_humaneval_plus,
    "gsm8k":          _run_gsm8k,
}


def run_tier1(
    model_id: str,
    base_url: str = "http://localhost:8000/v1",
    cache_dir: Path | None = None,
) -> TierResult:
    """Run Tier 1's three external benchmarks against a served model.

    Per the locked v1 spec, each benchmark contributes equally (1/3
    weight inside Tier 1), and the result is normalized to 0-100.

    Returns a partial TierResult (with ``error`` entries under
    ``components``) if any benchmark fails to run — the caller decides
    whether to accept that as a result or rerun.
    """
    simple_evaluate = _resolve_simple_evaluate()
    args = _model_args(model_id, base_url)

    fractions: dict[str, float | None] = {}
    errors: dict[str, str] = {}

    for bench in TIER1_BENCHMARKS:
        cached = _load_cached(cache_dir, bench)
        if cached is not None and cached.get("model_id") == model_id:
            fractions[bench] = cached["fraction"]
            continue
        runner = _BENCH_RUNNERS[bench]
        try:
            fraction = runner(args, simple_evaluate)
        except Exception as exc:  # noqa: BLE001 — surface every failure mode
            errors[bench] = f"{type(exc).__name__}: {exc}"
            fractions[bench] = None
            continue
        fractions[bench] = fraction
        _write_cache(cache_dir, bench, {
            "benchmark": bench, "model_id": model_id,
            "fraction": fraction, "ran_at": _utc_now(),
        })

    valid = [f for f in fractions.values() if f is not None]
    aggregate_fraction = sum(valid) / len(valid) if valid else 0.0
    score_0_to_100 = aggregate_fraction * 100.0

    components: dict[str, Any] = {
        bench: (None if fractions[bench] is None else round(fractions[bench] * 100.0, 2))
        for bench in TIER1_BENCHMARKS
    }
    if errors:
        components["errors"] = errors

    return TierResult(
        tier=1,
        score=round(score_0_to_100, 2),
        components=components,
        ran_at=_utc_now(),
        model_id=model_id,
        v=SUITE_VERSION,
    )
