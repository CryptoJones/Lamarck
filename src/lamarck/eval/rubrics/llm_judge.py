# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
"""Tier 2 LLM-judge rubric runner for the 4-axis rubric type.

Per the locked v1 spec, the curriculum-design and diagnostics
categories are graded by an LLM judge against a fixed 4-axis
rubric (1 binary point per axis). The shape of the rubric:

  * curriculum-design: completeness, ordering, specificity, plausibility
  * diagnostics:       correct_root_cause, viable_fix, no_new_bug,
                       mechanistic_explanation

This runner builds a judge prompt that includes the task prompt,
the reference solution, the candidate's output, and an explicit
per-axis scoring instruction. The judge responds with a JSON object
mapping each axis_id to 0 or 1, and the runner returns a
``RubricResult`` with the sum.

## Dependency injection

``score_llm_judge(task, model_output, *, judge_query=None,
                  judge_model_id='gpt-4o-mini',
                  base_url='http://localhost:8000/v1',
                  timeout=60)``

- ``judge_query`` is a callable ``(prompt, model_id, base_url,
  timeout) -> str`` that issues the judge API call. Defaults to a
  stdlib-only OpenAI-compatible ``/chat/completions`` POST (same
  shape as the L4 default model querier). Tests inject a mock.

- ``judge_model_id`` is the model the judge runs as. Picked by the
  caller; v1 doesn't lock a specific judge model, but L19's
  readiness review locks one before any G_N run.

## Failure modes

  * judge_query raises -> rationale "judge-error: <type>: <msg>",
    score 0/max.
  * Response not extractable JSON -> "judge-malformed-output".
  * Required axes missing -> "judge-incomplete: missing {axes}".
  * Per-axis value not 0 or 1 -> coerced (clamp + cast) but
    surfaced in rationale.
  * Judge refused (response contains common refusal substrings) ->
    "judge-refused".

All failure modes return a structured ``RubricResult`` rather than
raising so the Tier 2 dispatcher can keep scoring other tasks.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from ..tier2_engineering import RubricResult, Task


# Axis sets the judge runner supports. Adding a new rubric type
# means extending RUBRIC_AXES + verifying the task corpora ship
# rubrics with the matching axis-id-set.
RUBRIC_AXES: dict[str, tuple[str, ...]] = {
    "llm-judge-4axis-curriculum": (
        "completeness", "ordering", "specificity", "plausibility",
    ),
    "llm-judge-4axis-diagnostics": (
        "correct_root_cause", "viable_fix",
        "no_new_bug", "mechanistic_explanation",
    ),
}

# Common refusal substrings (case-insensitive) the judge might emit
# when it refuses to grade. We treat refusal as a grading failure
# (0/max + explicit rationale) rather than a default-pass.
_REFUSAL_PATTERNS: tuple[str, ...] = (
    "i cannot grade",
    "i can't grade",
    "i won't grade",
    "i refuse",
    "i'm unable to",
    "i am unable to",
)


JudgeQuery = Callable[[str, str, str, int], str]
"""(prompt, model_id, base_url, timeout) -> judge_response_text."""


# ---- Prompt construction ---------------------------------------------------

_JUDGE_SYSTEM = (
    "You are a strict grading assistant. You will receive a task "
    "prompt, a reference solution, and a candidate response. Grade "
    "the candidate response against a 4-axis rubric. Each axis gets "
    "1 if the candidate satisfies the criterion, 0 otherwise. "
    "Respond ONLY with a single JSON object of the form "
    '{"axis_id": 0|1, ...} - no prose, no explanation, no code fences.'
)


def _build_judge_prompt(task: Task, model_output: str, axes: tuple[str, ...]) -> str:
    """Stitch task + reference + candidate + per-axis criteria.

    The judge sees the rubric's per-axis description verbatim so the
    grading aligns with what the task author intended.
    """
    rubric = task.get("rubric", {})
    criteria = rubric.get("criteria", [])
    by_id = {c.get("id"): c for c in criteria}

    axis_lines = []
    for axis in axes:
        crit = by_id.get(axis, {})
        desc = crit.get("description", f"<no description for axis {axis}>")
        axis_lines.append(f'  - "{axis}": {desc}')
    axes_block = "\n".join(axis_lines)

    expected_keys = ", ".join('"' + a + '": 0' for a in axes)
    return (
        f"Task prompt:\n```\n{task.get('prompt', '')}\n```\n\n"
        f"Reference solution:\n```\n{task.get('reference_solution', '')}\n```\n\n"
        f"Candidate response:\n```\n{model_output}\n```\n\n"
        f"Rubric (4 axes, each 0 or 1):\n{axes_block}\n\n"
        f"Return your verdict ONLY as JSON: "
        f"{{{expected_keys}}} (with 0 or 1 per axis)."
    )


# ---- Default judge query (stdlib-only) ------------------------------------

def _default_judge_query(prompt: str, model_id: str, base_url: str,
                         timeout: int) -> str:
    """POST to an OpenAI-compatible /chat/completions endpoint."""
    import urllib.request  # noqa: WPS433 - cheap stdlib lazy import

    body = json.dumps({
        "model": model_id,
        "messages": [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 256,
        "response_format": {"type": "json_object"},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


# ---- Response parsing ------------------------------------------------------

def _extract_json(text: str) -> dict[str, Any] | None:
    """Pull a JSON object out of a possibly-noisy judge response.

    Some judge models wrap JSON in markdown fences or include a
    one-line preamble despite the system prompt. We try a strict
    parse first, then a regex-based extraction of the first {...}
    block.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # Strip a leading ```json fence + trailing ``` if present.
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except (json.JSONDecodeError, ValueError):
            pass
    # Fall back to first balanced-braces block.
    brace_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def _looks_like_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(p in lowered for p in _REFUSAL_PATTERNS)


def _resolve_axes(rubric_type: str, criteria: list[dict[str, Any]]) -> tuple[str, ...]:
    """Determine the axis-id-set for this rubric.

    v1 ships two 4-axis variants (curriculum, diagnostics) that both
    declare rubric.type = "llm-judge-4axis". We disambiguate by the
    axis-id-set the rubric itself declares - that's the contract
    the task corpus tests already lock.
    """
    axis_ids = tuple(c["id"] for c in criteria)
    axis_set = set(axis_ids)
    for canonical_axes in RUBRIC_AXES.values():
        if set(canonical_axes) == axis_set:
            return canonical_axes
    # Unknown axis set - fall through to a generic ordering.
    return axis_ids


# ---- Public entry point ----------------------------------------------------

def score_llm_judge(
    task: Task,
    model_output: str,
    *,
    judge_query: JudgeQuery | None = None,
    judge_model_id: str = "gpt-4o-mini",
    base_url: str = "http://localhost:8000/v1",
    timeout: int = 60,
) -> RubricResult:
    """Grade ``model_output`` for a 4-axis LLM-judge rubric task.

    Returns a ``RubricResult`` with score in [0, max_score]. On any
    failure path (judge error / refusal / malformed output / missing
    axes), returns score 0/max with an explicit failure rationale.
    """
    rubric = task.get("rubric", {})
    rubric_type = rubric.get("type", "")
    criteria = rubric.get("criteria", [])
    max_score = int(rubric.get("max_score", 0))

    if rubric_type != "llm-judge-4axis":
        return RubricResult(
            score=0, max_score=max(max_score, 4),
            rationale=(f"llm_judge=0: unsupported rubric type "
                       f"{rubric_type!r}"),
        )

    axes = _resolve_axes(rubric_type, criteria)
    if len(axes) != 4 or max_score != 4:
        return RubricResult(
            score=0, max_score=max(max_score, 4),
            rationale=(f"llm_judge=0: corrupt rubric - axes={axes}, "
                       f"max_score={max_score}"),
        )

    # Empty / whitespace-only candidate: short-circuit with 0/4.
    if not model_output or not model_output.strip():
        return RubricResult(
            score=0, max_score=4,
            rationale="llm_judge=0: empty candidate response",
        )

    query = judge_query if judge_query is not None else _default_judge_query
    prompt = _build_judge_prompt(task, model_output, axes)

    try:
        raw = query(prompt, judge_model_id, base_url, timeout)
    except Exception as exc:  # noqa: BLE001 - surface every failure mode
        return RubricResult(
            score=0, max_score=4,
            rationale=f"llm_judge=0: judge-error: {type(exc).__name__}: {exc}",
        )

    if _looks_like_refusal(raw):
        return RubricResult(
            score=0, max_score=4,
            rationale=f"llm_judge=0: judge-refused (raw: {raw[:120]!r})",
        )

    parsed = _extract_json(raw)
    if parsed is None:
        return RubricResult(
            score=0, max_score=4,
            rationale=(f"llm_judge=0: judge-malformed-output "
                       f"(raw: {raw[:200]!r})"),
        )

    missing = [a for a in axes if a not in parsed]
    if missing:
        return RubricResult(
            score=0, max_score=4,
            rationale=f"llm_judge=0: judge-incomplete: missing axes {missing}",
        )

    score = 0
    axis_lines = []
    for axis in axes:
        raw_val = parsed.get(axis)
        # Coerce: truthy non-zero -> 1, zero / falsey -> 0. Surface
        # unexpected values in the rationale rather than failing.
        try:
            v = int(raw_val)
        except (TypeError, ValueError):
            axis_lines.append(f"{axis}=0(coerced from {raw_val!r})")
            continue
        if v < 0 or v > 1:
            v = 1 if v >= 1 else 0
            axis_lines.append(f"{axis}={v}(clamped)")
        else:
            axis_lines.append(f"{axis}={v}")
        score += v

    return RubricResult(
        score=score, max_score=4,
        rationale="llm_judge: " + ", ".join(axis_lines),
    )
