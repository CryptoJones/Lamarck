# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
"""Tier 3 pass-rate scorer.

Given a fine-tuned tiny model (represented by its adapter Path)
and the locked held-out problem set, run inference per problem
and compute the pass rate. A problem passes iff the model output:

  1. Contains an extractable JSON object (some models wrap their
     answer in prose; this scorer is lenient about that).
  2. Parses successfully via ``json.loads``.
  3. Validates against the problem's JSON Schema.

The third step uses the ``jsonschema`` library if installed; if
not, it falls back to a structural check that the parsed JSON is
an object containing every key listed in ``schema['required']``.
The fallback is deliberately permissive (no type checks on
values) so a missing jsonschema dep underestimates pass rate
rather than crashes.

The module is fully test-isolated: ``score_json_mode`` takes the
``inference_fn`` as an argument, identical in signature to
``tier3_grounded.InferenceFn``. The default is the same
``requires-gpu-runtime`` stub so accidental CI calls fail loudly.

## Result shape

``score_json_mode(adapter_path, holdout, inference_fn) -> dict``::

  {
    "pass_count": int,
    "total":      int,
    "fraction":   float,                 # pass_count / total
    "score":      float,                 # fraction * 100, rounded
    "per_problem": [
      {"idx": int,
       "passed": bool,
       "reason": str,                    # short tag explaining failure
       "raw_excerpt": str},              # first 200 chars of output
      ...
    ],
    "errors": list[str],                 # inference failures by idx
  }
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from ..tier3_grounded import HoldoutProblem, InferenceFn


# ---- Result types ----------------------------------------------------------

PASS_REASON = "ok"
REASON_NO_JSON  = "no_json_found"
REASON_PARSE   = "parse_error"
REASON_SCHEMA  = "schema_validation_failed"
REASON_INFER   = "inference_error"


# ---- JSON extraction -------------------------------------------------------

def _extract_first_json_object(text: str) -> str | None:
    """Pull the first balanced ``{...}`` block out of the model output.

    Strategies, tried in order:
      1. Strict json.loads on the whole stripped string.
      2. Strip a leading ```json fence + trailing ``` if present.
      3. Walk the string finding the first ``{``, scan forward
         counting braces (respecting strings) until the matching
         ``}``.

    Returns the substring (still raw, caller json.loads it) or
    None if no balanced object can be found. Doesn't validate -
    the caller json.loads + jsonschema-validates separately.
    """
    if not text or not text.strip():
        return None
    stripped = text.strip()

    # Fast path: whole-string parses.
    try:
        json.loads(stripped)
        return stripped
    except (json.JSONDecodeError, ValueError):
        pass

    # ```json ... ``` fence.
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```",
                             stripped, re.DOTALL)
    if fence_match:
        return fence_match.group(1)

    # Brace-balanced scan from the first `{`. Respects strings so
    # `{"a": "}"}` doesn't end prematurely.
    start = stripped.find("{")
    if start < 0:
        return None
    depth = 0
    i = start
    in_string = False
    escape = False
    while i < len(stripped):
        ch = stripped[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return stripped[start:i + 1]
        i += 1
    return None  # unbalanced


# ---- Schema validation -----------------------------------------------------

def _validate_against_schema(
    parsed: Any, schema: dict[str, Any],
) -> tuple[bool, str]:
    """Validate ``parsed`` against ``schema``.

    Uses ``jsonschema.validate`` if installed; otherwise falls back to:
      - top-level type must be object
      - every key in schema['required'] must be present
      - no type/enum/pattern checks on values

    Returns ``(passed, reason)``. The reason is the empty string on
    pass; on fail it's a short tag (or jsonschema's error message
    when present).
    """
    try:
        from jsonschema import validate, ValidationError  # type: ignore
    except ImportError:
        # Fallback: structural check.
        expected_type = schema.get("type")
        if expected_type == "object" and not isinstance(parsed, dict):
            return False, "fallback: expected object"
        if expected_type == "array" and not isinstance(parsed, list):
            return False, "fallback: expected array"
        required = schema.get("required", [])
        if isinstance(parsed, dict):
            missing = [k for k in required if k not in parsed]
            if missing:
                return False, f"fallback: missing required keys {missing}"
        return True, ""
    try:
        validate(instance=parsed, schema=schema)
    except ValidationError as exc:
        # Compact reason; jsonschema's full message includes a
        # multi-line dump of the schema which floods rationales.
        path = "/".join(str(p) for p in exc.absolute_path) or "<root>"
        return False, f"{exc.message} at {path}"
    return True, ""


# ---- Per-problem scorer ----------------------------------------------------

def score_one_output(model_output: str, schema: dict[str, Any]) -> tuple[bool, str]:
    """Score a single (output, schema) pair. Returns (passed, reason)."""
    extracted = _extract_first_json_object(model_output)
    if extracted is None:
        return False, REASON_NO_JSON
    try:
        parsed = json.loads(extracted)
    except (json.JSONDecodeError, ValueError) as exc:
        return False, f"{REASON_PARSE}: {exc}"
    passed, why = _validate_against_schema(parsed, schema)
    if not passed:
        return False, f"{REASON_SCHEMA}: {why}"
    return True, PASS_REASON


# ---- Default inference stub (require GPU) ---------------------------------

def _requires_gpu_inference(adapter_path: Path, prompt: str) -> str:
    raise RuntimeError(
        "Tier 3 inference_fn requires-gpu-runtime - inject a real "
        "implementation or a mock for testing."
    )


# ---- Public scorer ---------------------------------------------------------

def score_json_mode(
    adapter_path: Path,
    holdout: list[HoldoutProblem],
    inference_fn: InferenceFn | None = None,
    *,
    output_truncate_chars: int = 200,
) -> dict[str, Any]:
    """Run ``inference_fn`` on each held-out problem and score the result.

    Returns a structured result dict (shape documented in the module
    docstring). Never raises - inference exceptions land in the
    per_problem entries' rationale AND aggregate errors list, the
    score is computed over surviving problems.
    """
    inf = inference_fn if inference_fn is not None else _requires_gpu_inference

    per_problem: list[dict[str, Any]] = []
    errors: list[str] = []
    pass_count = 0

    for idx, problem in enumerate(holdout):
        try:
            output = inf(adapter_path, problem.input)
        except Exception as exc:  # noqa: BLE001 - we surface all failures
            errors.append(f"problem {idx}: {type(exc).__name__}: {exc}")
            per_problem.append({
                "idx":         idx,
                "passed":      False,
                "reason":      f"{REASON_INFER}: {type(exc).__name__}",
                "raw_excerpt": "",
            })
            continue

        passed, reason = score_one_output(output, problem.schema)
        if passed:
            pass_count += 1
        per_problem.append({
            "idx":         idx,
            "passed":      passed,
            "reason":      reason,
            "raw_excerpt": (output or "")[:output_truncate_chars],
        })

    total = len(holdout)
    fraction = (pass_count / total) if total else 0.0
    return {
        "pass_count":  pass_count,
        "total":       total,
        "fraction":    fraction,
        "score":       round(fraction * 100.0, 2),
        "per_problem": per_problem,
        "errors":      errors,
    }
