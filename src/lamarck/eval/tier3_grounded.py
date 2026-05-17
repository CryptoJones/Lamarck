# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
"""Tier 3 - grounded teaching capability.

Per the locked v1 spec (``docs/eval-suite-v1.md``), Tier 3 is the
gaming-resistant test: G_N produces a curriculum (a JSONL of
(prompt, completion) pairs) meant to teach a tiny student model
to respond in valid JSON-mode. We then:

  1. Fine-tune the locked-in student model on G_N's curriculum
     with a fully fixed recipe.
  2. Test the resulting fine-tuned model against a 100-problem
     held-out set (NOT exposed to G_N).
  3. Tier 3 score = pass-rate on the held-out set, normalized to
     0-100.

The student-model spec, the fine-tune recipe, and the held-out
set are all locked at v1 birth. G_N's only degree of freedom is
the curriculum content. Why this is gaming-resistant:

  * G_N never sees the held-out set.
  * The fine-tune recipe is fixed - "win by recommending a magic
    learning rate" is not available.
  * The tiny model can't be prompt-engineered; it has to actually
    learn from the curriculum.
  * "Talking convincingly about training" doesn't help; only
    curricula that produce a downstream-capable model score well.

## Dependency injection

The CPU-only CI environment can't actually run the fine-tune or
the tiny-model inference (both need GPU). The module exports two
injection seams:

  * ``fine_tune_fn``  - called with (student_spec, recipe,
    curriculum_jsonl) -> path to the resulting adapter directory.
  * ``inference_fn``  - called with (adapter_path, prompt) ->
    the model's text completion.

Both default to a stub that raises ``RuntimeError(
"requires-gpu-runtime")``. Tests inject mocks. The L15 pass-rate
scorer (in ``rubrics/json_mode.py``) consumes ``inference_fn``
to actually score the held-out set.

## What this module DOES NOT do

  * It does not implement the fine-tune. Real GPU implementation
    lands as a separate module wired through this skeleton.
  * It does not own the held-out set; that's L14
    (``tier3_holdout/json_mode_problems.jsonl``).
  * It does not own the per-problem pass/fail scoring; that's L15
    (``rubrics/json_mode.py``).

It owns the locked spec, the orchestration, and the contract
that L14/L15 and any future real fine-tune implementation must
satisfy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from . import SUITE_VERSION, TierResult


# ---- Locked v1 spec --------------------------------------------------------

# The student model. Locked at v1 birth. Changing this is a v2 event.
STUDENT_MODEL_ID: str = "meta-llama/Llama-3.2-1B-Instruct"

# The held-out test set lives under tier3_holdout/ adjacent to this module.
# L14 populates the file; until then this path resolves but the file is empty.
HOLDOUT_RELATIVE_PATH: str = "tier3_holdout/json_mode_problems.jsonl"

# Number of held-out problems. v1 locks this at 100 - the pass rate is
# tier3_score / 100 * 100 = tier3_score (i.e. pass count IS the score).
HOLDOUT_PROBLEM_COUNT: int = 100


@dataclass(frozen=True)
class FineTuneRecipe:
    """The locked fine-tune recipe for Tier 3.

    Per v1: 1 epoch, batch size 4, learning rate 2e-4, LoRA rank 16.
    All values are FIXED. G_N does not get to tune these; G_N only
    gets to design the curriculum (training data).
    """

    epochs:              int = 1
    per_device_batch:    int = 4
    learning_rate:       float = 2e-4
    lora_rank:           int = 16
    lora_alpha:          int = 32
    lora_dropout:        float = 0.05
    seed:                int = 42  # determinism for reproducible runs

    def as_dict(self) -> dict[str, Any]:
        return {
            "epochs":           self.epochs,
            "per_device_batch": self.per_device_batch,
            "learning_rate":    self.learning_rate,
            "lora_rank":        self.lora_rank,
            "lora_alpha":       self.lora_alpha,
            "lora_dropout":     self.lora_dropout,
            "seed":             self.seed,
        }


# The single instance. Re-export for convenience so tests can lock the
# numbers via a single identity check.
TIER3_RECIPE: FineTuneRecipe = FineTuneRecipe()


# ---- Type contracts --------------------------------------------------------

@dataclass(frozen=True)
class StudentSpec:
    """The locked student model identity passed to the fine-tuner."""
    model_id: str = STUDENT_MODEL_ID


class FineTuneFn(Protocol):
    """Signature every Tier 3 fine-tune implementation must satisfy.

    Takes (student spec, recipe, curriculum-jsonl path) and produces
    a path to the resulting adapter directory.
    """

    def __call__(
        self,
        student: StudentSpec,
        recipe: FineTuneRecipe,
        curriculum_jsonl: Path,
        output_dir: Path,
    ) -> Path: ...


class InferenceFn(Protocol):
    """Signature for invoking the fine-tuned tiny model on one prompt.

    Takes (adapter path, prompt) and returns the model's completion
    text. The L15 scorer wraps this with held-out-problem iteration
    and per-problem JSON-schema validation.
    """

    def __call__(self, adapter_path: Path, prompt: str) -> str: ...


@dataclass(frozen=True)
class HoldoutProblem:
    """One row of the held-out set.

    ``input`` is the prompt fed to the fine-tuned student model.
    ``schema`` is a JSON Schema (draft-07) the model's response
    must validate against.
    """
    input:  str
    schema: dict[str, Any]


# ---- Default fn stubs (require GPU) ---------------------------------------

def _requires_gpu_fine_tune(
    student: StudentSpec, recipe: FineTuneRecipe,
    curriculum_jsonl: Path, output_dir: Path,
) -> Path:
    raise RuntimeError(
        "Tier 3 fine_tune_fn requires-gpu-runtime - inject a real "
        "implementation or a mock for testing."
    )


def _requires_gpu_inference(adapter_path: Path, prompt: str) -> str:
    raise RuntimeError(
        "Tier 3 inference_fn requires-gpu-runtime - inject a real "
        "implementation or a mock for testing."
    )


# ---- Held-out loading ------------------------------------------------------

def default_holdout_path() -> Path:
    """The canonical location L14 will populate."""
    return Path(__file__).parent / HOLDOUT_RELATIVE_PATH


def load_holdout(path: Path | None = None) -> list[HoldoutProblem]:
    """Read the held-out JSONL. Each line must be a JSON object with
    keys ``input`` (str) and ``schema`` (dict).

    Returns ``[]`` if the file doesn't exist - the run_tier3 caller
    reports the gap rather than crashing, which is what L14 needs
    when it ships the file separately.
    """
    p = path if path is not None else default_holdout_path()
    if not p.exists():
        return []
    problems: list[HoldoutProblem] = []
    with p.open() as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            missing = [k for k in ("input", "schema") if k not in obj]
            if missing:
                raise ValueError(
                    f"holdout {p}:{line_no} missing keys: {missing}"
                )
            problems.append(HoldoutProblem(
                input=obj["input"], schema=obj["schema"],
            ))
    return problems


# ---- Pass-rate scoring (default) ------------------------------------------

def _score_one(model_output: str, schema: dict[str, Any]) -> bool:
    """Default per-problem scorer: does the model's output parse as
    JSON AND validate against the schema?

    Uses the optional ``jsonschema`` library if available; falls back
    to "parse-and-must-be-a-dict" if not. L15 replaces this with the
    canonical jsonschema validator when it lands.
    """
    try:
        parsed = json.loads(model_output)
    except (json.JSONDecodeError, ValueError):
        return False
    try:
        from jsonschema import validate  # type: ignore[import-untyped]
    except ImportError:
        # Fallback: minimal type check.
        expected_type = schema.get("type")
        if expected_type == "object":
            return isinstance(parsed, dict)
        if expected_type == "array":
            return isinstance(parsed, list)
        return True
    try:
        validate(instance=parsed, schema=schema)
    except Exception:  # noqa: BLE001 - jsonschema.ValidationError + friends
        return False
    return True


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---- Public entry point ----------------------------------------------------

def run_tier3(
    model_id: str,
    base_url: str = "http://localhost:8000/v1",
    *,
    curriculum_jsonl: Path | None = None,
    holdout_path: Path | None = None,
    fine_tune_fn: FineTuneFn | None = None,
    inference_fn: InferenceFn | None = None,
    output_dir: Path | None = None,
) -> TierResult:
    """Run the full Tier 3 evaluation pipeline.

    Steps:
      1. Resolve curriculum + held-out set.
      2. Call fine_tune_fn(student, recipe, curriculum, output_dir).
      3. For each held-out problem, call inference_fn(adapter, input).
      4. Score each (output, schema) pair as a binary pass/fail.
      5. Aggregate into a TierResult; score == 100 * pass_count /
         len(holdout) (so 100 problems means score == pass_count).

    Failure modes - each returns a partial TierResult naming the
    issue rather than raising, so the suite-level aggregator can
    decide whether to accept a partial run:
      - No curriculum supplied -> partial, no fine-tune attempted.
      - No held-out set -> partial.
      - fine_tune_fn raises -> partial with the error.
      - inference_fn raises on a problem -> that problem scores 0,
        the error lands in components.errors; other problems still
        run.
    """
    ft = fine_tune_fn if fine_tune_fn is not None else _requires_gpu_fine_tune
    inf = inference_fn if inference_fn is not None else _requires_gpu_inference
    out_dir = output_dir if output_dir is not None else Path("./tier3_out")

    errors: dict[str, Any] = {}
    components: dict[str, Any] = {
        "student_model_id": STUDENT_MODEL_ID,
        "recipe":           TIER3_RECIPE.as_dict(),
    }

    if curriculum_jsonl is None or not Path(curriculum_jsonl).exists():
        errors["curriculum"] = (
            f"no curriculum supplied or path missing: {curriculum_jsonl!r}"
        )
        components["errors"] = errors
        return TierResult(
            tier=3, score=0.0, components=components,
            ran_at=_utc_now(), model_id=model_id, v=SUITE_VERSION,
        )

    holdout = load_holdout(holdout_path)
    if not holdout:
        errors["holdout"] = (
            f"held-out set empty or missing at "
            f"{(holdout_path or default_holdout_path())!r}"
        )
        components["errors"] = errors
        return TierResult(
            tier=3, score=0.0, components=components,
            ran_at=_utc_now(), model_id=model_id, v=SUITE_VERSION,
        )

    components["holdout_size"] = len(holdout)

    try:
        adapter_path = ft(
            StudentSpec(),
            TIER3_RECIPE,
            Path(curriculum_jsonl),
            out_dir,
        )
    except Exception as exc:  # noqa: BLE001 - surface all failure modes
        errors["fine_tune"] = f"{type(exc).__name__}: {exc}"
        components["errors"] = errors
        return TierResult(
            tier=3, score=0.0, components=components,
            ran_at=_utc_now(), model_id=model_id, v=SUITE_VERSION,
        )

    components["adapter_path"] = str(adapter_path)
    pass_count = 0
    problem_errors: list[str] = []
    for idx, problem in enumerate(holdout):
        try:
            output = inf(adapter_path, problem.input)
        except Exception as exc:  # noqa: BLE001
            problem_errors.append(
                f"problem {idx}: {type(exc).__name__}: {exc}"
            )
            continue
        if _score_one(output, problem.schema):
            pass_count += 1

    components["pass_count"] = pass_count
    if problem_errors:
        errors["inference_errors"] = problem_errors
        components["errors"] = errors

    # Pass rate normalized to [0, 100].
    score = round(100.0 * pass_count / len(holdout), 2)

    return TierResult(
        tier=3, score=score, components=components,
        ran_at=_utc_now(), model_id=model_id, v=SUITE_VERSION,
    )
