# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
"""Unit tests for the PEFT/LoRA 6-point programmatic rubric runner.

Each rung gets a focused test pair (passes when satisfied, fails when
not). The full reference-corpus pass is the integration test at the
bottom - every shipped reference_solution must score 4/6 in static
mode (rungs 1-4) and 6/6 in sandbox mode when the probe is mocked OK.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch, MagicMock

import pytest

from lamarck.eval.rubrics import peft_loop
from lamarck.eval.tier2_engineering import default_tasks_dir, load_tasks


# ---- Fixtures --------------------------------------------------------------

def _task() -> dict:
    """A minimal Task dict; the rubric doesn't actually use most fields."""
    return {
        "task_id": "test_task",
        "category": "peft-loops",
        "prompt": "test prompt",
        "rubric": {"type": "programmatic-6pt-ladder", "max_score": 6,
                   "criteria": []},
        "reference_solution": "",
    }


PERFECT_STATIC_SRC = """\
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

tokenizer = AutoTokenizer.from_pretrained('m')
model = AutoModelForCausalLM.from_pretrained('m')
lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                  target_modules=['q_proj', 'v_proj'],
                  bias='none', task_type='CAUSAL_LM')
args = SFTConfig(output_dir='./out')
trainer = SFTTrainer(model=model, args=args, peft_config=lora)
trainer.train()
PeftModel.from_pretrained(model, './out')
"""


# ---- Rung 1: parses --------------------------------------------------------

def test_rung1_syntax_error_scores_zero_overall():
    """A non-parsing source produces a 0/6 result with downstream rungs
    explicitly marked as skipped."""
    result = peft_loop.score_peft_loop(_task(), "def f(:\n    pass\n")
    assert result["score"] == 0
    assert result["max_score"] == 6
    assert "parses=0" in result["rationale"]
    # All downstream rungs explicitly skipped (not silently zeroed).
    for rid in ("imports", "lora_config", "trainer_train_called",
                "runs_to_completion", "adapter_loads"):
        assert f"{rid}=0" in result["rationale"]
        assert "skipped" in result["rationale"]


def test_rung1_parses_when_valid_python():
    """Valid Python parses successfully even if other rungs miss."""
    result = peft_loop.score_peft_loop(_task(), "x = 1\n")
    assert "parses=1" in result["rationale"]


# ---- Rung 2: imports -------------------------------------------------------

def test_rung2_missing_imports_scores_zero():
    src = "from peft import LoraConfig\nLoraConfig(r=8)\n"
    result = peft_loop.score_peft_loop(_task(), src)
    assert "imports=0" in result["rationale"]
    assert "missing required imports" in result["rationale"]


def test_rung2_all_three_imports_present():
    src = "import peft\nimport transformers\nimport trl\n"
    result = peft_loop.score_peft_loop(_task(), src)
    assert "imports=1" in result["rationale"]


def test_rung2_from_imports_count():
    src = ("from peft import LoraConfig\n"
           "from transformers import Trainer\n"
           "from trl import SFTTrainer\n")
    result = peft_loop.score_peft_loop(_task(), src)
    assert "imports=1" in result["rationale"]


# ---- Rung 3: LoraConfig ----------------------------------------------------

def test_rung3_no_lora_config_scores_zero():
    src = "import peft\nimport transformers\nimport trl\nx = 1\n"
    result = peft_loop.score_peft_loop(_task(), src)
    assert "lora_config=0" in result["rationale"]


def test_rung3_direct_lora_config_call():
    src = ("import peft\nimport transformers\nimport trl\n"
           "from peft import LoraConfig\n"
           "lora = LoraConfig(r=8)\n")
    result = peft_loop.score_peft_loop(_task(), src)
    assert "lora_config=1" in result["rationale"]


def test_rung3_attribute_lora_config_call():
    src = ("import peft\nimport transformers\nimport trl\n"
           "lora = peft.LoraConfig(r=8)\n")
    result = peft_loop.score_peft_loop(_task(), src)
    assert "lora_config=1" in result["rationale"]


# ---- Rung 4: Trainer.train() ----------------------------------------------

def test_rung4_no_train_call_scores_zero():
    src = ("import peft\nimport transformers\nimport trl\n"
           "from peft import LoraConfig\n"
           "lora = LoraConfig(r=8)\n")
    result = peft_loop.score_peft_loop(_task(), src)
    assert "trainer_train_called=0" in result["rationale"]


def test_rung4_named_trainer_train_call():
    src = ("import peft\nimport transformers\nimport trl\n"
           "from trl import SFTTrainer\n"
           "from peft import LoraConfig\n"
           "lora = LoraConfig(r=8)\n"
           "trainer = SFTTrainer()\n"
           "trainer.train()\n")
    result = peft_loop.score_peft_loop(_task(), src)
    assert "trainer_train_called=1" in result["rationale"]


def test_rung4_inline_trainer_train_call():
    src = ("import peft\nimport transformers\nimport trl\n"
           "from transformers import Trainer\n"
           "from peft import LoraConfig\n"
           "lora = LoraConfig(r=8)\n"
           "Trainer().train()\n")
    result = peft_loop.score_peft_loop(_task(), src)
    assert "trainer_train_called=1" in result["rationale"]


def test_rung4_unrelated_train_call_does_not_count():
    """``model.train()`` (eval->train mode) is not a Trainer.train() call."""
    src = ("import peft\nimport transformers\nimport trl\n"
           "from peft import LoraConfig\n"
           "lora = LoraConfig(r=8)\n"
           "model = None\n"
           "model.train()\n")
    result = peft_loop.score_peft_loop(_task(), src)
    assert "trainer_train_called=0" in result["rationale"]


# ---- Rungs 5-6: sandbox vs. static -----------------------------------------

def test_static_mode_skips_rungs_5_and_6():
    result = peft_loop.score_peft_loop(_task(), PERFECT_STATIC_SRC,
                                        sandbox=False)
    assert result["score"] == 4
    assert "runs_to_completion=0" in result["rationale"]
    assert "adapter_loads=0" in result["rationale"]
    assert "static-analysis-mode" in result["rationale"]


def test_sandbox_mode_skipped_when_torch_unavailable():
    """When the subprocess probe reports torch is missing, rungs 5-6
    score zero with the requires-torch-runtime rationale - rungs 1-4
    score as normal."""
    fake_verdict = {"status": "skipped-requires-torch-runtime",
                    "missing": "ImportError: torch"}
    with patch.object(peft_loop, "_sandbox_probe", return_value=fake_verdict):
        result = peft_loop.score_peft_loop(_task(), PERFECT_STATIC_SRC,
                                            sandbox=True)
    assert result["score"] == 4  # rungs 1-4 only
    assert "requires-torch-runtime" in result["rationale"]
    # Sandbox skip is distinct from the static-mode skip.
    assert "static-analysis-mode" not in result["rationale"]


def test_sandbox_mode_full_six_when_probe_ok():
    fake_verdict = {"status": "ok", "adapter_ok": True}
    with patch.object(peft_loop, "_sandbox_probe", return_value=fake_verdict):
        result = peft_loop.score_peft_loop(_task(), PERFECT_STATIC_SRC,
                                            sandbox=True)
    assert result["score"] == 6
    assert "runs_to_completion=1" in result["rationale"]
    assert "adapter_loads=1" in result["rationale"]


def test_sandbox_mode_ok_but_missing_adapter_file():
    """Probe ran, but adapter_config.json wasn't written. Rung 5 scores 1,
    rung 6 scores 0."""
    fake_verdict = {"status": "ok", "adapter_ok": False}
    with patch.object(peft_loop, "_sandbox_probe", return_value=fake_verdict):
        result = peft_loop.score_peft_loop(_task(), PERFECT_STATIC_SRC,
                                            sandbox=True)
    assert result["score"] == 5
    assert "runs_to_completion=1" in result["rationale"]
    assert "adapter_loads=0" in result["rationale"]


def test_sandbox_mode_probe_failure_scores_zero_for_5_and_6():
    fake_verdict = {"status": "probe-failed",
                    "error": "RuntimeError: kaboom"}
    with patch.object(peft_loop, "_sandbox_probe", return_value=fake_verdict):
        result = peft_loop.score_peft_loop(_task(), PERFECT_STATIC_SRC,
                                            sandbox=True)
    assert result["score"] == 4
    assert "sandbox probe failed: probe-failed" in result["rationale"]


def test_sandbox_probe_handles_subprocess_timeout():
    """The probe function itself - not just the scorer - handles a
    TimeoutExpired by emitting a structured verdict."""
    from subprocess import TimeoutExpired

    def _raise_timeout(*_a, **_kw):
        raise TimeoutExpired(cmd="x", timeout=1)

    with patch.object(peft_loop.subprocess, "run", side_effect=_raise_timeout):
        verdict = peft_loop._sandbox_probe(timeout=1)
    assert verdict["status"] == "timeout"
    assert verdict["timeout_seconds"] == 1


def test_sandbox_probe_handles_malformed_subprocess_output():
    fake = MagicMock(returncode=0, stdout="not json at all", stderr="")
    with patch.object(peft_loop.subprocess, "run", return_value=fake):
        verdict = peft_loop._sandbox_probe(timeout=1)
    assert verdict["status"] == "malformed-output"
    assert "not json" in verdict["raw_stdout"]


# ---- Reference-corpus integration ------------------------------------------

def test_every_reference_solution_scores_four_in_static_mode():
    """The ladder is the contract between L5's corpus and L6's runner -
    if any reference fails to hit 4/4 on rungs 1-4, either the corpus
    is broken or the runner is over-strict."""
    grouped = load_tasks(default_tasks_dir())
    for task in grouped["peft-loops"]:
        result = peft_loop.score_peft_loop(task, task["reference_solution"],
                                            sandbox=False)
        assert result["score"] == 4, (
            f"reference for {task['task_id']!r} scored "
            f"{result['score']}/6 in static mode\n"
            f"{result['rationale']}"
        )


def test_every_reference_solution_scores_six_in_mocked_sandbox():
    """With a mocked-ok sandbox probe, every reference must score 6/6."""
    grouped = load_tasks(default_tasks_dir())
    fake_verdict = {"status": "ok", "adapter_ok": True}
    with patch.object(peft_loop, "_sandbox_probe", return_value=fake_verdict):
        for task in grouped["peft-loops"]:
            result = peft_loop.score_peft_loop(task, task["reference_solution"],
                                                sandbox=True)
            assert result["score"] == 6, (
                f"reference for {task['task_id']!r} scored "
                f"{result['score']}/6 in mocked-sandbox mode\n"
                f"{result['rationale']}"
            )
