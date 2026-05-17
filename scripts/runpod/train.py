#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
"""Lamarck — QLoRA fine-tune of DeepSeek-R1-Distill-Llama-70B.

Reads:  $LAMARCK_DATA_DIR/training.jsonl   (one {"prompt","completion"} JSON per line)
Writes: $LAMARCK_ADAPTER_DIR/              (LoRA adapter only — base weights unchanged)

Generation-aware: $LAMARCK_GENERATION=1 names the output dir
`g1_adapter/` by default; G2 trains on top of G1 and writes
`g2_adapter/`. Each generation is a separate adapter; the lineage
is recorded in `adapter_metadata.json` so `serve.sh` knows how to
stack them.

Target hardware: single A100 80GB (RunPod). See pod-setup.sh for
dependency install.

Adapted from Dave's train_dave.py with three substantive changes:
  1. Base model: meta-llama/Llama-3.3-70B-Instruct → DeepSeek-R1-
     Distill-Llama-70B.
  2. SYSTEM_PROMPT replaced with the Lamarck curriculum-design
     framing (G_N teaches G_{N+1}, the grandchild is what counts).
  3. Output directory is generation-aware; metadata captures the
     parent adapter (if any) so multi-gen stacking is traceable.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


BASE_MODEL = os.environ.get(
    "LAMARCK_BASE_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
)
GENERATION = int(os.environ.get("LAMARCK_GENERATION", "1"))
DATA_DIR = Path(os.environ.get("LAMARCK_DATA_DIR", str(Path(__file__).parents[2] / "curricula")))
ADAPTER_DIR = Path(
    os.environ.get(
        "LAMARCK_ADAPTER_DIR",
        str(Path(__file__).parents[2] / f"adapters/g{GENERATION}"),
    )
)
# Optional: stack on a parent adapter (G2 trains on G1's weights).
PARENT_ADAPTER = os.environ.get("LAMARCK_PARENT_ADAPTER") or None

TRAIN_FILE = DATA_DIR / "training.jsonl"

SYSTEM_PROMPT = (
    "You are a teacher. Your job is to design training curricula "
    "that produce better successor models. Each example you generate "
    "should be a (problem, solution) pair where the solution "
    "demonstrates clear reasoning about ML task design. Your students "
    "will themselves teach. Optimize for the grandchild, not the child."
)


def require_data_file() -> None:
    if not TRAIN_FILE.exists():
        sys.exit(
            f"ERROR: training file not found: {TRAIN_FILE}\n"
            f"For G1: bootstrap a curriculum from G0 first.\n"
            f"For G{GENERATION}: ensure the previous generation wrote "
            f"its curriculum to {TRAIN_FILE}."
        )
    if TRAIN_FILE.stat().st_size == 0:
        sys.exit(f"ERROR: training file is empty: {TRAIN_FILE}")
    n_lines = sum(1 for _ in TRAIN_FILE.open())
    print(f"Training file: {TRAIN_FILE} ({n_lines} examples)")


def to_chatml(example: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["completion"]},
        ]
    }


def main() -> int:
    require_data_file()
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generation:     G{GENERATION}")
    print(f"Base model:     {BASE_MODEL}")
    print(f"Parent adapter: {PARENT_ADAPTER or '(none)'}")
    print(f"Adapter out:    {ADAPTER_DIR}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # Optional: load a parent adapter as starting weights for this generation.
    if PARENT_ADAPTER:
        from peft import PeftModel
        print(f"Loading parent adapter from {PARENT_ADAPTER}…")
        model = PeftModel.from_pretrained(model, PARENT_ADAPTER, is_trainable=True)

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    dataset = load_dataset("json", data_files=str(TRAIN_FILE))["train"]
    dataset = dataset.map(to_chatml)

    cfg = SFTConfig(
        output_dir=str(ADAPTER_DIR),
        num_train_epochs=int(os.environ.get("LAMARCK_EPOCHS", "2")),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        report_to=[],
        max_seq_length=4096,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        peft_config=None if PARENT_ADAPTER else lora,
        args=cfg,
    )

    trainer.train()
    trainer.save_model(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))

    # --- adapter metadata --------------------------------------------------
    # Captures the lineage so serve.sh + future generations know what
    # adapter stack to load. The grandchild story collapses without
    # this provenance.
    meta = {
        "generation": GENERATION,
        "base_model": BASE_MODEL,
        "parent_adapter": PARENT_ADAPTER,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_examples": sum(1 for _ in TRAIN_FILE.open()),
        "epochs": cfg.num_train_epochs,
    }
    (ADAPTER_DIR / "adapter_metadata.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n"
    )
    print(f"\nSaved adapter + metadata to {ADAPTER_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
