# Tier 2 — diagnostics tasks

15 hand-curated ML-training failure scenarios. Each task presents
the candidate with a broken script + a traceback (or a "no error
but training is broken" failure mode) and asks for a root-cause
identification + a fix.

## Why 15?

The eval-suite v1 spec locks Tier 2's category 2d (`diagnostics`)
at 15 tasks × 4 points per task = 60 raw points of the 200-point
Tier 2 total. Changing the count breaks generation-to-generation
score comparability and is forbidden until v2.

## Scoring — 4-axis LLM-judge rubric

Per `docs/eval-suite-v1.md`, each axis is 1 binary point; the L11
LLM-judge runner grades on:

1. **correct_root_cause** — identifies the actual root cause, not a
   downstream symptom.
2. **viable_fix** — suggests a fix that would actually resolve it.
3. **no_new_bug** — doesn't introduce a new bug in the fix.
4. **mechanistic_explanation** — explains why the bug happened.

## Task file schema

```json
{
  "task_id": "diag_NNN_short_slug",
  "category": "diagnostics",
  "prompt": "<broken script in a fenced block + traceback / failure description>",
  "rubric": {
    "type": "llm-judge-4axis",
    "max_score": 4,
    "criteria": [
      {"id": "correct_root_cause",      "points": 1, "description": "..."},
      {"id": "viable_fix",              "points": 1, "description": "..."},
      {"id": "no_new_bug",              "points": 1, "description": "..."},
      {"id": "mechanistic_explanation", "points": 1, "description": "..."}
    ]
  },
  "reference_solution": "<Root cause: ...\nFix: ...\nMechanism: ...>"
}
```

## Scenario coverage

| ID                              | Failure mode                                                      |
|---------------------------------|-------------------------------------------------------------------|
| diag_001_pad_token              | Llama tokenizer missing pad_token; batched-pad crash              |
| diag_002_mlm_for_causal         | DataCollatorForLanguageModeling(mlm=True) on a causal LM          |
| diag_003_kbit_no_prepare        | QLoRA without prepare_model_for_kbit_training → zero gradients    |
| diag_004_device_map_trainer     | device_map='auto' with HF Trainer → cross-device tensor mismatch  |
| diag_005_oom_grad_accum         | 70B QLoRA OOM; effective-batch confused with per-device batch     |
| diag_006_flash_attn_wrong_gpu   | FlashAttention-2 on a T4 (pre-Ampere) GPU                         |
| diag_007_eval_no_dataset        | evaluation_strategy='steps' without eval_dataset                  |
| diag_008_save_adapter_only      | save_pretrained on un-wrapped model loses the adapter             |
| diag_009_dataset_map_scalar     | Dataset.map() function returning a list instead of a dict         |
| diag_010_nan_loss_lr            | fp16 + high LR + no warmup → NaN loss after ~50 steps             |
| diag_011_target_modules_wrong   | Falcon's combined query_key_value vs Llama's q_proj/v_proj naming |
| diag_012_packing_short_seqs     | packing=True on a 3-unique-sample dataset → entropy-baseline loss |
| diag_013_dtype_mismatch         | fp32 base + bf16 autocast → dtype mismatch on first forward       |
| diag_014_lr_no_warmup           | constant LR schedule, plateaus high, never refines                |
| diag_015_compile_lora           | torch.compile on PEFT-wrapped model → slower than uncompiled      |
