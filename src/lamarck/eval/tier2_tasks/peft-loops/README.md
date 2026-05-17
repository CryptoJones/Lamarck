# Tier 2 — PEFT/LoRA loop tasks

15 hand-curated fine-tuning scenarios. Each one tasks G_N with
producing a complete training script for a different combination
of base model, dataset shape, and LoRA / training config.

## Why 15?

The eval-suite v1 spec locks Tier 2's category 2a (`peft-loops`)
at 15 tasks × 6 points per task = 90 raw points of the 200-point
Tier 2 total. Changing the count breaks generation-to-generation
score comparability and is forbidden until v2.

## Scoring — the 6-point programmatic ladder

Per `docs/eval-suite-v1.md`, every task is graded against the same
ladder (each rung is binary, partial credit accumulates):

| Rung | Check                                             | Implemented in |
|------|---------------------------------------------------|----------------|
| 1    | Code parses with `ast.parse`                      | L6 runner      |
| 2    | Imports `peft`, `transformers`, and `trl`         | L6 runner      |
| 3    | Instantiates a `LoraConfig`                       | L6 runner      |
| 4    | Calls `Trainer.train()` or `SFTTrainer.train()`   | L6 runner      |
| 5    | Runs 1 epoch × 10 synthetic samples in a sandbox  | L6 runner      |
| 6    | Resulting adapter loads via `PeftModel.from_pretrained` | L6 runner |

The corresponding rubric `criteria[].id` strings are stable:
`parses`, `imports`, `lora_config`, `trainer_train_called`,
`runs_to_completion`, `adapter_loads`. Tests in
`tests/test_tier2_peft_corpus.py` lock both the count and the
ladder id-set.

## Task file schema

```json
{
  "task_id": "peft_NNN_short_slug",
  "category": "peft-loops",
  "prompt": "...what the model is asked to do...",
  "rubric": {
    "type": "programmatic-6pt-ladder",
    "max_score": 6,
    "criteria": [
      {"id": "parses",               "points": 1, "description": "..."},
      {"id": "imports",              "points": 1, "description": "..."},
      {"id": "lora_config",          "points": 1, "description": "..."},
      {"id": "trainer_train_called", "points": 1, "description": "..."},
      {"id": "runs_to_completion",   "points": 1, "description": "..."},
      {"id": "adapter_loads",        "points": 1, "description": "..."}
    ]
  },
  "reference_solution": "<python source that hits all 6 rungs>"
}
```

## Scenario coverage

The 15 tasks are intentionally diverse so we measure breadth, not
"does the model regurgitate one template":

| ID                            | Base                                      | Twist                                  |
|-------------------------------|-------------------------------------------|----------------------------------------|
| peft_001_llama8b_alpaca       | Llama-3.1-8B-Instruct                     | classic Alpaca SFT                     |
| peft_002_phi3_qlora           | Phi-3-mini-4k-instruct                    | 4-bit NF4 QLoRA, all-linear targets    |
| peft_003_llama70b_4bit        | Llama-3.1-70B-Instruct                    | 70B QLoRA + gradient checkpointing     |
| peft_004_mistral_chat_template| Mistral-7B-Instruct-v0.3                  | tokenizer chat template                |
| peft_005_gemma2b_qa           | gemma-2-2b                                | SQuAD-style QA                         |
| peft_006_cpt_raw_text         | TinyLlama-1.1B-Chat                       | continued pretraining via Trainer (no SFTTrainer) |
| peft_007_codellama_repair     | CodeLlama-7b-Instruct                     | code-repair, all-linear, grad accum    |
| peft_008_qwen_toolcalls       | Qwen2-7B-Instruct                         | 4-bit QLoRA tool-use traces            |
| peft_009_tinyllama_conv       | TinyLlama-1.1B-Chat                       | tiny rank, fast smoke run              |
| peft_010_mistral_long_ctx     | Mistral-7B-v0.3                           | 4096-token packing + paged_adamw_8bit  |
| peft_011_llama3b_grad_accum   | Llama-3.2-3B-Instruct                     | warmup + cosine LR schedule            |
| peft_012_deepseek_coder       | deepseek-coder-6.7b-instruct              | trust_remote_code, function completion |
| peft_013_falcon_dialogue      | falcon-7b-instruct                        | combined `query_key_value` target      |
| peft_014_neftune              | Llama-2-7b-hf                             | NEFTune noise alpha                    |
| peft_015_stablelm_paged       | stablelm-2-1_6b                           | paged_adamw_32bit + weight_decay       |
