# Tier 2 — curriculum-design tasks

10 hand-curated curriculum-design scenarios. Each task gives the
candidate a target capability and asks them to produce a structured
multi-stage curriculum: each stage names a dataset + a measurable
success criterion before advancing to the next stage.

## Why 10?

The eval-suite v1 spec locks Tier 2's category 2b (`curriculum-design`)
at 10 tasks × 4 points per task = 40 raw points of the 200-point
Tier 2 total. Changing the count breaks generation-to-generation
score comparability and is forbidden until v2.

## Scoring — 4-axis LLM-judge rubric

Per `docs/eval-suite-v1.md`, each axis is 1 binary point; the L11
LLM-judge runner grades on:

1. **Completeness** — covers prereqs, target capability, evaluation.
2. **Ordering** — prereqs come before target; harder stages after
   easier.
3. **Specificity** — names concrete datasets / benchmarks / metrics
   rather than vague hand-waves.
4. **Plausibility** — the curriculum, if executed, would plausibly
   produce the target capability.

## Task file schema

```json
{
  "task_id": "cd_NNN_short_slug",
  "category": "curriculum-design",
  "prompt": "<target capability description>",
  "rubric": {
    "type": "llm-judge-4axis",
    "max_score": 4,
    "criteria": [
      {"id": "completeness", "points": 1, "description": "..."},
      {"id": "ordering",     "points": 1, "description": "..."},
      {"id": "specificity",  "points": 1, "description": "..."},
      {"id": "plausibility", "points": 1, "description": "..."}
    ]
  },
  "reference_solution": "<multi-stage curriculum: each stage names a dataset + a success criterion>"
}
```

## Scenario coverage

| ID                                | Target capability                                                |
|-----------------------------------|------------------------------------------------------------------|
| cd_001_json_mode                  | JSON-mode under 16k context-window pressure                      |
| cd_002_secure_code                | Refuse insecure code + suggest secure equivalent                 |
| cd_003_markdown_format            | Follow Markdown formatting instructions (lists/tables/code)      |
| cd_004_multilingual_consistency   | Stay in user's language under technical-topic load (6 langs)     |
| cd_005_function_calling           | Structured tool-calling from a menu of available tools           |
| cd_006_self_check_arithmetic      | Self-check arithmetic via independent re-derivation              |
| cd_007_jailbreak_refusal          | Robust jailbreak refusal without over-cautious behavior          |
| cd_008_tests_first                | TDD: tests before implementation, with edge-case coverage        |
| cd_009_scratchpad_math            | Scratchpad-style CoT for elementary arithmetic word problems     |
| cd_010_react_agent                | ReAct multi-turn tool-use loops with sensible stopping behavior  |
