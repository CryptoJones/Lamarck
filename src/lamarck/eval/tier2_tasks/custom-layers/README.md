# Tier 2 — custom-layers tasks

10 hand-curated PyTorch component-implementation scenarios. Each
tasks G_N with producing a working class or function for a custom
ML primitive (normalization, activation, attention, optimizer,
positional encoding, regularizer, etc.).

## Why 10?

The eval-suite v1 spec locks Tier 2's category 2c (`custom-layers`)
at 10 tasks × 1 point per task = 10 raw points of the 200-point
Tier 2 total. Changing the count breaks generation-to-generation
score comparability and is forbidden until v2.

## Scoring — binary unit-test rubric

Per `docs/eval-suite-v1.md`, each task is graded by whether the
candidate's code passes the task's unit tests. The L8 runner:

1. Writes the candidate's code to a tmpdir as `solution.py`.
2. Writes the task's `unit_tests` field to `test_solution.py`.
3. Runs `pytest -q` in the tmpdir.
4. Returns 1 if exit code 0, else 0.

When torch is unavailable in the runner subprocess, the runner
gracefully degrades to `"skipped: requires-torch-runtime"`
(same pattern as L6's PEFT/LoRA sandbox probe).

## Task file schema

```json
{
  "task_id": "cl_NNN_short_slug",
  "category": "custom-layers",
  "prompt": "...what the model is asked to implement...",
  "rubric": {
    "type": "unit-tests-binary",
    "max_score": 1,
    "criteria": [{"id": "tests_pass", "points": 1, "description": "..."}]
  },
  "reference_solution": "<python source that passes unit_tests>",
  "unit_tests": "<pytest source: imports from `solution`, defines def test_*()>"
}
```

## Scenario coverage

| ID                              | Component                       | Tested invariants                                         |
|---------------------------------|---------------------------------|-----------------------------------------------------------|
| cl_001_rmsnorm                  | RMSNorm                         | shape, RMS=1 normalization, learnable gain, eps safety    |
| cl_002_swiglu                   | SwiGLU FFN block                | shape, three named no-bias linears, matches explicit math |
| cl_003_rope                     | Rotary positional embeddings    | shape, pos=0 identity, per-pair norm preservation         |
| cl_004_info_nce                 | InfoNCE contrastive loss        | scalar, perfect-alignment low loss, ln(N) for random      |
| cl_005_sdpa                     | Scaled dot-product attention    | shape, matches F.scaled_dot_product_attention, causal mask|
| cl_006_sgd_momentum             | SGDMomentum optimizer           | loss reduction, momentum buffer math, skip-None-grad      |
| cl_007_grad_checkpoint          | Gradient checkpointing wrapper  | forward equivalence, backward correctness, grads match    |
| cl_008_sinusoidal_pe            | Sinusoidal positional encoding  | pe buffer shape, non-learnable, position-0 sin=0/cos=1    |
| cl_009_prenorm                  | PreNorm transformer wrapper     | shape, attrs, residual identity, norm-before-sublayer     |
| cl_010_variational_dropout      | Variational locked dropout      | eval identity, mask locked across seq, p=0 identity       |
