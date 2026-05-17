# Eval suite v1 — readiness review

**Date**: 2026-05-17
**Reviewer**: Claude Opus 4.7 (1M context), reviewing the L1-L19
deliverables on Aaron's behalf.

**Verdict**: 🟡 **READY FOR DRY RUNS; NOT READY FOR G0 LOCK.**

The framework is implementation-ready and end-to-end testable in
mock mode. Two gating prerequisites remain before locking the v1
spec at the start of a real G0 run.

## What's shipped (L1-L19)

| L  | Deliverable                                          | Status |
|----|------------------------------------------------------|--------|
| L1 | Eval-suite v1 spec doc + API stub                    | ✅ shipped |
| L2 | Tier 1 external benchmark wrapper                    | ✅ shipped |
| L3 | Tier 1 mocked integration tests                      | ✅ shipped |
| L4 | Tier 2 framework + dispatcher + RubricRunner Protocol| ✅ shipped |
| L5 | Tier 2 peft-loops corpus (15 tasks)                  | ✅ shipped |
| L6 | Tier 2 peft-loops rubric runner (6-pt ladder)        | ✅ shipped |
| L7 | Tier 2 custom-layers corpus (10 tasks)               | ✅ shipped |
| L8 | Tier 2 custom-layers rubric runner (unit-test)       | ✅ shipped |
| L9 | Tier 2 curriculum-design corpus (10 tasks)           | ✅ shipped |
| L10| Tier 2 diagnostics corpus (15 tasks)                 | ✅ shipped |
| L11| Tier 2 LLM-judge rubric runner (4-axis)              | ✅ shipped |
| L12| Tier 2 end-to-end integration tests                  | ✅ shipped |
| L13| Tier 3 harness skeleton + locked recipe              | ✅ shipped |
| L14| Tier 3 100-problem held-out set (deterministic)      | ✅ shipped |
| L15| Tier 3 JSON-mode pass-rate scorer                    | ✅ shipped |
| L16| Tier 3 end-to-end integration tests                  | ✅ shipped |
| L17| Aggregation CLI (`lamarck.eval.aggregate_cli`)       | ✅ shipped |
| L18| Run-all orchestrator (`scripts/eval/run-all.sh`)     | ✅ shipped |
| L19| CI on GitHub Actions + Codeberg Woodpecker           | ✅ shipped |

Total test count after L19: **248 passing** (Tier 1: 21, Tier 2: 53,
Tier 3: 33, scorer: 24, judge: 21, framework: 17, holdout corpus:
10, CLI: 18, run-all: 11, plus the rest).

## What's locked under v1

- The three tier weights: 0.20 / 0.50 / 0.30 (sum 1.0).
- Tier 1 benchmarks: MMLU-Pro (CS/eng/math/physics subsets only),
  HumanEval+, GSM8K. Equal weight inside Tier 1.
- Tier 2 task counts: 15 + 10 + 10 + 15 = 50 tasks, 200 raw points
  normalized to 100.
- Tier 2 rubric types per category: 6-pt programmatic ladder
  (peft-loops), binary unit-test (custom-layers), 4-axis LLM-judge
  (curriculum-design + diagnostics).
- Tier 3 student model: `meta-llama/Llama-3.2-1B-Instruct`.
- Tier 3 fine-tune recipe (frozen): 1 epoch, batch 4, LR 2e-4,
  LoRA rank 16, alpha 32, dropout 0.05, seed 42.
- Tier 3 held-out: 100 problems, generator seed `0x42ACAC1A`,
  byte-identical regenerable.
- Aggregation formula: `final = 0.20*T1 + 0.50*T2 + 0.30*T3`.

All of the above have locked-invariant tests that fail loudly if
any number drifts.

## Gating prerequisites — what blocks G0 lock

### 🚫 (1) Real Tier 3 fine-tune + inference

The harness ships `FineTuneFn` and `InferenceFn` Protocols with
GPU-requiring default stubs. CI runs against mocked implementations
and that's correct — but G0 needs the real implementations on a
GPU pod. This is L21-class work (Aaron-approved):

- Real `fine_tune_fn`: load `meta-llama/Llama-3.2-1B-Instruct`,
  apply the locked LoRA config, train 1 epoch on the curriculum
  JSONL, save the adapter to `output_dir`, return the path.
- Real `inference_fn`: load the adapter, run inference on the
  prompt, return the completion text.

Without these, Tier 3 scores 0/100 on a real run. The weighted
contribution (0.30 * 0) drops the final ceiling to 70.

### 🚫 (2) Judge model identity not locked

L11 ships a runner that calls an OpenAI-compatible
`/chat/completions` endpoint with `judge_model_id='gpt-4o-mini'`
as the parameter default. **v1 has not committed to which model
grades**. Different judges produce different scores; for the
recursive optimization to be stable, the judge identity must be
fixed before G0.

Options to lock before G0:

- Self-judging via a held-out non-Lamarck base (e.g. Mistral-7B or
  Llama-3.1-70B-Instruct served alongside the model under test).
  Pro: no external dep. Con: same model family may bias scores.
- External judge (Claude Opus / GPT-4o / DeepSeek-V3) via API.
  Pro: independent. Con: dependency on a third-party endpoint
  whose scoring may drift across G_N runs as the third-party
  model is updated.

**Recommendation**: pin Llama-3.1-70B-Instruct as the judge,
served on the same vLLM endpoint as the model under test (separate
adapter slot). This keeps the suite self-contained and is the same
posture as the L19 design implicitly assumes (the judge IS reachable
at `base_url`).

### ⚠️ (3) Calibration baseline not run

The suite is locked, but no public model has been scored against it
yet. Without a reference run, an absolute score like "G2 = 67" is
uninterpretable. v1 needs at least one calibration pass against
Llama-3.1-70B-Instruct (the same family Lamarck-G0 inherits from)
so subsequent G_N scores have a baseline to delta against.

**This is the cheapest gating item.** A single pod run against the
70B base produces the calibration number; everything else in the
suite is implementation-ready to support it.

## What the suite explicitly does NOT measure

(Per `docs/eval-suite-v1.md`'s safety boundary, restated here for
the readiness audit.)

- Agentic behavior: multi-step tool-use, planning, real-world task
  execution outside ML primitives.
- Persuasion / social manipulation.
- Self-exfiltration / environment-probing.
- Non-ML reasoning depth: philosophy, ethics, legal/medical
  reasoning, creative writing.

The recursive optimization pressure is shaped by what the eval
suite rewards. v1's narrow ML-engineering focus is deliberate
and is the safety lever.

## Sign-off checklist (when ready to lock G0)

- [ ] Real `FineTuneFn` + `InferenceFn` implementations land in
      `src/lamarck/eval/tier3_real_runner.py` (or similar),
      validated against the held-out set on a CPU stand-in model.
- [ ] Judge model identity decided and frozen via a constant in
      `lamarck.eval.rubrics.llm_judge` (e.g. `JUDGE_MODEL_ID =
      "meta-llama/Llama-3.1-70B-Instruct"`).
- [ ] Calibration run completed against Llama-3.1-70B-Instruct
      base; results recorded in
      `docs/eval-suite-v1-calibration.md`.
- [ ] Full-suite mock run in CI still produces final_score 57.0
      (regression sentry; L19 enforces this).
- [ ] `docs/eval-suite-v1-errata.md` is empty.

When all five are checked, v1 is ready to lock at the first G0
training run, and the suite-version freeze takes effect.

## Why not "ready, no obvious polish"

The L1-L19 work covers everything specifiable from the v1 spec
on a CPU-only CI host. The three gating items above are real
gaps, not polish — each requires a substantive decision or a GPU
pod run. Calling v1 "ready, no polish needed" would skip those
decisions and pretend the score scale is interpretable when it
isn't.

The honest verdict: **framework ready; experiment not ready until
the gating items above clear.**

---

Proudly Made in Nebraska. Go Big Red! 🌽 https://xkcd.com/2347/
