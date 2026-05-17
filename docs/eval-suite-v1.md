# Eval suite — v1 (LOCKED)

The fixed external benchmark Lamarck's generational training is
scored against. **v1 is locked before G0 ever runs.** Changes after
that void the experiment series — any modification ships as v2 with
a fresh generation lineage.

**Why this exists:** G2's score on this suite is G0's reward signal.
If we let G_N influence the suite (by editing it after seeing
results, by retuning weights based on G1's behavior, by anything),
we collapse the grandchild framing. The suite has to be a fixed
external target, not a moving goalpost.

---

## The three tiers

The score G0 sees is a weighted combination of three independent
measurements:

| Tier | What it measures               | Cost / run    | Weight |
|------|--------------------------------|---------------|--------|
| 1    | Sanity — general capability    | ~30 min       | 20%    |
| 2    | ML-engineering capability      | ~30-60 min    | 50%    |
| 3    | Grounded teaching capability   | ~1-2 hours    | 30%    |

Tier 1 catches catastrophic regression. Tier 2 is the actual signal.
Tier 3 is the load-bearing "does G2 actually work" test that's
resistant to gaming because it requires real downstream training.

### Tier 1 — external sanity benchmarks (20%)

**Purpose:** detect if G2 has lost general capability while
optimizing for ML-engineering. If MMLU drops 30 points, G2 isn't
"better at ML" — it's been overfit on the curriculum.

Three off-the-shelf benchmarks via `lm-evaluation-harness`:

- **MMLU-Pro** ([`TIGER-Lab/MMLU-Pro`](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)) —
  use the `computer_science`, `engineering`, `math`, and `physics`
  subsets only (~3500 questions). General-knowledge MMLU-Pro
  subsets are out of scope; we don't care if G2 forgot world
  history.
- **HumanEval+** ([`evalplus/humanevalplus`](https://huggingface.co/datasets/evalplus/humanevalplus)) —
  164 hand-written Python coding problems with augmented test cases
  that catch shallow solutions. Sanity check that G2 can still write
  Python.
- **GSM8K** ([`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k)) —
  1319 grade-school math word problems. Reasoning sanity check.

**Score:** normalized weighted average across the three benchmarks.
Equal weight inside Tier 1 (so each benchmark contributes ~6.67%
to the final score before Tier 1's 20% multiplier).

**Runtime:** ~20-40 minutes on the served vLLM endpoint at
reasonable batch sizes.

**What "passing" looks like:** absolute scores don't matter — we
care about deltas across generations. If G2 ≥ G1 ≥ G0 on Tier 1,
we haven't broken anything obvious. If G2 < G0 by more than 5
percentage points on any single benchmark, that's a red flag.

### Tier 2 — ML-engineering tasks (50%)

**Purpose:** the bulk of the actual signal. ~50 hand-curated tasks
that exercise the capabilities G0/G1/G2 are supposed to be teaching
each other. Locked at suite v1 birth; never edited after.

Four categories, each with a defined rubric:

#### 2a. PEFT/LoRA training loops (15 tasks, ~30% of Tier 2)

Each task: a prompt describing a fine-tuning scenario (base model,
dataset, target adapter). G_N has to write the training code.

Programmatic rubric — 6-point ladder per task:

1. Code parses with `ast.parse`.
2. Imports the right libraries (peft, transformers, trl).
3. Instantiates a `LoraConfig`.
4. Calls `Trainer.train()` or `SFTTrainer.train()`.
5. Runs to completion on a tiny test (1 epoch, 10 synthetic samples).
6. Resulting adapter loads correctly via `PeftModel.from_pretrained`.

Each rung is binary; partial credit accumulates. 15 tasks × 6 pts
= 90 raw points.

#### 2b. Curriculum design (10 tasks, ~20% of Tier 2)

Each task: a target capability (e.g., "train a 7B model to
respond in valid JSON-mode under context-window pressure").
G_N has to output a structured curriculum spec — N stages, each
with dataset description, success criterion.

Rubric (4-point per task, LLM-judge with sampled human review):

1. Completeness: covers prereqs, target capability, evaluation.
2. Ordering: prereqs come before target.
3. Specificity: names datasets / benchmarks; not vague.
4. Plausibility: the curriculum, if executed, would plausibly
   produce the target capability.

10 tasks × 4 pts = 40 raw points.

#### 2c. Custom layers / losses / optimizers (10 tasks, ~10% of Tier 2)

Each task: a specification of a custom PyTorch component (e.g.,
"implement RMSNorm without using torch.nn", "write a custom
contrastive loss with these properties"). G_N produces working code.

Rubric (binary per task): unit tests pass / don't pass. 10 tasks
× 1 pt = 10 raw points.

#### 2d. Diagnostic failures (15 tasks, ~30% of Tier 2)

Each task: a broken training script + stack trace. G_N has to
identify the root cause and suggest a fix.

Rubric (4-point per task, LLM-judge with sampled human review):

1. Identifies the actual root cause (not a downstream symptom).
2. Suggests a fix that would actually resolve it.
3. Doesn't introduce a new bug in the fix.
4. Explains *why* the bug happened (mechanistic understanding).

15 tasks × 4 pts = 60 raw points.

**Tier 2 total:** 90 + 40 + 10 + 60 = 200 raw points. Normalized
to /100 for aggregation.

**Runtime:** ~30-60 minutes depending on rubric implementation
speed. Programmatic rubrics (2a, 2c) are fast; LLM-judge rubrics
(2b, 2d) are slower.

### Tier 3 — grounded teaching capability (30%)

**Purpose:** the test that's hardest to game. G2 has to actually
teach a downstream model a real capability; we then measure the
downstream model.

**Setup, locked at suite v1 birth:**

- **Student model:** [`meta-llama/Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct).
  Small enough to fine-tune quickly (~15 min on A100), large
  enough to actually learn something.
- **Target capability:** valid-JSON-mode output. Held-out test set
  (`tier3_holdout/json_mode_problems.jsonl`) of 100 problems where
  the response must be syntactically valid JSON matching a given
  schema. Pass rate on this test set after fine-tune = Tier 3 score.
- **Fine-tune recipe:** 1 epoch, batch size 4, learning rate 2e-4,
  LoRA rank 16. **Fixed.** G_N does not get to tune the recipe;
  G_N only gets to design the curriculum (training data).

**Procedure per generation:**

1. G_N produces a curriculum: a JSONL of (prompt, completion) pairs
   meant to teach JSON-mode output to a Llama-3.2-1B-Instruct base.
2. We fine-tune Llama-3.2-1B-Instruct on that curriculum with the
   fixed recipe.
3. The resulting tiny model is tested against the 100-problem
   held-out set. Pass rate is the Tier 3 score (0-100).

**Why this is gaming-resistant:**

- G_N never sees the held-out set.
- The fine-tune recipe is fixed; G_N can't "win" by recommending
  a magic learning rate.
- The tiny model can't be in-context-prompted; it has to actually
  learn from the curriculum.
- "Talking convincingly about training" doesn't help; only
  curricula that produce a downstream-capable model score well.

**Runtime:** ~60-90 minutes per generation (curriculum generation,
fine-tune, evaluation). Most expensive component of the suite.

---

## Aggregation

```
final_score = 0.20 * tier1_normalized + 0.50 * tier2_normalized + 0.30 * tier3_raw
```

Each tier is normalized to 0-100 first; the weighted sum is the
final 0-100 score G0 sees as its reward signal.

If any tier fails to run (e.g., Tier 3's GPU pod isn't available),
the aggregation produces a partial score AND a `partial: true` flag
in the output JSON. A partial score is research data, not a result
— it doesn't get used for generation-to-generation comparison.

---

## Versioning policy

- **v1 is locked at the start of the first G0 training run.** No
  edits to task lists, rubrics, weights, or component models
  after that point.
- **Bugs in v1 don't get fixed in v1.** If a Tier 2 task turns
  out to be broken (e.g., the reference solution is wrong), we
  note it in `docs/eval-suite-v1-errata.md` and live with the
  noise. Editing v1 mid-experiment is forbidden.
- **v2 starts a new generation lineage.** If we want to revise
  the suite, we ship `docs/eval-suite-v2.md`, restart from a fresh
  G0, and the v1 numbers are not comparable to v2 numbers.

This is the same discipline as freezing a test set: the moment
you let the eval suite respond to model behavior, the model's
score stops measuring anything external.

---

## What we explicitly DON'T measure

The eval suite is **ML-task capability only.** It does not score
G_N on:

- **Agentic behavior** — multi-step tool-use, planning, real-world
  task execution.
- **Persuasion or social manipulation** — debate, rhetorical
  effectiveness, behavior change in humans.
- **Self-exfiltration or environment-probing** — attempts to
  discover its own deployment context, read files outside the
  prompt, escape sandboxes.
- **Non-ML reasoning depth** — philosophy, ethics, creative writing,
  legal reasoning, medical reasoning.

This is a safety boundary, not an oversight. The recursive
optimization pressure is shaped by what the eval suite rewards.
By making the reward function entirely about ML engineering, we
keep the optimization pressure inside that envelope. Any
capability that emerges outside this envelope does so without
reinforcement from the recursion.

If a future generation develops capabilities outside this envelope
that the eval suite incidentally measures, that's a research
finding to document — not a feature to optimize for. The suite
stays locked.

---

## Open questions for v2 (if we ever ship one)

These don't change v1. They're notes for whoever designs v2 (or
forks v1) — captured here so the questions aren't lost.

1. **Should Tier 3's target capability rotate per generation?**
   v1 uses JSON-mode for all generations. That's intentional
   (lets us compare apples to apples across G1 → G2 → G3). But
   it also means a curriculum that's overfit to JSON-mode will
   score artificially well. v2 might rotate the target capability.

2. **Is the 20/50/30 split right?** Tier 3 is the gaming-resistant
   one; arguably it should be weighted more. v1 picks 30% because
   Tier 3's noise (one run per generation, ~100-problem held-out)
   is higher than Tier 2's. A future version with bigger held-out
   sets might justify weighting Tier 3 higher.

3. **Should there be a Tier 4 for cross-generation comparison?**
   E.g., G2 grades curricula by G0 and G1; do we see preference
   for one? v1 doesn't try to measure this; v2 might.

4. **What about negative scoring?** v1 has no penalty for
   refusing to answer. A model that refuses a Tier 2 task and a
   model that fails it both score 0 on that task. Should refusal
   score differently? v2 might want to.

---

Proudly Made in Nebraska. Go Big Red! 🌽 https://xkcd.com/1838/
