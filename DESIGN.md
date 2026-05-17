# Lamarck — design

A scaffold for studying Lamarckian generational ML at the 70B
scale. Three generations: G0 (the base model — DeepSeek-R1-Distill-
Llama-70B), G1 (the child, fine-tuned by G0), G2 (the grandchild,
fine-tuned by G1). G0's reward signal is **G2's** evaluation score
on a fixed external suite, not G1's loss or G0's self-judgement.

## Why this exists

Naive "self-improving model" attempts hit a degenerate objective:
the model optimizes to make its own next training run better, which
collapses into optimizing its own training loss against its own
data. Nothing forces the model to produce a *better student* — only
a more confident one.

Pushing the success metric out by one generation breaks that
degeneracy. G0 cannot satisfy the objective by being a more
confident teacher; G1 has to actually teach well enough that G2
can beat the evaluation suite. The grandchild is what counts.

The conceit is borrowed from Jean-Baptiste Lamarck (1809):
traits acquired by a parent in its lifetime are passed to its
offspring. Lamarck was wrong about biology — but he's right about
neural networks, which are exactly the systems where parental
fine-tuning literally becomes the child's starting weights.

## Architecture

```
                           ┌──────────────────────┐
                           │   eval_suite (frozen)│
                           │  • SWE-bench tasks   │
                           │  • curriculum tests  │
                           │  • ML-task harness   │
                           └──────────┬───────────┘
                                      │ score
                                      ▼
       ┌──────┐ curriculum ┌──────┐ curriculum ┌──────┐ measured
       │  G0  │──────────▶│  G1  │──────────▶│  G2  │──────────▶ G0's reward
       └──────┘            └──────┘            └──────┘
        DeepSeek-R1-       QLoRA on G0          QLoRA on G1
        Distill-Llama-70B
```

Three components:

### 1. G0 — the base model

[`deepseek-ai/DeepSeek-R1-Distill-Llama-70B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B),
loaded onto a RunPod A100 80GB pod (the canonical config; see
`reference_runpod_config` in memory).

G0 is frozen. We don't fine-tune G0 itself in any single
generational round. We only run inference: G0 generates the
curriculum that becomes G1's training data.

### 2. Curriculum — the data G0 emits for G1

A "curriculum" here is a structured dataset of `(problem, solution)`
pairs describing **how to train a model that is good at training
other models**. Concretely, each item is something like:

```json
{
  "task": "design a fine-tuning curriculum for a Llama-class model
           targeting code-completion on Python",
  "approach": "<reasoning trace from G0>",
  "curriculum": "<the actual training-data design that approach
                  would produce>",
  "expected_outcome": "<what student model capability would result>"
}
```

The curriculum's size is the first knob. Too small → G1 is just
G0 with a slight bias. Too large → bootstrap costs explode. M3
will start with 1k-10k items and scale from there.

### 3. Eval suite — what G2 is measured on

The eval suite is **fixed before training begins** and isolated
from G0/G1 so neither generation can game it. Candidate components:

- **SWE-bench Verified** — does G2 produce working patches for
  real GitHub issues? (External grounding.)
- **Curriculum-design task** — given a target capability, produce
  a fine-tuning curriculum; measured by training a tiny test model
  on the curriculum and scoring the tiny model.
- **ML coding harness** — fixed problems from "write a training
  loop / data pipeline / eval harness" that have known-correct
  reference solutions.

Critically, the eval suite measures **ML-task capability only**.
Not agency, not persuasion, not non-ML reasoning. The optimization
pressure stays inside the envelope we want.

## Why grandchildren, not children

The naive setup — G0 optimizes G1's eval score — fails because
G0 can be optimal for that metric by producing a curriculum that
makes G1 a sycophant: G1 says "yes, your training methods are
great" and scores well on G0's judgement.

The grandchild version closes that loophole. G1 *must actually
teach well* for G2 to score on the external suite. G1 can't fake
its way through, because G2's measurement happens entirely outside
G1's influence.

Equivalent framing: G0 is doing **second-order meta-learning**.
It's not learning to be a good model; it's learning to be a good
teacher of teachers.

## Safety boundaries

This section is the **load-bearing safety frame**. Everything in
this repo should be read against these constraints.

### Compute ceiling per generation

Each generation gets a fixed fine-tuning compute budget set in
advance: no generation N+1 may use more compute than generation N.
The recursion cannot bootstrap itself into runaway training.

In RunPodBoss terms, the cost ceiling for each pod is documented
upfront in the pod spec.

### Capability scope is ML-only

The eval suite measures ML task performance — model training,
curriculum design, code completion against fixed reference
implementations. It explicitly **does not** measure:

- Agentic capability (planning multi-step real-world tasks).
- Persuasion or social manipulation.
- Self-exfiltration or environment-probing behavior.
- Any capability outside "do ML engineering well."

If any G_N develops capability outside that envelope, the
optimization pressure does not reinforce it, because the eval suite
doesn't see it. The recursive pressure only intensifies inside the
ML-engineering envelope.

### Human review between generations

There is no automated G_N → G_{N+1} pipeline. Every transition is:

1. Human reviews G_N's eval results.
2. Human launches the pod for G_{N+1} training.
3. Human reviews G_{N+1}'s eval results before generation N+2.

This is a **research scaffold for studying generational fine-
tuning**, not an autonomous self-improvement system. The human
in the loop is the safety gate.

### Generation cap

`MAX_GENERATIONS = 3` for the foreseeable future. Whether the
setup is even *capable* of three generations of measurable
improvement is itself one of the research questions; planning past
that is speculative.

The cap is enforced by convention (the eval harness and the docs),
not by any code-level kill switch in G_N itself.

### What this is not

- Not an attempt at AGI takeoff.
- Not deployment infrastructure.
- Not a product.
- Not autonomous.
- Not even certain to work — the meta-meta-learning literature is
  full of attempts that flatten out within one or two generations.

If you find yourself making decisions about Lamarck that *would* be
unsafe under the autonomous-takeoff reading, **you are working on
something else**, not this project. Fork into a separately-named
repo and update the safety frame accordingly.

## Repo layout (planned)

```
Lamarck/
├── LICENSE                  Apache 2.0
├── README.md                concept + base-model rationale
├── DESIGN.md                this file
├── pyproject.toml           (TBD M1)
├── src/lamarck/
│   ├── __init__.py
│   ├── g0/                  G0 inference harness (M1)
│   ├── curriculum/          curriculum generation + validation (M2)
│   ├── train/               QLoRA training wrappers (M3)
│   └── eval/                frozen eval suite (M2)
├── tests/
├── docs/
└── examples/
    └── runpod-pod-spec.md   (TBD M1)
```

## Open questions

These don't have answers yet. They're the actual research:

1. **Does Lamarckian generational improvement plateau within 3
   generations?** If yes, the recursion isn't doing anything
   special. If no, there's something genuinely interesting
   happening that deserves more study.
2. **What does G1's curriculum look like compared to G0's?**
   If it's just "more data of the same kind G0 made," the model
   isn't actually learning to teach better — it's just memorizing
   one teacher's style.
3. **How sensitive is the result to the eval suite?** If a small
   change in eval task selection produces wildly different G2s,
   the optimization signal is too noisy for the recursion to
   converge.
4. **Can the compute ceiling be tightened across generations?**
   If G_{N+1} can be trained with strictly less compute than
   G_N and still improve, that's a genuinely positive sign for
   the approach. If it needs more, the setup has hit a wall.
5. **At what generation does the eval suite stop reflecting
   meaningful improvement?** Eval suites have ceilings. Hitting
   them isn't a successful recursion — it's an exhausted measure.

---

Proudly Made in Nebraska. Go Big Red! 🌽 https://xkcd.com/1838/
