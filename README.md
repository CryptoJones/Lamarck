<div align="center">

```
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║                  L  A  M  A  R  C  K                         ║
║                                                              ║
║      a 70B model whose only job is to raise better children  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

**Lamarckian generational ML.** Train a 70B model whose sole
objective is to produce a successor that produces a better
successor. The grandchild — not the child — is the unit of
success.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?logo=apache)](LICENSE)
[![Base model](https://img.shields.io/badge/G0-DeepSeek--R1--Distill--Llama--70B-4f46e5)](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B)
[![Status](https://img.shields.io/badge/status-scaffold-orange)](DESIGN.md)

</div>

> Will live on both [GitHub](https://github.com/CryptoJones/Lamarck)
> and [Codeberg](https://codeberg.org/CryptoJones/Lamarck) once
> the empty repos exist. Issues on either forge welcome; commits
> land on both.

---

## The idea

Most ML self-improvement attempts optimize a model to make its
*own next step* better. That's a degenerate objective — the model
hill-climbs its own training loss, and "better at training itself"
collapses into "better at training on its own outputs."

Lamarck moves the optimization target out by one generation:

```
G0  ─train data→  G1  ─train data→  G2

       └────────  evaluate ────────┘
```

**G0's objective is the eval performance of G2**, not of G1.
G0 generates the *training curriculum* that produces G1; G1
generates the curriculum that produces G2; and only G2's
ML-task capability counts.

Why this matters:

- **G1 can't just memorize G0's outputs** — that wouldn't help G2.
- **The pressure is on the *teaching strategy*, not the student.**
  G1 has to learn to be a *better teacher* than G0 was, because
  that's what makes G2 strong.
- **The metric is grounded.** G2's capability is measured on a
  fixed, externally-defined ML evaluation suite — not on anything
  G0 or G1 chose. The model can't game its own goalposts.

The name is Lamarck's: traits acquired during a parent's lifetime
(in our case, weights learned by fine-tuning) are passed directly
to offspring. That's the exact opposite of Darwinian evolution
and the exact mechanism of model fine-tuning. The biology turned
out to be wrong; the engineering turns out to be right.

---

## Base model

**G0:** [`deepseek-ai/DeepSeek-R1-Distill-Llama-70B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B)

Picked because:

1. **70B parameter scale.** The original spec.
2. **R1's reasoning trace habit.** The meta-task — "design a
   curriculum that will produce a better-at-curriculum-design
   successor" — is fundamentally a reasoning task. R1 was
   trained to think before answering; the distilled 70B inherits
   that disposition.
3. **Open weights, permissive license.** No vendor lock-in,
   no per-call API costs once we're past the bootstrap, and
   no terms-of-service constraint on the recursive setup.
4. **Llama-family architecture.** Plays nicely with the
   QLoRA-on-A100 fine-tuning recipe already in our infrastructure
   (see RunPodBoss + the canonical [RunPod 70B QLoRA pod recipe]).

[RunPod 70B QLoRA pod recipe]: ../../.claude/projects/-home-akclark-Source-repos-Dave/memory/reference_runpod_config.md

---

## How it works (in 10 lines)

```python
# Conceptual sketch — see DESIGN.md for the actual scaffold.
G0 = DeepSeekR1DistillLlama70B()

curriculum_for_G1 = G0.generate_training_curriculum(target="produce a model that can generate training curricula")
G1 = qlora_finetune(base=G0, data=curriculum_for_G1)

curriculum_for_G2 = G1.generate_training_curriculum(target="produce a model that is good at ML tasks")
G2 = qlora_finetune(base=G1, data=curriculum_for_G2)

score = eval_suite.score(G2)  # this is G0's reward signal.
```

The reward routes backward: G2's score becomes G1's reward
becomes G0's reward, two generations later. G0 never sees a
direct gradient from G2's evaluation, but the data it generates
for G1 is shaped over training rounds to produce better G2s.

---

## Status

**Scaffold only.** Repo, design doc, base-model choice, scope
boundaries. Nothing trains yet.

| Milestone | Description                                                  | Status |
|-----------|--------------------------------------------------------------|--------|
| M0        | Repo scaffold + DESIGN.md + safety boundaries                | shipped |
| M1        | G0 inference harness (load DeepSeek-R1-Distill-Llama-70B on A100, generate sample curriculum) | planned |
| M2        | Eval suite frozen (the ML tasks G2 will be measured against) | planned |
| M3        | G0 → G1 single training round (QLoRA on RunPod)              | planned |
| M4        | G1 → G2 single training round + first G2 eval                | planned |
| M5        | Curriculum search loop (multiple G0 rounds, each with a new G1/G2 pair) | planned |
| M6        | Generation-3 reach test (does the recursion hold?)           | speculative |

---

## Safety boundaries

This is **a research scaffold, not an autonomous improvement loop**.
The boundaries baked into the design (and enforced by the eval
suite) are intentional, not aspirational:

- **Compute ceiling per generation.** No generation gets more
  fine-tuning hours than its parent. The recursion can't bootstrap
  itself into a runaway budget.
- **Capability scope is ML-only.** The eval suite measures
  performance on ML benchmarks (model-training, curriculum-design,
  code-completion). It does not reward agency, persuasion,
  self-exfiltration, or any non-ML capability. If G_N develops
  capability outside that envelope, the eval suite doesn't see it
  and the optimization pressure doesn't reinforce it.
- **Human review between generations.** No automated G_N → G_{N+1}
  pipeline. Each transition is a human-launched RunPod pod with a
  documented spec.
- **Generation cap = 3** for now. Whether the setup actually has
  to be cut off earlier is itself one of the research questions.

See [`DESIGN.md`](DESIGN.md) for the safety section in detail.

---

## License

Apache 2.0. See [LICENSE](LICENSE).

Proudly Made in Nebraska. Go Big Red! 🌽 https://xkcd.com/1838/
