# `seed-data/` — committed knowledge corpora

This directory holds **fixed knowledge corpora** that G0 can draw
from when designing curricula. Distinct from [`../curricula/`],
which holds *generated* per-generation training data and is
gitignored.

Files here are checked in deliberately, by hand. Each one should:

1. Be self-contained — no external links that rot.
2. Carry attribution at the top (author / source / license, where
   determinable).
3. Be loaded via the curriculum-design step, not used as direct
   fine-tuning data, unless explicitly noted otherwise.

## Current contents

| Entry                                      | What it is                                                                                    |
|--------------------------------------------|-----------------------------------------------------------------------------------------------|
| `training-knowledge-from-archives.md`      | Mixed-corpus knowledge dump — Linux kernel 2.4 internals (Tigran Aivazian), GNOME-era systems content, older OS/dev material. Curated from Aaron's Obsidian vault as seed material for G0's curriculum-design rounds. |
| `arxiv-ml/`                                | ~215k full-text arXiv ML papers, sourced from `armanc/scientific_papers` on HF. **Reproducible-from-script, not committed** — the corpus is ~30-50 GB expanded. See [`arxiv-ml/README.md`](arxiv-ml/README.md) for the fetch flow. |

[`../curricula/`]: ../curricula/
