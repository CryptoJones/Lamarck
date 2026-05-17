# `seed-data/arxiv-ml/` — arXiv ML papers corpus

Full-text arXiv ML papers, sourced from the
[`armanc/scientific_papers`](https://huggingface.co/datasets/armanc/scientific_papers)
HF dataset (arxiv subset, ~215k full-text papers prepared by Cohan
et al. 2018 — the de-facto "arxiv full text on HF").

**The actual papers are not committed to this repo.** Multi-GB
corpora are git-hostile. The fetch script ([`../../scripts/data_collection/fetch-arxiv-papers.py`])
is the reproducible artifact; the corpus lives on whatever machine
runs the script (a pod, a workstation with disk).

[`../../scripts/data_collection/fetch-arxiv-papers.py`]: ../../scripts/data_collection/fetch-arxiv-papers.py

## How to fetch

On the pod (or any machine with ~50GB disk + good network):

```bash
pip install --break-system-packages datasets

# Full corpus (~215k papers, ~30-50GB JSONL on disk):
python3 scripts/data_collection/fetch-arxiv-papers.py \
    --out seed-data/arxiv-ml/full

# Or a sample for validation (100 papers, ~25MB):
python3 scripts/data_collection/fetch-arxiv-papers.py \
    --out seed-data/arxiv-ml/sample --limit 100
```

Output is chunked JSONL — 10,000 papers per file:

```
seed-data/arxiv-ml/full/papers-0000.jsonl
seed-data/arxiv-ml/full/papers-0001.jsonl
...
seed-data/arxiv-ml/full/manifest.json
```

Each JSONL line is a paper:

```json
{
  "abstract": "...",
  "article":  "<full text body>",
  "section_names": ["Introduction", "Related Work", ...]
}
```

## Why this corpus

The dataset already extracted full text from arXiv PDFs (which is
the hard part) and packages it as clean structured records. Using
it spares us from running our own PDF→text pipeline against tens of
thousands of papers.

The trade-off: this is a snapshot from ~2018; recent papers are not
included. For Lamarck's purposes — G0 absorbing "ML research
methodology" knowledge — that's fine; the methodological content
of ML hasn't shifted so much that 2010-2018 papers are useless. If
later generations need cutting-edge work, we can layer on a fresh
arXiv API pull then.

## License + provenance

The `armanc/scientific_papers` dataset card cites:

> Cohan, A. et al. "A Discourse-Aware Attention Model for Abstractive
> Summarization of Long Documents." NAACL-HLT 2018.

Individual arXiv papers carry their authors' chosen licenses
(CC BY, CC0, arXiv-perpetual, etc.). The dataset itself is on HF
under standard HF dataset terms. Use for research is well-trodden
ground; redistributing the corpus directly is on each user.

For Lamarck's purposes — feeding the corpus to G0 as seed material
for curriculum design, with the actual training data being
G0-generated rather than the corpus itself — we're in standard
text-and-data-mining research territory.

## Why not committed

- The full corpus is ~30-50 GB expanded. Git is hostile to that
  size of repo even with LFS.
- The corpus is fully reproducible from the script + dataset id.
  No information loss by not committing.
- A `--limit 100` sample _could_ be committed (~25 MB) for
  validation, but if you actually want to inspect the data you
  can just run the sample command yourself.
