#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
"""Fetch arXiv full-text ML papers from a Hugging Face dataset.

Uses `armanc/scientific_papers` (arxiv subset) — ~215k full-text
papers with abstracts and section names. The dataset was prepared
by Cohan et al. (2018) and is the de-facto "arxiv full text on HF"
dataset for ML/NLP work.

Defaults to streaming (doesn't materialize the full ~7GB tar before
reading), writes JSONL to disk one paper per line:

    {"abstract": "...", "article": "...", "section_names": [...]}

Run on a pod or a machine with disk + bandwidth — pulling the full
215k is multi-GB on disk and tens of minutes over good network.

Usage:
    # Full pull (215k papers, multi-GB):
    python3 fetch-arxiv-papers.py --out seed-data/arxiv-ml/full

    # Sample pull (validation; 100 papers):
    python3 fetch-arxiv-papers.py --out seed-data/arxiv-ml/sample --limit 100

    # Custom dataset name + split:
    python3 fetch-arxiv-papers.py --dataset ccdv/arxiv-classification \\
                                  --split train --limit 1000 \\
                                  --out seed-data/arxiv-ml/classified

Dependencies (install on the pod, not in-repo):
    pip install --break-system-packages datasets

Output layout:
    <out>/papers-0000.jsonl   first 10k papers
    <out>/papers-0001.jsonl   next 10k papers
    ...
    <out>/manifest.json       counts + dataset spec + fetch timestamp
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


CHUNK_SIZE = 10_000  # papers per output file — keeps individual JSONLs grep-able


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset",
        default="armanc/scientific_papers",
        help="HF dataset id (default: armanc/scientific_papers)",
    )
    parser.add_argument(
        "--config",
        default="arxiv",
        help="Dataset configuration / subset (default: arxiv)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split (default: train)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max papers to fetch (default: all)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory — will be created if missing",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Materialize the full dataset first instead of streaming. "
             "Use only if you have the disk budget and want random access.",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit(
            "ERROR: datasets library not installed.\n"
            "  pip install --break-system-packages datasets"
        )

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Dataset:  {args.dataset} / {args.config} / {args.split}")
    print(f"Stream:   {not args.no_stream}")
    print(f"Limit:    {args.limit or 'all'}")
    print(f"Out:      {args.out}")
    print()

    ds = load_dataset(
        args.dataset,
        args.config,
        split=args.split,
        streaming=not args.no_stream,
    )

    chunk: list[dict] = []
    chunk_idx = 0
    total = 0

    def flush() -> None:
        nonlocal chunk, chunk_idx
        if not chunk:
            return
        out_path = args.out / f"papers-{chunk_idx:04d}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for paper in chunk:
                fh.write(json.dumps(paper) + "\n")
        print(f"  wrote {out_path}  ({len(chunk)} papers)")
        chunk_idx += 1
        chunk = []

    try:
        for item in ds:
            # Schema for armanc/scientific_papers: article, abstract,
            # section_names. Other datasets may differ — we just take
            # whatever fields are present.
            paper = {
                k: v for k, v in item.items()
                if isinstance(v, (str, list, int, float, bool, type(None)))
            }
            chunk.append(paper)
            total += 1

            if len(chunk) >= CHUNK_SIZE:
                flush()

            if args.limit and total >= args.limit:
                break

        flush()

    except KeyboardInterrupt:
        print("\nInterrupted — flushing partial chunk…")
        flush()

    manifest = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "stream": not args.no_stream,
        "limit": args.limit,
        "total_papers": total,
        "chunks": chunk_idx,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nDone. {total} papers in {chunk_idx} chunk(s).")
    print(f"Manifest: {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
