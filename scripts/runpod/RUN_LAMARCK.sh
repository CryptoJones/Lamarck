#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
#
# Lamarck end-to-end pod orchestrator.
#
#   pod-setup.sh  →  train.py  →  serve.sh  (foreground; pod stays alive)
#
# Unlike Dave's RUN_DAVE.sh, this script does NOT exit after training.
# After train.py completes successfully, it exec's into serve.sh so
# the pod keeps running and the freshly-baked adapter is loaded into
# vLLM, ready for an SSH-tunneled Hermes agent to talk to.
#
# Flags (positional):
#   --skip-setup        skip pod-setup.sh (deps already installed)
#   --skip-train        skip train.py (e.g. loading a published adapter
#                       you've already trained elsewhere)
#   --gen N             generation number (default: 1)
#
# Environment is the same as the underlying scripts — see each one
# for documented LAMARCK_* knobs.

set -euo pipefail

SKIP_SETUP=0
SKIP_TRAIN=0
GENERATION=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-setup)  SKIP_SETUP=1; shift ;;
        --skip-train)  SKIP_TRAIN=1; shift ;;
        --gen)         GENERATION="$2"; shift 2 ;;
        --gen=*)       GENERATION="${1#--gen=}"; shift ;;
        -h|--help)
            sed -n '4,21p' "$0"
            exit 0
            ;;
        *)
            echo "ERROR: unknown flag: $1" >&2
            exit 2
            ;;
    esac
done

export LAMARCK_GENERATION="$GENERATION"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "================================================="
echo "  Lamarck — RunPod pipeline (G${GENERATION})"
echo "================================================="
echo "  Skip setup:  $SKIP_SETUP"
echo "  Skip train:  $SKIP_TRAIN"
echo "  Started:     $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================="
echo

# Step 1 — Environment.
if [ "$SKIP_SETUP" -eq 0 ]; then
    echo "[1/3] Pod setup…"
    bash "$REPO_ROOT/scripts/runpod/pod-setup.sh"
else
    echo "[1/3] Skipping pod setup (--skip-setup)."
fi
echo

# Step 2 — Train.
if [ "$SKIP_TRAIN" -eq 0 ]; then
    echo "[2/3] Training G${GENERATION}…"
    python3 "$REPO_ROOT/scripts/runpod/train.py"
else
    echo "[2/3] Skipping training (--skip-train)."
fi
echo

# Step 3 — Serve. exec replaces this shell so vLLM owns the pod.
echo "[3/3] Loading adapter into vLLM + serving on localhost:8000…"
echo "    (Ctrl+C to stop the server and release the GPU.)"
exec bash "$REPO_ROOT/scripts/runpod/serve.sh"
