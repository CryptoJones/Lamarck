#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
#
# Lamarck end-to-end pod orchestrator.
#
#   pod-setup → [pull parent] → [train → publish] → [pull self] → serve
#
# Unlike Dave's RUN_DAVE.sh, this script does NOT exit after training.
# Hugging Face is the persistent home for the adapter; the pod is
# ephemeral compute. After train.py, the adapter is pushed to HF
# immediately (so pod-death doesn't lose the model), then vLLM is
# launched in the foreground so the pod stays alive and a local
# Hermes agent can talk to it via SSH tunnel.
#
# Flags (positional):
#   --skip-setup        skip pod-setup.sh (deps already installed)
#   --skip-train        skip train.py — pull adapter from HF instead
#                       and go straight to serve
#   --no-publish        skip the post-train HF upload (e.g. for
#                       throwaway experimental runs)
#   --gen N             generation number (default: 1)
#
# Environment knobs (all optional; see each underlying script):
#   LAMARCK_HF_ADAPTER_REPO       this generation's HF repo
#                                 (default: CryptoJones/lamarck-g${GEN}-adapter)
#   LAMARCK_HF_PARENT_REPO        G_{N-1}'s HF repo (for G2+)
#                                 (default: CryptoJones/lamarck-g$((GEN-1))-adapter)
#   LAMARCK_PARENT_ADAPTER        local path for parent adapter
#                                 (default: <repo>/adapters/g$((GEN-1)))
#   HF_TOKEN                      required for publish (write access);
#                                 optional for pull (only if repos private)

set -euo pipefail

SKIP_SETUP=0
SKIP_TRAIN=0
NO_PUBLISH=0
GENERATION=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-setup)  SKIP_SETUP=1; shift ;;
        --skip-train)  SKIP_TRAIN=1; shift ;;
        --no-publish)  NO_PUBLISH=1; shift ;;
        --gen)         GENERATION="$2"; shift 2 ;;
        --gen=*)       GENERATION="${1#--gen=}"; shift ;;
        -h|--help)
            sed -n '4,33p' "$0"
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

# Default HF repos. Override LAMARCK_HF_ADAPTER_REPO / LAMARCK_HF_PARENT_REPO
# directly if you want to point at someone else's published lineage.
HF_REPO="${LAMARCK_HF_ADAPTER_REPO:-CryptoJones/lamarck-g${GENERATION}-adapter}"
export LAMARCK_HF_ADAPTER_REPO="$HF_REPO"

if [ "$GENERATION" -gt 1 ]; then
    PARENT_GEN=$((GENERATION - 1))
    PARENT_REPO="${LAMARCK_HF_PARENT_REPO:-CryptoJones/lamarck-g${PARENT_GEN}-adapter}"
    PARENT_DIR="${LAMARCK_PARENT_ADAPTER:-$REPO_ROOT/adapters/g${PARENT_GEN}}"
    export LAMARCK_PARENT_ADAPTER="$PARENT_DIR"
fi

echo "================================================="
echo "  Lamarck — RunPod pipeline (G${GENERATION})"
echo "================================================="
echo "  Skip setup:   $SKIP_SETUP"
echo "  Skip train:   $SKIP_TRAIN"
echo "  No publish:   $NO_PUBLISH"
echo "  HF repo:      $HF_REPO"
if [ "$GENERATION" -gt 1 ]; then
    echo "  Parent repo:  $PARENT_REPO"
fi
echo "  Started:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================="
echo

# Step 1 — Environment.
if [ "$SKIP_SETUP" -eq 0 ]; then
    echo "[1/5] Pod setup…"
    bash "$REPO_ROOT/scripts/runpod/pod-setup.sh"
else
    echo "[1/5] Skipping pod setup (--skip-setup)."
fi
echo

# Step 2 — Pull parent adapter if we're training G2+ and don't have it locally.
if [ "$GENERATION" -gt 1 ] && [ "$SKIP_TRAIN" -eq 0 ]; then
    echo "[2/5] Pulling parent adapter G${PARENT_GEN} for training…"
    LAMARCK_GENERATION="$PARENT_GEN" \
    LAMARCK_HF_ADAPTER_REPO="$PARENT_REPO" \
    LAMARCK_ADAPTER_DIR="$PARENT_DIR" \
        bash "$REPO_ROOT/scripts/runpod/pull-adapter.sh"
else
    echo "[2/5] No parent to pull (G1, or --skip-train)."
fi
echo

# Step 3 — Train (or pull this generation's adapter from HF instead).
if [ "$SKIP_TRAIN" -eq 0 ]; then
    echo "[3/5] Training G${GENERATION}…"
    python3 "$REPO_ROOT/scripts/runpod/train.py"
else
    echo "[3/5] Skipping training — pulling G${GENERATION} adapter from HF…"
    bash "$REPO_ROOT/scripts/runpod/pull-adapter.sh"
fi
echo

# Step 4 — Publish to HF immediately. The pod is ephemeral; if it
# dies before we get here we have nothing. Run this BEFORE serve.sh.
if [ "$SKIP_TRAIN" -eq 0 ] && [ "$NO_PUBLISH" -eq 0 ]; then
    echo "[4/5] Publishing G${GENERATION} adapter to HF…"
    bash "$REPO_ROOT/scripts/runpod/publish-adapter.sh"
elif [ "$SKIP_TRAIN" -eq 1 ]; then
    echo "[4/5] Skipping publish (--skip-train; adapter pulled from HF)."
else
    echo "[4/5] Skipping publish (--no-publish)."
fi
echo

# Step 5 — Serve. exec replaces this shell so vLLM owns the pod.
echo "[5/5] Loading adapter into vLLM + serving on localhost:8000…"
echo "    (Ctrl+C to stop the server and release the GPU.)"
exec bash "$REPO_ROOT/scripts/runpod/serve.sh"
