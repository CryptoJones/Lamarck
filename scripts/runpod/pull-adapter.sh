#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
#
# Pull a Lamarck adapter from Hugging Face Hub to a local directory
# on the pod. The HF copy is the persistent source of truth — pods
# are ephemeral; the adapter has to live on HF before the pod dies.
#
# Usage:
#   bash pull-adapter.sh
#
# Environment:
#   LAMARCK_HF_ADAPTER_REPO   HF repo to pull (default:
#                             CryptoJones/lamarck-g${GEN}-adapter)
#   LAMARCK_HF_REVISION       revision/branch/commit (default: main)
#   LAMARCK_ADAPTER_DIR       where to write (default:
#                             <repo>/adapters/g${GEN})
#   LAMARCK_GENERATION        generation number (default: 1)
#   HF_TOKEN                  optional; required if the HF repo is
#                             private. Read access is enough.
#
# Idempotent — safe to re-run. If the local dir already has a valid
# adapter_config.json and `--force` isn't passed, the pull is a no-op.

set -euo pipefail

GENERATION="${LAMARCK_GENERATION:-1}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HF_REPO="${LAMARCK_HF_ADAPTER_REPO:-CryptoJones/lamarck-g${GENERATION}-adapter}"
HF_REVISION="${LAMARCK_HF_REVISION:-main}"
ADAPTER_DIR="${LAMARCK_ADAPTER_DIR:-$REPO_ROOT/adapters/g${GENERATION}}"

FORCE=0
for arg in "$@"; do
    case "$arg" in
        --force|-f) FORCE=1 ;;
        --help|-h)  sed -n '2,21p' "$0"; exit 0 ;;
        *)          echo "ERROR: unknown flag: $arg" >&2; exit 2 ;;
    esac
done

echo "=== Lamarck adapter pull ==="
echo "HF repo:      $HF_REPO @ $HF_REVISION"
echo "Local dir:    $ADAPTER_DIR"
echo

# --- short-circuit if we already have it -------------------------------------
if [ "$FORCE" -eq 0 ] && [ -f "$ADAPTER_DIR/adapter_config.json" ]; then
    echo "[=] adapter already present at $ADAPTER_DIR — skipping pull"
    echo "    (re-run with --force to overwrite)"
    exit 0
fi

# --- preflight ---------------------------------------------------------------
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "ERROR: huggingface_hub not installed. Run pod-setup.sh first." >&2
    exit 1
fi

mkdir -p "$ADAPTER_DIR"

# --- pull via HfApi.snapshot_download ----------------------------------------
# Uses Python rather than `huggingface-cli download` because the CLI
# is awkward for "download to this exact dir" — snapshot_download with
# local_dir lands the files where vLLM expects them.
python3 - <<PY
import os, sys
from huggingface_hub import snapshot_download

repo_id   = "$HF_REPO"
revision  = "$HF_REVISION"
local_dir = "$ADAPTER_DIR"
token     = os.environ.get("HF_TOKEN")

print(f"Downloading {repo_id}@{revision} → {local_dir}")
path = snapshot_download(
    repo_id=repo_id,
    revision=revision,
    local_dir=local_dir,
    token=token,
)
print(f"Done. Files at: {path}")
PY

# --- sanity check ------------------------------------------------------------
if [ ! -f "$ADAPTER_DIR/adapter_config.json" ]; then
    echo "ERROR: pull completed but $ADAPTER_DIR/adapter_config.json is missing." >&2
    echo "       The HF repo may not contain a valid PEFT adapter." >&2
    exit 1
fi

echo
echo "[+] adapter pulled to $ADAPTER_DIR"
if [ -f "$ADAPTER_DIR/adapter_metadata.json" ]; then
    echo "=== adapter metadata ==="
    cat "$ADAPTER_DIR/adapter_metadata.json"
fi
