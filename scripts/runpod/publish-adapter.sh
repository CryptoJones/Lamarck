#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
#
# Publish a trained Lamarck adapter to Hugging Face Hub. Run on the
# pod immediately after training so the adapter is safe even if the
# pod dies. Companion to pull-adapter.sh — together they make the
# pod ephemeral and the HF copy authoritative.
#
# Usage:
#   bash publish-adapter.sh
#
# Environment:
#   LAMARCK_HF_ADAPTER_REPO   HF repo to push to (default:
#                             CryptoJones/lamarck-g${GEN}-adapter).
#                             Auto-created if it doesn't exist.
#   LAMARCK_ADAPTER_DIR       adapter to upload (default:
#                             <repo>/adapters/g${GEN})
#   LAMARCK_GENERATION        generation number (default: 1)
#   LAMARCK_HF_PRIVATE        1 = private repo (default 1; flip to 0
#                             only when you mean to publish to the world)
#   HF_TOKEN                  required; must have WRITE access. Generate
#                             at https://huggingface.co/settings/tokens.
#
# Idempotent. Re-running uploads any new/changed files; HfApi handles
# the diff. Adapter weights themselves are atomic (single safetensors).

set -euo pipefail

GENERATION="${LAMARCK_GENERATION:-1}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HF_REPO="${LAMARCK_HF_ADAPTER_REPO:-CryptoJones/lamarck-g${GENERATION}-adapter}"
ADAPTER_DIR="${LAMARCK_ADAPTER_DIR:-$REPO_ROOT/adapters/g${GENERATION}}"
HF_PRIVATE="${LAMARCK_HF_PRIVATE:-1}"

echo "=== Lamarck adapter publish ==="
echo "Local dir:    $ADAPTER_DIR"
echo "HF repo:      $HF_REPO (private=$HF_PRIVATE)"
echo

# --- preflight ---------------------------------------------------------------
if [ ! -f "$ADAPTER_DIR/adapter_config.json" ]; then
    echo "ERROR: $ADAPTER_DIR is not a valid PEFT adapter dir" >&2
    echo "       (no adapter_config.json found)" >&2
    exit 1
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN env var not set. Generate a write token at" >&2
    echo "       https://huggingface.co/settings/tokens and re-run." >&2
    exit 1
fi
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "ERROR: huggingface_hub not installed. Run pod-setup.sh first." >&2
    exit 1
fi

# --- create-if-missing + upload via HfApi ------------------------------------
# Python > shell here because `huggingface-cli upload` historically had
# rough edges around private-repo creation + auto-init.
python3 - <<PY
import os, sys
from huggingface_hub import HfApi, RepositoryNotFoundError

api = HfApi(token=os.environ["HF_TOKEN"])
repo_id     = "$HF_REPO"
private     = $HF_PRIVATE == 1
local_dir   = "$ADAPTER_DIR"
generation  = $GENERATION

try:
    api.repo_info(repo_id=repo_id, repo_type="model")
    print(f"Repo exists: {repo_id}")
except RepositoryNotFoundError:
    print(f"Creating repo: {repo_id} (private={private})")
    api.create_repo(repo_id=repo_id, private=private, repo_type="model")

print(f"Uploading {local_dir} → {repo_id}")
api.upload_folder(
    folder_path=local_dir,
    repo_id=repo_id,
    repo_type="model",
    commit_message=f"Lamarck G{generation} adapter",
)
print(f"Done: https://huggingface.co/{repo_id}")
PY

echo
echo "[+] adapter published"
echo "    https://huggingface.co/$HF_REPO"
