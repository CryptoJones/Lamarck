#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
#
# Post-train serve. Loads base model + the latest adapter into vLLM
# and exposes an OpenAI-compatible API on localhost:8000.
#
# This script does NOT expose the endpoint publicly. From your local
# machine, ssh-tunnel into the pod and Hermes (running locally) hits
# http://localhost:8000:
#
#   # On your local machine:
#   ssh -L 8000:localhost:8000 root@<pod-host>
#   # In a separate local shell:
#   hermes model add lamarck-g1 http://localhost:8000/v1 --no-key
#
# Environment:
#   LAMARCK_BASE_MODEL      base weights (default DeepSeek-R1-Distill-Llama-70B)
#   LAMARCK_ADAPTER_DIR     adapter to load (default adapters/g${LAMARCK_GENERATION})
#   LAMARCK_GENERATION      which generation to serve (default 1)
#   LAMARCK_SERVE_PORT      listen port (default 8000)
#   LAMARCK_SERVE_HOST      listen address (default 127.0.0.1 — localhost only)
#
# The script foregrounds vLLM so the pod stays alive as long as the
# server is running. Kill the process (Ctrl+C / SIGTERM) to release
# the GPU; RunPodBoss can be wired in separately to enforce an idle
# / cost ceiling so the pod doesn't burn money forever.

set -euo pipefail

GENERATION="${LAMARCK_GENERATION:-1}"
BASE_MODEL="${LAMARCK_BASE_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Llama-70B}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ADAPTER_DIR="${LAMARCK_ADAPTER_DIR:-$REPO_ROOT/adapters/g${GENERATION}}"
SERVE_PORT="${LAMARCK_SERVE_PORT:-8000}"
SERVE_HOST="${LAMARCK_SERVE_HOST:-127.0.0.1}"

# --- pre-flight ---------------------------------------------------------------
if [ ! -d "$ADAPTER_DIR" ] || [ ! -f "$ADAPTER_DIR/adapter_config.json" ]; then
    echo "ERROR: no valid PEFT adapter at: $ADAPTER_DIR" >&2
    echo "Options:" >&2
    echo "  1. Train fresh:   bash scripts/runpod/train.py" >&2
    echo "  2. Pull from HF:  bash scripts/runpod/pull-adapter.sh" >&2
    echo "  3. Set LAMARCK_ADAPTER_DIR to wherever you have it." >&2
    exit 1
fi
if [ ! -f "$ADAPTER_DIR/adapter_metadata.json" ]; then
    echo "WARN: $ADAPTER_DIR has no adapter_metadata.json — proceeding but" \
         "lineage / provenance won't be logged." >&2
else
    echo "=== adapter metadata ==="
    cat "$ADAPTER_DIR/adapter_metadata.json"
    echo
fi

if ! python3 -c "import vllm" 2>/dev/null; then
    echo "ERROR: vllm not installed. Run scripts/runpod/pod-setup.sh first." >&2
    exit 1
fi

echo "=== Lamarck serve (G${GENERATION}) ==="
echo "Base model:   $BASE_MODEL"
echo "Adapter:      $ADAPTER_DIR"
echo "Listen on:    $SERVE_HOST:$SERVE_PORT"
echo "Started:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo
echo "From your local machine, set up the SSH tunnel:"
echo "  ssh -L $SERVE_PORT:localhost:$SERVE_PORT root@<pod-host>"
echo
echo "Then point Hermes at the local end of the tunnel:"
echo "  hermes model add lamarck-g${GENERATION} \\"
echo "      http://localhost:$SERVE_PORT/v1 --no-key"
echo
echo "Press Ctrl+C to stop the server (releases the GPU)."
echo "================================="

# --- launch vLLM --------------------------------------------------------------
# --enable-lora           on; this is the whole point.
# --lora-modules          named "lamarck-g${GENERATION}=$ADAPTER_DIR" so
#                         the OpenAI request model name = "lamarck-g${GENERATION}".
# --max-loras 1           one adapter at a time; we're not multi-tenanting.
# --max-lora-rank 16      matches the rank used in train.py.
# --gpu-memory-utilization 0.92  leave a sliver for the KV cache.
# --max-model-len 4096    matches max_seq_length in train.py.
# --host 127.0.0.1        localhost-only; SSH tunnel is the access path.
exec python3 -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --enable-lora \
    --lora-modules "lamarck-g${GENERATION}=$ADAPTER_DIR" \
    --max-loras 1 \
    --max-lora-rank 16 \
    --gpu-memory-utilization 0.92 \
    --max-model-len 4096 \
    --host "$SERVE_HOST" \
    --port "$SERVE_PORT"
