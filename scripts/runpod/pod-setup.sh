#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
#
# Pod-side dependency install. Runs on a fresh RunPod A100 80GB pod
# (canonical config: torch-v280 image). Idempotent — safe to re-run.
#
# Adapted from Dave's setup_dave.sh, with vLLM added for the
# post-train serve step (Lamarck's serve.sh keeps the pod alive
# after training instead of tearing it down).
#
# Run as: bash pod-setup.sh
#
# PEP-668 gotcha: RunPod's base image uses an externally-managed
# Python. We use `--break-system-packages` rather than venv because
# the pod is single-purpose throwaway infrastructure — no need to
# isolate from a "system Python" that's also single-purpose.

set -euo pipefail

echo "=== Lamarck pod setup ==="
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# --- pre-flight ---------------------------------------------------------------
python_version=$(python3 --version | awk '{print $2}')
echo "Python: $python_version"

if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q True; then
    gpu_name=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    vram_gb=$(python3 -c "import torch; print(int(torch.cuda.get_device_properties(0).total_memory / 1024**3))" 2>/dev/null)
    echo "GPU: $gpu_name (${vram_gb}GB)"
else
    echo "WARNING: no CUDA GPU detected. Install will proceed but training + serving will not."
fi

# --- training stack -----------------------------------------------------------
echo ""
echo "[1/4] Installing training stack…"
pip install --break-system-packages --upgrade \
    "transformers>=4.46" \
    "peft>=0.13" \
    "trl>=0.12" \
    "bitsandbytes>=0.44" \
    "accelerate>=1.1" \
    "datasets>=3.0" \
    "huggingface_hub[hf_transfer]>=0.26" \
    >/tmp/pod-setup-train.log 2>&1
echo "[+] training stack installed"

# --- inference stack ----------------------------------------------------------
echo ""
echo "[2/4] Installing vLLM (inference server)…"
# vLLM pins its own torch — install last so it doesn't get clobbered.
pip install --break-system-packages --upgrade "vllm>=0.6" \
    >/tmp/pod-setup-vllm.log 2>&1
echo "[+] vLLM installed"

# --- HF transfer accelerator --------------------------------------------------
# hf_transfer dramatically speeds up the 70B base-model download
# (multi-GB shards). Without it, fetching DeepSeek-R1-Distill-Llama-70B
# can take 30+ minutes.
echo ""
echo "[3/4] Enabling hf_transfer for fast model downloads…"
export HF_HUB_ENABLE_HF_TRANSFER=1
echo "    set HF_HUB_ENABLE_HF_TRANSFER=1 in current shell"
echo "    (persist in ~/.bashrc for sshd-inherited shells)"
if ! grep -q HF_HUB_ENABLE_HF_TRANSFER ~/.bashrc 2>/dev/null; then
    echo "export HF_HUB_ENABLE_HF_TRANSFER=1" >> ~/.bashrc
    echo "[+] persisted to ~/.bashrc"
fi

# --- sanity check -------------------------------------------------------------
echo ""
echo "[4/4] Sanity checks…"
python3 -c "
import torch, transformers, peft, trl, bitsandbytes, vllm
print(f'    torch         {torch.__version__}')
print(f'    transformers  {transformers.__version__}')
print(f'    peft          {peft.__version__}')
print(f'    trl           {trl.__version__}')
print(f'    bitsandbytes  {bitsandbytes.__version__}')
print(f'    vllm          {vllm.__version__}')
print(f'    cuda          {torch.cuda.is_available()}')
"

echo ""
echo "=== pod setup complete ==="
echo "Next: bash scripts/runpod/train.py   # or RUN_LAMARCK.sh"
