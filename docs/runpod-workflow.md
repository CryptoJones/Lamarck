# RunPod workflow — train, persist to HF, serve, keep the pod alive

How to run a Lamarck generation on a RunPod A100 pod, push the
trained adapter to Hugging Face so it survives pod death, then
leave the pod running so a local Hermes agent can chat with the
adapter via an SSH tunnel.

**The pod is ephemeral compute. Hugging Face is the persistent
home for the adapter.** Every flow either pulls the adapter from
HF or publishes to it — local pod storage is a working copy,
never the source of truth.

This document is the operator playbook. The actual scripts live
under [`scripts/runpod/`](../scripts/runpod/).

---

## TL;DR

```bash
# 0. One-time: a HF token with write access in HF_TOKEN.
export HF_TOKEN=hf_***  # https://huggingface.co/settings/tokens

# 1. Spin up a RunPod A100 80GB pod (canonical config: torch-v280).
# 2. On the pod (over SSH):
git clone <lamarck-repo> /workspace/Lamarck
cd /workspace/Lamarck
HF_TOKEN=$HF_TOKEN bash scripts/runpod/RUN_LAMARCK.sh --gen 1
#    pod-setup → train → publish to HF → serve (foreground; pod stays alive)

# 3. From your local machine, SSH-tunnel:
ssh -L 8000:localhost:8000 root@<pod-host>

# 4. From your local Hermes (e.g. as the hermes user):
hermes model add lamarck-g1 http://localhost:8000/v1 --no-key
hermes
```

If the pod dies later, spin up a fresh one and run with `--skip-train`:

```bash
HF_TOKEN=$HF_TOKEN bash scripts/runpod/RUN_LAMARCK.sh --gen 1 --skip-train
#    pod-setup → pull G1 from HF → serve
```

Same Hermes session resumes after re-tunneling.

---

## The six scripts

### `pod-setup.sh` — install dependencies

One-time, idempotent. Installs:

- **Training stack:** `transformers`, `peft`, `trl`, `bitsandbytes`,
  `accelerate`, `datasets`, `huggingface_hub[hf_transfer]`.
- **Inference stack:** `vllm` (LoRA-aware OpenAI-compatible server).
- **HF transfer accelerator:** sets `HF_HUB_ENABLE_HF_TRANSFER=1`
  in `~/.bashrc` so the 70B base-model download takes minutes,
  not half an hour.

Uses `pip install --break-system-packages` because RunPod's base
image gates pip with PEP 668. The pod is single-purpose throwaway
infra — no point in fighting it with a venv.

### `pull-adapter.sh` — fetch adapter from HF

Downloads a published adapter from Hugging Face into the local
adapter dir. Required env: `LAMARCK_HF_ADAPTER_REPO` (defaults to
`CryptoJones/lamarck-g${GEN}-adapter`). If the local dir already
has a valid `adapter_config.json`, it short-circuits — pass
`--force` to overwrite.

Used in three places by the orchestrator:

1. **G2+ training:** pulls the parent adapter so `train.py` can
   stack on it.
2. **`--skip-train`:** pulls *this* generation's adapter to skip
   straight to serving (e.g. fresh pod after the old one died).
3. **Pre-flight in `serve.sh`:** if the local dir is missing,
   `serve.sh` points at this script.

Uses `huggingface_hub.snapshot_download` rather than the
`huggingface-cli` because the Python API gives finer control over
the destination directory and PEFT layout.

### `publish-adapter.sh` — push adapter to HF

Uploads `LAMARCK_ADAPTER_DIR` to `LAMARCK_HF_ADAPTER_REPO` (default
`CryptoJones/lamarck-g${GEN}-adapter`, private by default — flip
`LAMARCK_HF_PRIVATE=0` for public). Requires `HF_TOKEN` with write
access.

Auto-creates the repo if it doesn't exist. Uses `HfApi.upload_folder`
under the hood (the path Dave's `publish_adapter.sh` settled on
after `huggingface-cli upload` had rough edges).

Runs immediately after `train.py` in the orchestrator. **This is the
"insurance" step:** if the pod dies any time after this point, the
adapter is recoverable from HF without retraining.

### `train.py` — QLoRA fine-tune

Reads `curricula/training.jsonl` (one `{"prompt", "completion"}`
JSON object per line), QLoRA fine-tunes the base model, writes the
adapter to `adapters/g${LAMARCK_GENERATION}/`.

Generation-aware: `LAMARCK_GENERATION=2` writes to `adapters/g2/`
and reads `LAMARCK_PARENT_ADAPTER` to stack on top of an earlier
generation's adapter.

Writes `adapter_metadata.json` alongside the adapter so the lineage
(base model, parent adapter, training data size, timestamp) is
captured. `serve.sh` prints this on launch so you know exactly what
you're talking to.

Knobs (env vars):

| Var                        | Default                                            |
|----------------------------|----------------------------------------------------|
| `LAMARCK_BASE_MODEL`       | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B`        |
| `LAMARCK_GENERATION`       | `1`                                                |
| `LAMARCK_DATA_DIR`         | `<repo>/curricula`                                 |
| `LAMARCK_ADAPTER_DIR`      | `<repo>/adapters/g${GEN}`                          |
| `LAMARCK_PARENT_ADAPTER`   | (none — G1 trains from base; set for G2+)         |
| `LAMARCK_EPOCHS`           | `2`                                                |

### `serve.sh` — load adapter + start vLLM

Foregrounds vLLM with the base model + the freshly-trained adapter,
listens on **`127.0.0.1:8000`** only. No public exposure — the SSH
tunnel is the access path.

Key vLLM flags:

```
--enable-lora
--lora-modules "lamarck-g${GEN}=<adapter dir>"
--max-loras 1
--max-lora-rank 16          # matches train.py's LoraConfig(r=16)
--max-model-len 4096         # matches train.py's max_seq_length
--gpu-memory-utilization 0.92
--host 127.0.0.1             # localhost only
--port 8000
```

The server runs in the foreground; the pod stays alive as long as
the process is alive. **Ctrl+C / SIGTERM releases the GPU and
ends the pod's billable time** (modulo RunPod's per-minute rounding).

### `RUN_LAMARCK.sh` — orchestrator

Runs the full pipeline in five steps:

```
1. pod-setup            (skippable with --skip-setup)
2. pull parent adapter  (G2+ only; pulls G_{N-1} from HF for training)
3. train OR pull self   (train fresh, OR --skip-train → pull G_N from HF)
4. publish to HF        (after fresh train; skippable with --no-publish)
5. serve via vLLM       (foreground; pod stays alive)
```

The serve step uses `exec` so vLLM owns the pod's foreground process.

Flags:

- `--skip-setup`  — deps already installed.
- `--skip-train`  — pull this generation's adapter from HF and skip
  straight to serve. Useful when a previous pod died and you're
  resuming on a fresh one.
- `--no-publish`  — skip the post-train HF upload (only useful for
  throwaway experimental runs).
- `--gen N`       — which generation (default 1). G2+ also pulls
  the parent adapter from HF.

---

## The SSH tunnel

Once `serve.sh` is up on the pod, from your local machine:

```bash
ssh -L 8000:localhost:8000 root@<pod-host>
```

Replace `<pod-host>` with whatever RunPod gives you in the pod's
"Connect" panel (typically a URL ending in `proxy.runpod.net` with
a numeric port). Leave the SSH session open in a terminal somewhere
— the tunnel dies with the SSH session.

If you'd rather background it:

```bash
ssh -fN -L 8000:localhost:8000 root@<pod-host>
# To kill the tunnel later:
ps aux | grep "ssh -fN" | awk '{print $2}' | xargs kill
```

`-fN` = "fork to background, run no remote command."

---

## Pointing Hermes at the tunnel

From whichever local user is running Hermes (you set up the
isolated `hermes` user earlier — that's where Hermes should be
running):

```bash
hermes model add lamarck-g1 http://localhost:8000/v1 --no-key
```

Then launch Hermes and select the model:

```bash
hermes
> /model lamarck-g1
> what curriculum would produce a better successor model?
```

Hermes talks to the vLLM server over the SSH tunnel. The pod sees
requests coming in from `127.0.0.1`; the network never sees the
traffic.

---

## When you're done

Two paths:

1. **Keep the pod alive (default).** vLLM is still running; you can
   come back later, re-tunnel, and resume the Hermes session.
   **Costs money continuously.** Set a RunPodBoss cost ceiling
   per the existing integration so this doesn't leak overnight.

2. **Shut down.** SSH into the pod, `Ctrl+C` the `serve.sh`
   process. The vLLM server stops; on the RunPod web console,
   stop the pod. The adapter still lives on the pod's persistent
   volume (if you configured one) or vanishes with the pod
   (if you didn't — pull `adapters/g${GEN}/` to local first).

The whole point of this workflow — over Dave's "train and tear
down" — is **path 1**. We trade ongoing cost for low-latency
Hermes interaction with the trained model. If you want path 2
behavior, just use Dave's RUN_DAVE.sh pattern instead.

---

## Pulling the adapter to your local machine

Usually unnecessary — HF is the persistent home. But if you want a
local backup of the adapter on your laptop / workstation:

```bash
huggingface-cli download \
    CryptoJones/lamarck-g1-adapter \
    --local-dir ~/Source/adapters/lamarck-g1
```

Or via Python:

```python
from huggingface_hub import snapshot_download
snapshot_download("CryptoJones/lamarck-g1-adapter",
                  local_dir="~/Source/adapters/lamarck-g1")
```

You can later run `serve.sh` locally (on any A100-class GPU) by
setting `LAMARCK_ADAPTER_DIR` to that path and skipping all the
pod plumbing.

---

Proudly Made in Nebraska. Go Big Red! 🌽 https://xkcd.com/1838/
