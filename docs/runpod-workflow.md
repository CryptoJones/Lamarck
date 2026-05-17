# RunPod workflow — train and serve, then keep the pod alive

How to run a Lamarck generation on a RunPod A100 pod, then leave
the pod running so a local Hermes agent can chat with the freshly-
trained adapter via an SSH tunnel.

This document is the operator playbook. The actual scripts live
under [`scripts/runpod/`](../scripts/runpod/).

---

## TL;DR

```bash
# 1. Spin up a RunPod A100 80GB pod (canonical config in your
#    memory: torch-v280 image).
# 2. On the pod (over SSH):
git clone <lamarck-repo> /workspace/Lamarck
cd /workspace/Lamarck
bash scripts/runpod/RUN_LAMARCK.sh --gen 1
#    pod-setup → train → serve (foreground; pod stays alive).

# 3. From your local machine, SSH-tunnel:
ssh -L 8000:localhost:8000 root@<pod-host>

# 4. From your local Hermes (e.g. as the hermes user):
hermes model add lamarck-g1 http://localhost:8000/v1 --no-key
hermes
```

---

## The four scripts

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

Runs pod-setup → train → serve in sequence. The serve step uses
`exec` so the pod owns the vLLM process directly (no orphaned shell).

Flags:

- `--skip-setup`  — deps already installed.
- `--skip-train`  — load + serve an existing adapter without re-training.
- `--gen N`       — which generation (default 1).

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

## Pulling the adapter back to local

If you want to keep the adapter around after the pod dies, pull it
to local before stopping the pod:

```bash
rsync -avz root@<pod-host>:/workspace/Lamarck/adapters/g1/ \
    /home/akclark/Source/adapters/lamarck-g1/
```

Then `serve.sh` can load it locally too if you ever spin up a new
A100 pod and want to skip retraining.

---

Proudly Made in Nebraska. Go Big Red! 🌽 https://xkcd.com/1838/
