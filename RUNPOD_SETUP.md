# RunPod Setup — Phase 1: Qwen-Image-2.1

Step-by-step deployment, every terminal command needed, and — because we hit a *lot* of them getting this working — every error we ran into along the way with the fix that actually worked. If you hit the same error, search this file for the exact text before troubleshooting from scratch.

See **[HOWITWORKS.md](HOWITWORKS.md)** for what the code actually does.

## Prerequisites
- A [RunPod](https://www.runpod.io/) account with billing set up.
- A [GitHub](https://github.com/) account (this repo is pulled onto the pod via `git`, no Docker build needed).
- Python 3.9+ locally, to run the frontend.

---

## 1. Deploy the pod

RunPod console → **Pods** → **Deploy**.

| Field | Value |
|---|---|
| **GPU** | 24GB is workable (see the memory section below for the settings needed to make it fit); 40GB+ avoids most of the memory tuning entirely. |
| **Pricing** | On-Demand (Spot pods can be killed mid-run). |
| **Container Image** | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (or whatever the **RunPod PyTorch** template offers — doesn't matter much, `bootstrap.sh` installs its own exact torch build regardless). |
| **Container Disk** | 30GB+ |
| **Volume Disk** | 100GB+, mounted at `/workspace` — **but see the disk quota error below**, this setting doesn't always mean what it says. |
| **Expose HTTP Ports** | `8888,8000` |
| **Expose TCP Ports** | `22` |

**Container Start Command** (replace the repo URL if you forked it):
```
bash -c "/start.sh & (bash -c 'set -e; (git clone https://github.com/Dravinesh/image_testing.git /workspace/image_testing || git -C /workspace/image_testing pull) && bash /workspace/image_testing/runpod_server/bootstrap.sh' || echo '=== BOOTSTRAP FAILED, see above — container staying alive for debugging ==='); tail -f /dev/null"
```
This clones (or pulls, on restart) the repo and runs `bootstrap.sh`. Critically, if anything fails, it prints the error and then just sits idle (`tail -f /dev/null`) instead of crash-looping the whole container — see the crash-loop error below for why this matters.

**Environment variables** to set on the pod:

| Var | Value | When |
|---|---|---|
| `API_KEY` | a secret you choose | Optional, but recommended — the proxy URL is publicly guessable. |
| `HF_TOKEN` | a Hugging Face token | Only if the model download 401s/403s (gated repo). |
| `CPU_OFFLOAD` | `1` or `sequential` | Only if you hit an out-of-memory error — see below. Start without it. |

---

## 2. Open a terminal and set up

Pod → **Connect** → **Start Web Terminal**.

```bash
# Confirm the GPU
nvidia-smi

# Check disk BEFORE downloading anything (see disk quota error below)
df -h

# Get the code
mkdir -p /workspace && cd /workspace
git clone https://github.com/Dravinesh/image_testing.git
cd /workspace/image_testing
git log -1 --oneline

# Run it in the foreground the first time, so you see exactly what happens
bash /workspace/image_testing/runpod_server/bootstrap.sh
```

Watch for, in order: `installing requirements...` → `using torch 2.5.1+cu124 from ...` → `Starting Qwen-Image server` → model files downloading from Hugging Face → `INFO: Application startup complete.`

In a **second** terminal tab, confirm it's really using the GPU:
```bash
curl http://localhost:8000/health
```
Want `"cuda":true`.

Get the public URL: pod → **Connect** → **HTTP Service [Port 8000]** → copy the `https://<pod-id>-8000.proxy.runpod.net` URL.

---

## 3. Run the frontend locally

```powershell
copy .env.example .env
```
Fill in `.env`:
```
RUNPOD_BASE_URL=https://<pod-id>-8000.proxy.runpod.net
API_KEY=<same value you set on the pod, or blank>
```
```powershell
py -m pip install -r frontend/requirements.txt
py -m streamlit run frontend/app.py
```
Open `http://localhost:8501` (or whatever port it prints — see the "port already in use" note below), click **Check health**, upload an image, enter a prompt, **Generate**.

---

## Every error we hit, and the fix that worked

These are in the rough order we hit them. If your symptom matches, jump straight to the fix — don't re-diagnose from scratch.

### 1. `infer_schema(func): Parameter q has unsupported type torch.Tensor`
Also seen as: `[transformers] Disabling PyTorch because PyTorch >= 2.5 is required but found 2.4.1+cu124`

**Cause:** The pod template's preinstalled PyTorch was 2.4.1. `diffusers`' dev branch (needed for `QwenImage21Pipeline`) and `transformers>=5.17` both require PyTorch ≥ 2.5.

**Fix:** Pin `torch>=2.5.1` in `requirements.txt`. (This alone wasn't enough — see the next error.)

### 2. Same error, still — even after pinning torch >= 2.5.1
**Cause:** The venv was created with `--system-site-packages`, so even though pip *did* install a newer torch into the venv, the pod image's **system** torch (2.4.1, at `/usr/local/lib/.../dist-packages/torch`) was shadowing it — Python imported the old one anyway. Confirmed by the traceback pointing at `dist-packages`, not the venv's own `site-packages`.

**Fix:** Rebuilt `bootstrap.sh` to:
- Create a **fully isolated** venv (dropped `--system-site-packages` entirely).
- `unset PYTHONPATH` defensively, in case the image exports it pointing at system packages.
- **Self-heal**: after activating the venv, actually `import torch` and check the version; if it's missing or `< 2.5`, wipe `/workspace/venv` and reinstall clean — automatically, on the very next restart, no manual SSH race against a crash loop required.

```bash
# manual equivalent, if you ever need to force it:
rm -rf /workspace/venv
bash /workspace/image_testing/runpod_server/bootstrap.sh
```

### 3. `ImportError: Qwen3VLVideoProcessor requires the Torchvision library`
**Cause:** `torchvision` wasn't listed in `requirements.txt` at all — the model's processor needs it.

**Fix:** Added `torchvision==0.20.1+cu124` to `requirements.txt`.

### 4. `UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 12080)`
**Cause:** With an unconstrained `torch>=2.5.1`, pip grabbed the newest available torch (`2.14.0`), which bundles **CUDA 13.0** — newer than the pod's driver (which supported up to CUDA 12.8). Left alone, this risks the GPU silently not being used at all.

**Fix:** Pin torch/torchvision to an exact CUDA 12.4 build via PyTorch's own wheel index, matching the pod's actual CUDA 12.4.1 toolkit:
```
--extra-index-url https://download.pytorch.org/whl/cu124
torch==2.5.1+cu124
torchvision==0.20.1+cu124
```

### 5. `RuntimeError: ... File reconstruction error: IO Error: Disk quota exceeded (os error 122)`
Happened ~32GB into a ~33GB model download.

**Cause:** `df -h` showed `/workspace` mounted on a multi-**petabyte** network filesystem (`mfs#...`) — that's the whole cluster's storage, not your personal allocation. RunPod enforces a separate quota on your slice that `df` doesn't reveal. Resizing "Volume Disk" to 110GB in the pod editor actually resized the **container** disk (`/`), not the `/workspace` volume — these are two distinct settings, easy to conflate.

**Fix (workaround, used to get unblocked):** point the HF cache at the container disk instead, which had plenty of free space:
```bash
export HF_HOME=/root/hf_cache
mkdir -p $HF_HOME
rm -rf /workspace/hf_cache
bash /workspace/image_testing/runpod_server/bootstrap.sh
```
**Trade-off:** `/root` is ephemeral — the ~33GB model re-downloads on every pod stop/restart. Fine for testing; for a permanent fix, confirm which field actually controls the `/workspace` mount's real quota (may require asking RunPod support, since `df` doesn't show it), or use a dedicated **Network Volume** instead of the pod's built-in volume.

**Note:** a later pod deploy showed no separate `/workspace` line in `df -h` at all (it was just part of the `overlay` root filesystem with 108GB free) — in that case the quota issue didn't occur, because there was no separate network-mounted volume with its own hidden quota. Whether you hit this depends on how RunPod provisions storage for that specific pod/region.

### 6. `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 96.00 MiB. GPU 0 has a total capacity of 23.53 GiB`
**Cause:** A 24GB GPU (RTX 4090) doesn't have enough VRAM to hold the entire model at once.

**Fix:** Enable CPU offload:
```bash
export CPU_OFFLOAD=1
bash /workspace/image_testing/runpod_server/bootstrap.sh
```
Then make it permanent: **Edit Pod** → add environment variable `CPU_OFFLOAD=1` → save, so it survives restarts without manually `export`ing it each time.

### 7. Denoising steps complete fine (`40/40`), then the process just says `Killed` — no Python traceback
**Cause:** Not a GPU memory issue (the steps already finished) — the Linux **OOM killer** hit the **container's memory cgroup limit**. `free -h` inside the container reported the **host's** RAM (251GB, mostly free), which was misleading — the actual limit was found with:
```bash
cat /sys/fs/cgroup/memory.max
# 30999998464  ≈ 28.9 GiB — the real, much lower, per-container cap
```
`dmesg` wasn't accessible (permission denied) to directly confirm the OOM killer in the container's logs, but the cgroup cap combined with the exact failure point (right after the loop, i.e. during VAE decode / the offload handoff) was conclusive enough.

**What we tried, in order:**
1. Enable VAE slicing/tiling to shrink the final decode step's memory. `QwenImage21Pipeline` doesn't implement the pipeline-level convenience methods (`enable_vae_slicing`/`enable_vae_tiling` — confirmed via warning in the logs), so we also tried calling them directly on the `.vae` submodule, which **did** succeed (`AutoencoderKLQwenImage21.enable_slicing() enabled`). Didn't fix it alone.
2. Lowered `MAX_INPUT_DIM` default from 2048 to 1024, to shrink the tensor being decoded. Didn't fix it alone either — still killed at the same point.
3. **This is what actually worked:** switched from `enable_model_cpu_offload()` (moves whole submodules between CPU/GPU — the handoff itself, e.g. transformer back to CPU + VAE onto GPU, can transiently spike memory) to `enable_sequential_cpu_offload()` (moves individual **layers** on/off GPU one at a time — much lower peak memory, noticeably slower generation):
   ```bash
   export CPU_OFFLOAD=sequential
   bash /workspace/image_testing/runpod_server/bootstrap.sh
   ```
   Make permanent via the pod's `CPU_OFFLOAD=sequential` environment variable.

**Takeaway:** on a tightly memory-capped pod, the offload *strategy* itself matters more than the output resolution or decode tiling. If `CPU_OFFLOAD=1` OOMs, jump straight to `CPU_OFFLOAD=sequential` rather than tuning resolution first.

### 8. `GET / HTTP/1.1" 404 Not Found` spamming the server log
**Not an error.** These are RunPod's own automated health-check pings hitting the root path `/`, which the server never defines a route for (only `/health` and `/generate` exist). The server is working correctly — ignore these lines. Only worry if `/health` itself 404s or times out.

### 9. Frontend shows `server error 404` when clicking Generate, even though the pod is healthy
**Cause:** The Streamlit sidebar's "API URL" text box keeps whatever was last typed/loaded in that **browser session** — if the tab was open from before the pod's URL changed (redeployed, restarted with a new proxy hostname, etc.), it silently keeps pointing at a stale/dead URL.

**Fix:** Hard-refresh the browser tab (`Ctrl+F5`) or close/reopen it, re-verify the sidebar's API URL field matches the current pod's URL exactly, then **Check health** before **Generate**.

### 10. Renaming `.env.example` to `.env` via the file explorer/IDE silently creates a stray `.example` file instead
**Cause:** Windows Explorer / some IDEs treat the first `.` in a dotfile name as the "filename" and the rest as the "extension" — renaming `.env.example` this way strips to just `.example` instead of producing `.env`.

**Fix:** Never rename dotfiles via right-click. Use the terminal:
```powershell
copy .env.example .env
```

### 11. Pod crash-loops every ~15s, and it's hard to tell why because Container Logs scroll past before you can read them
**Cause:** The original start command propagated any failure (a failed `git pull`, or `bootstrap.sh`/`uvicorn` crashing) straight to the container's main process exiting, which made RunPod immediately restart the **entire container** — Nginx, SSH, Jupyter, everything — every ~15-17 seconds. This also meant you couldn't get a stable terminal open fast enough to debug ("racing the crash loop").

**Fix:** Wrap the whole thing so a failure prints its error and then the container just idles instead of exiting:
```
bash -c "/start.sh & (bash -c 'set -e; ... && bash bootstrap.sh' || echo 'BOOTSTRAP FAILED, see above'); tail -f /dev/null"
```
This is now the standing Container Start Command (see section 1 above). With this, a failed pod stays reachable via SSH/web terminal indefinitely for calm debugging.

### 12. Testing a pod's direct IP:port from an automated/sandboxed tool environment times out or refuses to connect
**Cause:** Some tool/agent sandboxes only allow outbound standard HTTP(S) traffic, not arbitrary raw TCP to non-standard ports — this looks identical to the pod's port genuinely being unreachable, but isn't.

**Fix:** Always verify a direct IP:port from **your own machine's terminal** (`curl http://<ip>:<port>/health`) before concluding the pod itself is misconfigured. If it works from your machine but not from an automated tool, it's the tool's network restriction, not the pod.

### 13. `bash: $'\r': command not found` / `start.sh: line N: syntax error`
**Cause:** `.sh` files edited/committed from Windows can pick up CRLF line endings, which break bash on Linux.

**Fix:** Added `.gitattributes`:
```
*.sh text eol=lf
```
This forces shell scripts to always be checked out with LF endings regardless of the committer's OS. If you ever hit this on a file that predates `.gitattributes`, fix it with `git add --renormalize .` and re-commit.

### 14. A stray `frontend/package-lock.json` got committed
**Cause:** An accidental `npm install` (or similar) run inside `frontend/`, which is a pure-Python directory — npm has no business there.

**Fix:** `git rm --cached` it and added `package-lock.json` to `.gitignore`.

---

## Restarting after a pod stop
```bash
cd /workspace/image_testing
git pull   # picks up any code fixes since you last ran it
bash /workspace/image_testing/runpod_server/bootstrap.sh
```
If `/workspace` wasn't a real persistent volume (see error 5), you'll also re-download the model — that's expected, not a new bug.

## Shutting down
- **Stop** the pod when not testing — GPU billing stops. Whether your code/model survives depends on whether `/workspace` is genuinely persistent for your pod (see error 5) — check before assuming it is.
- **Terminate** deletes the pod and its volume disk entirely. Only do this when fully done.
