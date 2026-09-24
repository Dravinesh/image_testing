# Image Testing — Phase 1: Qwen-Image-2.1 on RunPod

Phase 1 runs [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1) behind a FastAPI server on a **RunPod GPU Pod**. A **Streamlit** frontend sends it an image + prompt and shows the result.
The YOLO / Gemini / OpenAI cleanup pipeline is **not** part of this phase.

## How it works

No Docker build is needed. The pod uses RunPod's stock **PyTorch** image, and its **Container Start Command** clones this repo from GitHub and runs `runpod_server/bootstrap.sh`. That script:
1. creates a fully isolated Python venv on the `/workspace` volume (does **not** reuse the image's preinstalled PyTorch — the model needs a newer version than most templates ship, and reusing it risks the venv's own install being silently shadowed),
2. installs `runpod_server/requirements.txt`, only on first boot or when the file changes (and self-heals — recreates the venv from scratch — if torch ever ends up missing or older than 2.5 in it),
3. starts the FastAPI server on port 8000. Model weights are cached in `/workspace/hf_cache`.

On every pod restart, the start command runs `git pull` first, so pushing code changes to GitHub and restarting the pod is all it takes to update.

```
image_testing/
├── runpod_server/
│   ├── app.py              # FastAPI: /health, /generate
│   ├── bootstrap.sh        # run on the pod by the start command
│   └── requirements.txt    # server deps (installed on the pod)
├── frontend/
│   ├── app.py              # Streamlit UI (runs locally)
│   └── requirements.txt
├── .env.example
├── .gitattributes          # keeps *.sh with LF line endings
├── .gitignore
└── README.md
```

## Prerequisites

- A [RunPod](https://www.runpod.io/) account with billing set up.
- A [GitHub](https://github.com/) account.
- Python 3.9+ locally, to run the frontend.

## 1. Install frontend dependencies

```powershell
py -m pip install -r frontend/requirements.txt
copy .env.example .env
```

## 2. Push this project to GitHub

1. On github.com → **New repository** → e.g. `image_testing`. Keep it **Public** (it has no secrets, because `.env` is git-ignored), and don't add a README.
2. From this folder:
   ```powershell
   git init
   git add .
   git commit -m "Phase 1: Qwen-Image-2.1 RunPod server + Streamlit frontend"
   git branch -M main
   git remote add origin https://github.com/<github-user>/image_testing.git
   git push -u origin main
   ```
   `<github-user>` is your GitHub username.

**Private repo instead?** Create a GitHub fine-grained token with *Contents: Read-only* for this repo only. Then use `https://<token>@github.com/<github-user>/image_testing.git` as the URL in the start command below.

## 3. Deploy on RunPod

1. RunPod console → **Pods** → **Deploy** → pick a GPU.
   - A 48–80 GB card (**L40S / RTX A6000 48 GB**, **A100 80GB**, **H100 80GB**). VRAM needs aren't published on the model card, so 80 GB is the safest for the first test. On a smaller GPU, set `CPU_OFFLOAD=1` (slower).
   - Pricing: **On-Demand** (Spot pods can be killed mid-run).
2. Template: **RunPod PyTorch**. Click **Edit Template** / **Customize Deployment** and set:

   | Field | Value |
   |---|---|
   | **Container image** | The template's image, e.g. `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`. Pick the newest `runpod/pytorch` tag offered (PyTorch ≥ 2.4, CUDA 12.x). |
   | **Container start command** | see below |
   | **Container disk** | 30 GB |
   | **Volume disk** | **100 GB**, mount path `/workspace` (venv + model cache; kept across stop/start) |
   | **Expose HTTP ports** | `8888,8000` (Jupyter + our server) |
   | **Expose TCP ports** | `22` |

   **Container start command** (replace `<github-user>`):
   ```
   bash -c "/start.sh & (git clone https://github.com/<github-user>/image_testing.git /workspace/image_testing || git -C /workspace/image_testing pull) && bash /workspace/image_testing/runpod_server/bootstrap.sh"
   ```
   - `/start.sh &` keeps RunPod's normal services (SSH, Jupyter, web terminal) running in the background.
   - `git clone ... || git pull` clones on first boot and pulls the latest code on later boots.
   - `bootstrap.sh` installs the deps if needed and starts the server.

3. **Environment variables**:

   | Var | Default | Purpose |
   |---|---|---|
   | `API_KEY` | *(empty = no auth)* | Bearer token required by `/generate`. **Recommended**, because the proxy URL is public. |
   | `HF_TOKEN` | — | Hugging Face token. Only needed if the model repo is gated. |
   | `MODEL_ID` | `Qwen/Qwen-Image-2.1` | HF model id. |
   | `CPU_OFFLOAD` | `0` | `1` enables `enable_model_cpu_offload()` for lower VRAM. `sequential` enables `enable_sequential_cpu_offload()` — much lower peak memory (both VRAM and system RAM), but noticeably slower; use this if `1` still gets OOM-killed on a memory-capped pod. |
   | `DEFAULT_STEPS` | `40` | Default inference steps. |
   | `MAX_INPUT_DIM` | `2048` | Input images are downscaled to this max side. |

4. **Deploy On-Demand**. Then open **Connect → HTTP Service [Port 8000]** to get the public URL. It looks like `https://<pod-id>-8000.proxy.runpod.net`.

To change a setting later, edit the pod's env vars and restart the pod.

### Gated model? (only if the download fails with 401/403)

1. Log in at [huggingface.co](https://huggingface.co), open the model page, and accept its terms if asked.
2. **Settings → Access Tokens → Create new token** (Read role).
3. Add `HF_TOKEN=hf_...` as a pod env var and restart the pod.

## 4. Wait for the server to be ready

First boot installs the packages and downloads the model, which takes several minutes. Watch the pod **Logs** in the RunPod console for:
`installing requirements...` → `Starting Qwen-Image server` → `model loaded in ...s` → `Application startup complete`.
Later boots skip the install and the download.

Or poll:
```bash
curl https://<pod-id>-8000.proxy.runpod.net/health
```
`"status": "ok"` means it's ready. `"loading"` means the weights are still loading. A `502` means the server isn't up yet.

## 5. Test it

Fill in `.env` with the pod's URL (and the same `API_KEY` you set on the pod):
```
RUNPOD_BASE_URL=https://<pod-id>-8000.proxy.runpod.net
API_KEY=choose-a-secret
```

Run the frontend:
```powershell
py -m streamlit run frontend/app.py
```
It opens `http://localhost:8501`. The sidebar is pre-filled from `.env`. Click **Check health** → upload an image → type a prompt → **Generate** → download the PNG.

### API (for curl / scripts)

`POST /generate` takes `multipart/form-data` and returns `image/png`.

| Field | Required | Notes |
|---|---|---|
| `prompt` | yes | Edit instruction (or generation prompt if no image). |
| `image` | no | Input image. If present → **edit** mode, otherwise **text-to-image**. |
| `num_inference_steps` | no | Default 40. |
| `seed` | no | Integer for reproducible results. |
| `width`, `height` | no | Output size (mainly for text-to-image). The model card lists 2048×2048, plus the 4:3, 3:4, 3:2, 2:3, 16:9 and 9:16 aspect ratios. |

The `Authorization: Bearer <API_KEY>` header is required only when `API_KEY` is set on the pod.
Response headers: `X-Mode` (`edit` / `t2i`), `X-Inference-Seconds`.

```bash
curl -X POST https://<pod-id>-8000.proxy.runpod.net/generate \
  -H "Authorization: Bearer $API_KEY" \
  -F "prompt=Remove the cardboard boxes from the floor" \
  -F "image=@room.jpg" \
  -o result.png
```

## 6. Updating the code

1. Edit locally, then `git commit` and `git push`.
2. On RunPod, **restart** the pod. The start command pulls the new code, and `bootstrap.sh` reinstalls packages only if `requirements.txt` changed.
3. To force a clean reinstall (e.g. to pick up a newer diffusers commit), delete `/workspace/venv` from the web terminal and restart.

## 7. Shut down

RunPod bills by the second while a pod is running.
- **Stop** the pod when not testing. GPU billing stops, and `/workspace` (code, venv, model cache) is kept, with only a small storage charge. **Start** it again later and the server comes up on its own.
- **Terminate** deletes the pod **and its volume disk**. Use it only when you're done.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ImportError: cannot import name 'QwenImage21Pipeline'` | diffusers is too old. Delete `/workspace/venv` and restart the pod to reinstall from git. |
| `infer_schema(func): Parameter q has unsupported type torch.Tensor` at startup | The venv is using an old/shadowed PyTorch (< 2.5). `bootstrap.sh` now self-heals: it checks the torch version after activating the venv and automatically wipes + recreates the venv if it's missing or too old, so a crash-looping pod fixes itself within one or two auto-restarts once it has pulled this fix. Watch the logs for `torch missing or older than 2.5 in venv ... — recreating venv from scratch`. If it's still stuck after a few minutes, delete `/workspace/venv` from the web terminal and restart. |
| `[transformers] Disabling PyTorch because PyTorch >= 2.5 is required but found ...` | Same fix as above. |
| `ImportError: Qwen3VLVideoProcessor requires the Torchvision library` | `torchvision` was missing from `requirements.txt` — now pinned. `git pull` this fix and restart; the hash check will reinstall automatically. |
| `UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version ...)` | pip grabbed a torch build for a newer CUDA than the pod's driver supports. `torch`/`torchvision` are now pinned to exact CUDA 12.4 builds via `--extra-index-url https://download.pytorch.org/whl/cu124`, matching the pod image's CUDA 12.4.1 toolkit. `git pull` and restart. |
| `CUDA out of memory` / HTTP 507 | Redeploy on a bigger GPU, or set `CPU_OFFLOAD=1` and restart. |
| Process gets `Killed` right after the denoising steps finish (no Python traceback) | The container's memory cgroup limit was hit (check with `cat /sys/fs/cgroup/memory.max` — this can be much lower than what `free -h` reports, which shows the *host's* RAM). `CPU_OFFLOAD=1`'s submodule handoff (transformer back to CPU, VAE onto GPU) can itself spike peak memory. Set `CPU_OFFLOAD=sequential` instead — much lower peak memory, slower generation — and restart. |
| `GatedRepoError` / `401` / `403` in the logs while downloading | Add `HF_TOKEN` (see *Gated model?* above). |
| `fatal: could not read Username` in the logs | The repo is private. Make it public or use the token URL (section 2). |
| `$'\r': command not found` | `bootstrap.sh` has Windows line endings. `.gitattributes` prevents this; if it still happens, re-commit after `git add --renormalize .`. |
| `No space left on device` | Increase the Volume disk. The venv and weights live in `/workspace`. |
| Pod keeps restarting | Check the pod **Logs**. It's usually one of the errors above, since the container exits if the server fails to start. |
| Request fails after ~100 s (`524`) | RunPod proxy timeout. See *Notes* below. |
| `401 invalid or missing API key` from `/generate` | The `API_KEY` in `.env` / the sidebar doesn't match the pod's `API_KEY`. |

## Notes / known caveats

- **Proxy timeout**: RunPod's HTTP proxy (`*.proxy.runpod.net`) cuts requests at ~100 s. If a generation takes longer (large images / many steps), reduce steps or size. Or expose port 8000 as a **TCP port** in the pod settings and use `http://<public-ip>:<mapped-port>` as `RUNPOD_BASE_URL`.
- The server processes one request at a time (single GPU lock). Concurrent requests wait in line.
- Nothing in this phase has been run or tested yet. The first pod boot is the integration test.
