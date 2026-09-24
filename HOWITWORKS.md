# How It Works — Phase 1: Qwen-Image-2.1 on RunPod

This document explains the architecture and code, not how to deploy it — see **[RUNPOD_SETUP.md](RUNPOD_SETUP.md)** for step-by-step deployment, commands, and the full list of errors hit + fixes.

## Goal

Run [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1) (an image generation/editing model) behind a small API on a **RunPod GPU Pod**, and a **Streamlit** page on your own machine that sends it an image + prompt and shows the result. Nothing else — no YOLO/Gemini/OpenAI cleanup pipeline (that's a separate, later phase).

## High-level architecture

```
┌─────────────────────────┐         HTTPS          ┌──────────────────────────────────┐
│  Your machine            │  ───────────────────▶  │  RunPod GPU Pod                   │
│  frontend/app.py         │   POST /generate        │  runpod_server/app.py (FastAPI)   │
│  (Streamlit, port 8501)  │   GET  /health           │  QwenImage21Pipeline on the GPU   │
└─────────────────────────┘  ◀───────────────────   └──────────────────────────────────┘
        reads .env                  PNG bytes              loaded once at startup,
   (RUNPOD_BASE_URL,                                        cached in memory for
        API_KEY)                                            every request after
```

There's no message queue, no database, no auth beyond an optional bearer token. One process, one model, one GPU.

## Repo layout

```
image_testing/
├── runpod_server/
│   ├── app.py                # the FastAPI server (this IS the product)
│   ├── bootstrap.sh          # what actually runs on the pod at boot
│   └── requirements.txt      # server-side Python deps
├── frontend/
│   ├── app.py                # Streamlit page (runs on your machine)
│   └── requirements.txt
├── .env.example               # template — copy to .env, fill in, never commit .env
├── .gitattributes              # forces *.sh to LF line endings (see RUNPOD_SETUP.md)
├── README.md                   # original deploy walkthrough
├── HOWITWORKS.md                # this file
└── RUNPOD_SETUP.md              # step-by-step setup + full error/fix log
```

## `runpod_server/app.py` — the server

A single-file FastAPI app. On startup (`lifespan`), it loads `QwenImage21Pipeline` once and keeps it in a module-level `pipe` variable — the model stays resident in memory/VRAM for the life of the process, so requests don't pay the load cost each time.

### Endpoints

**`GET /health`**
```json
{"status": "ok", "model": "Qwen/Qwen-Image-2.1", "cuda": true, "gpu": "NVIDIA GeForce RTX 4090"}
```
`status` is `"loading"` until the model finishes loading, then `"ok"`. `cuda` tells you whether the GPU is actually being used — if this is ever `false`, something is wrong with the torch/CUDA build (see RUNPOD_SETUP.md's error log).

**`POST /generate`** — `multipart/form-data`, returns raw `image/png` bytes.

| Field | Required | Meaning |
|---|---|---|
| `prompt` | yes | Text instruction. |
| `image` | no | If present → **edit mode** (image-to-image). If absent → **text-to-image mode**. |
| `num_inference_steps` | no | Default from `DEFAULT_STEPS` env var (40). More steps = higher quality, slower. |
| `seed` | no | For reproducible output. |
| `width`, `height` | no | Mainly relevant for t2i mode. |

Response headers: `X-Mode` (`edit`/`t2i`), `X-Inference-Seconds`.

### Why a `threading.Lock`
```python
pipe_lock = threading.Lock()
```
There's exactly one GPU and one model instance. If two requests arrive at once, the second one just waits for the first to finish rather than both trying to run on the GPU simultaneously (which would either crash or silently corrupt results). This means the server processes **one generation at a time** — it's not built for concurrent throughput, just correctness for a test/dev setup.

### Input image handling
`_load_input_image()` converts to RGB and downscales anything larger than `MAX_INPUT_DIM` (default 1024px) using Lanczos resampling. This exists for two reasons: the model has a practical resolution ceiling, and — more importantly, discovered the hard way — **larger images make the final VAE decode step use much more memory**, which matters a lot on a memory-constrained pod (see RUNPOD_SETUP.md, "Process gets Killed").

### `CPU_OFFLOAD` — the three modes
The model doesn't necessarily fit entirely in a GPU's VRAM (a 24GB card was too small). `CPU_OFFLOAD_MODE` (from the `CPU_OFFLOAD` env var) controls how diffusers handles this:

| Value | Method | Behavior |
|---|---|---|
| `0` (default) | `pipe.to("cuda")` | Whole model resident on GPU. Fastest. Needs the most VRAM. |
| `1` / `model` | `enable_model_cpu_offload()` | Whole submodules (text encoder, transformer, VAE) move between CPU and GPU as needed. Lower VRAM, but the handoff between submodules can itself spike peak memory. |
| `sequential` | `enable_sequential_cpu_offload()` | Individual **layers** move on/off GPU one at a time. Much lower peak memory (both VRAM and system RAM), but noticeably slower generation. |

Pick the cheapest mode that actually works on your GPU/RAM combination — try `0`, then `1`, then `sequential` if you keep hitting out-of-memory errors.

### VAE slicing/tiling
At load time, the server tries to enable memory-saving decode modes — both the usual pipeline-level convenience methods and, since `QwenImage21Pipeline` doesn't implement those, directly on the VAE submodule as a fallback:
```python
targets = [(p, "enable_vae_slicing"), (p, "enable_vae_tiling")]
vae = getattr(p, "vae", None)
if vae is not None:
    targets += [(vae, "enable_slicing"), (vae, "enable_tiling")]
```
Whatever succeeds gets logged; whatever doesn't exist just logs a warning and is skipped. This is defensive — it doesn't assume any particular diffusers pipeline API surface.

## `runpod_server/bootstrap.sh` — how the server actually starts

There's **no Docker image** for this project. Instead, the RunPod pod's **Container Start Command** clones this repo from GitHub and runs this script, which:

1. Creates a **fully isolated** Python venv at `/workspace/venv` (deliberately *not* using `--system-site-packages` — see RUNPOD_SETUP.md's shadowing bug).
2. Installs `requirements.txt`, but only if it changed since last time (hash-checked, so restarts are fast once everything's installed).
3. **Self-heals**: after activating the venv, it actually imports torch and checks the version. If it's missing or older than 2.5 (stale venv, some other shadowing issue), it wipes the venv and reinstalls from scratch — automatically, without a human needing to SSH in.
4. Starts `uvicorn app:app` on port 8000.

Model weights are cached in `HF_HOME` (`/workspace/hf_cache` by default) so they aren't re-downloaded every restart — assuming `/workspace` has enough real, persistent quota (not guaranteed — see RUNPOD_SETUP.md).

## `runpod_server/requirements.txt`
Notably pins `torch==2.5.1+cu124` and `torchvision==0.20.1+cu124` from PyTorch's own CUDA-12.4 wheel index — deliberately, not left to pip's default resolution, because that grabbed a much newer CUDA-13-linked torch build than most pod drivers support. `diffusers` is installed straight from GitHub, since `QwenImage21Pipeline` only exists in the unreleased dev branch.

## `frontend/app.py` — the Streamlit page

A single-page app: sidebar for the server URL + API key + generation settings, a file uploader, a prompt box, and a Generate button. It's a thin HTTP client over the `/generate` endpoint — all the actual work happens on the pod. `load_dotenv()` pre-fills the sidebar from your local `.env` so you don't have to retype the pod URL every time you restart it.

## Environment variables — full reference

| Var | Where it's used | Default | Purpose |
|---|---|---|---|
| `MODEL_ID` | pod | `Qwen/Qwen-Image-2.1` | HF model repo id. |
| `API_KEY` | pod + frontend | *(empty = no auth)* | Bearer token for `/generate`. |
| `CPU_OFFLOAD` | pod | `0` | `0` / `1` (or `model`) / `sequential` — see table above. |
| `DEFAULT_STEPS` | pod | `40` | Default inference steps if the client doesn't specify one. |
| `MAX_INPUT_DIM` | pod | `1024` | Input images are downscaled to this max side before processing. |
| `HF_TOKEN` | pod | — | Only needed if the model repo is gated. |
| `HF_HOME` | pod (set by `bootstrap.sh`) | `/workspace/hf_cache` | Where model weights are cached. |
| `RUNPOD_BASE_URL` | frontend | — | The pod's public URL (proxy or direct TCP). |

## What this phase deliberately does *not* do
- No Docker image / container registry.
- No queueing, batching, or concurrent request handling.
- No persistence of generated images beyond the browser download button.
- No retry/backoff logic on the frontend.
- No production-grade auth (a single shared bearer token, or none).

These are fine for a testing/dev setup; they'd need revisiting before this became a real product.
