# Local Setup — running the frontend + pipeline on your own machine

This covers everything that runs **locally**: the Streamlit frontend, and the Phase 2 Gemini + Qwen pipeline. For deploying the Qwen model itself on RunPod, see **[RUNPOD_SETUP.md](RUNPOD_SETUP.md)**. For how the pieces fit together, see **[HOWITWORKS.md](HOWITWORKS.md)**.

## Prerequisites
- Python 3.9+ (check with `py --version`)
- Git
- The RunPod pod already deployed and running (see RUNPOD_SETUP.md) — you'll need its URL
- `yolov8n-seg.pt` present at the project root (already included in this repo's working copy; it's git-ignored since it's a large binary, so if you're on a fresh clone you'll need to get this file separately)

---

## 1. Get the code
If you haven't already:
```powershell
git clone https://github.com/Dravinesh/image_testing.git
cd image_testing
```

## 2. Create and activate a virtual environment

A dedicated environment keeps this project's packages (streamlit, ultralytics, opencv, etc.) separate from anything else installed on your machine — installing them globally earlier caused a version conflict with an existing package, which this avoids.

```powershell
py -m venv venv
```

Activate it (do this every time you open a new terminal to work on this project):
```powershell
.\venv\Scripts\Activate.ps1
```
Your prompt should now start with `(venv)`. If PowerShell blocks the script with an execution-policy error, run this once, then retry:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

To leave the environment later: `deactivate`.

## 3. Install dependencies
With the venv active:
```powershell
pip install -r frontend/requirements.txt
```
This installs Streamlit, httpx, python-dotenv, and the Phase 2 pipeline's deps (`ultralytics`, `opencv-python-headless`, `numpy`). The first install downloads a fair amount (opencv and ultralytics aren't tiny) — give it a few minutes.

## 4. Get a Gemini API key
Used by the pipeline to detect clutter/objects to remove (`pipeline/orchestrator.py`).

1. Go to **[Google AI Studio](https://aistudio.google.com/apikey)**.
2. Sign in with a Google account.
3. Click **Create API key** (choose or create a Google Cloud project if prompted — the free tier is enough for testing).
4. Copy the key (starts with `AIza...`).

Gemini's free tier has rate limits; if you process a large batch quickly you may see `429` errors — space out requests if so.

## 5. Set up `.env`
```powershell
copy .env.example .env
```
Open `.env` and fill in:
```
RUNPOD_BASE_URL=https://<your-pod-id>-8000.proxy.runpod.net
API_KEY=<same value you set as API_KEY on the pod, or leave blank if you didn't set one>
GEMINI_API_KEY=<the key from step 4>
```
Leave the other variables (`MODEL_ID`, `CPU_OFFLOAD`, etc.) as-is — those describe the pod's own settings for reference; they aren't read by the frontend.

Never commit `.env` — it's already in `.gitignore`.

## 6. (Optional) Confirm the YOLO model file is present
The current pipeline (Gemini writes the removal prompt directly) doesn't use YOLO, so this file isn't required for normal use. It's only needed if you call the legacy `process_image_masked()` pipeline (see `pipeline/orchestrator.py`) directly.
```powershell
Test-Path yolov8n-seg.pt
```
If you do need it, it must sit at the project root (same level as `README.md`), not inside `pipeline/` or `frontend/`.

## 7. Run the frontend
```powershell
streamlit run frontend/app.py
```
Opens `http://localhost:8501` (or the next free port if that one's taken). The sidebar is pre-filled from `.env`. Click **Check health** to confirm the pod is reachable before doing anything else.

---

## Using the app

### Batch Cleanup tab (Phase 2 — the real pipeline)
1. Upload one or more photos.
2. Click **Process Queue**. Each image runs through Gemini (looks at the whole photo, decides what to remove, and writes a short 2-3 sentence removal prompt itself) → Qwen (edits the image using that prompt), in order — one at a time, matching the pod's single-GPU limit. Qwen's output is the final result directly — no mask, no compositing.
3. Each thumbnail's status icon updates live: ⏳ pending → 🔄 processing → ✅ done (something was removed) / ➖ skipped (Gemini flagged nothing) / ❌ error.
4. Finished results are saved into the **`results/`** folder at the project root, named `<original-filename>_cleaned.png`.

**Testing Gemini without the pod connected:** only a Gemini API key is required to click **Process Queue** — the Qwen URL is optional. Gemini's step always runs and saves its generated prompt to **`mask_previews/gemini_prompt/<name>.txt`** (or a note that nothing was flagged), so it can be checked even before the pod is connected. If the Qwen URL isn't set (or the pod isn't reachable), each image ends with a `❌ Qwen on RunPod is not connected` status — that's expected in this case, not a bug.

*A previous version of this pipeline used YOLO (local person detection) + Gemini (bounding boxes) merged into a pixel mask, burned into the image as an overlay, with Qwen's output composited back against that mask. That code is still in `pipeline/orchestrator.py` (as `process_image_masked()` and its helpers) but isn't used by the frontend by default — the current approach lets Gemini decide and describe removals directly instead.*

### Single Test tab (Phase 1 — raw Qwen only)
Sends your image + your own typed prompt straight to the Qwen server, no Gemini involved. Useful for testing the pod itself in isolation.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'ultralytics'` (or `cv2`, `streamlit`, ...) | The venv isn't activated, or step 3 wasn't run. Activate it (`.\venv\Scripts\Activate.ps1`) and re-run `pip install -r frontend/requirements.txt`. |
| `FileNotFoundError: YOLO model not found at ...\yolov8n-seg.pt` | Only happens if you call the legacy `process_image_masked()` directly — the file isn't at the project root, see step 6. Not needed for normal Batch Cleanup use. |
| Batch Cleanup's **Process Queue** button stays disabled | A Gemini API key is required — fill it in in the sidebar. The Qwen URL is *not* required to start (see above). |
| Every image ends in `❌ Qwen on RunPod is not connected` | Expected if the Qwen API URL is blank or the pod isn't reachable — Gemini still ran; check `mask_previews/gemini_prompt/`. To actually generate results, fill in `RUNPOD_BASE_URL` and confirm the pod is up (RUNPOD_SETUP.md). |
| `server error ...` in either tab | The pod isn't reachable or isn't healthy — click **Check health** first, and see RUNPOD_SETUP.md's error log if it fails. |
| Gemini returns `429` / rate limit errors | You're on Gemini's free tier and sent requests too fast — wait a bit, or space out batch runs. |
| Results look odd at the mask edges (a visible seam) | Expected to some degree — Qwen isn't natively mask-conditioned, so `pipeline/orchestrator.py` blends its whole-image edit back onto the original using a feathered mask (`MASK_FEATHER_PX` in that file) rather than true inpainting. Increase the feather radius there if seams are too sharp. |
| Executing scripts is disabled on this system (activating the venv) | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then retry `.\venv\Scripts\Activate.ps1`. |

## Updating later
```powershell
git pull
.\venv\Scripts\Activate.ps1
pip install -r frontend/requirements.txt   # only needed if requirements changed
streamlit run frontend/app.py
```
