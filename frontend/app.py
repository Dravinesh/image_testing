"""
Streamlit frontend.

Two tabs:
  - "Batch Cleanup" (Phase 2): upload multiple images, each is queued and run
    through the full local pipeline (YOLO person masking + Gemini clutter
    detection -> merged mask -> Qwen-Image-2.1 on RunPod -> composited back
    using the mask). Results are saved into results/ and shown with a tick
    once each image finishes.
  - "Single Test" (Phase 1): raw one-off calls straight to the Qwen server,
    no YOLO/Gemini/masking — useful for testing the RunPod server itself.

Run (from the project root):
    streamlit run frontend/app.py
"""
import io
import os
import sys
from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

# Make sure `pipeline` (project root) is importable regardless of how
# Streamlit was invoked.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline.orchestrator import process_image  # noqa: E402

load_dotenv()

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEFAULT_API_URL = os.getenv("RUNPOD_BASE_URL", "")   # e.g. https://<pod-id>-8000.proxy.runpod.net
DEFAULT_API_KEY = os.getenv("API_KEY", "")
DEFAULT_GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
REQUEST_TIMEOUT = 600.0

st.set_page_config(page_title="Qwen-Image-2.1 Tester", layout="wide")
st.title("Property Photo Cleanup")

with st.sidebar:
    st.header("Server")
    api_url = st.text_input("Qwen API URL", value=DEFAULT_API_URL, placeholder="https://<POD_ID>-8000.proxy.runpod.net")
    api_key = st.text_input("Qwen API key (optional)", value=DEFAULT_API_KEY, type="password")
    gemini_key = st.text_input("Gemini API key", value=DEFAULT_GEMINI_KEY, type="password")
    if st.button("Check health", disabled=not api_url):
        try:
            r = httpx.get(f"{api_url.rstrip('/')}/health", timeout=15.0)
            st.json(r.json())
        except Exception as exc:
            st.error(f"health check failed: {exc}")

    st.header("Settings")
    steps = st.slider("Inference steps", 10, 60, 40)
    seed_text = st.text_input("Seed (blank = random, single-test only)", value="")

tab_batch, tab_single = st.tabs(["Batch Cleanup", "Single Test"])

# ─────────────────────────────────────────────────────────────────────────
# Batch Cleanup (Phase 2): YOLO + Gemini + Qwen, queued, multiple images
# ─────────────────────────────────────────────────────────────────────────
with tab_batch:
    st.caption("Upload one or more photos. Each is queued, then run through YOLO (people) + Gemini (clutter) + Qwen (removal), and saved into `results/`.")

    uploaded_files = st.file_uploader(
        "Input images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="batch_uploader",
    )

    # Rebuild the queue in session_state whenever the uploaded file set changes,
    # but keep prior statuses/results for files that are still in the set
    # (so re-running doesn't lose already-processed items).
    if uploaded_files:
        current_names = [f.name for f in uploaded_files]
        if "batch_queue" not in st.session_state or st.session_state.get("batch_names") != current_names:
            queue = []
            for f in uploaded_files:
                queue.append({
                    "name": f.name,
                    "bytes": f.getvalue(),
                    "status": "pending",   # pending -> processing -> done | error | skipped
                    "result_bytes": None,
                    "info": None,
                    "error": None,
                })
            st.session_state.batch_queue = queue
            st.session_state.batch_names = current_names

        queue = st.session_state.batch_queue

        cols = st.columns(4)
        placeholders = []
        for i, item in enumerate(queue):
            with cols[i % 4]:
                ph = st.empty()
                placeholders.append(ph)

        def render_item(ph, item):
            icon = {"pending": "⏳", "processing": "🔄", "done": "✅", "skipped": "➖", "error": "❌"}[item["status"]]
            with ph.container():
                st.image(item["result_bytes"] or item["bytes"], caption=f"{icon} {item['name']}", use_container_width=True)
                if item["status"] == "error":
                    st.caption(f"error: {item['error']}")
                if item["info"]:
                    labels = item["info"].get("labels") or []
                    bits = []
                    if item["info"].get("person_found"):
                        bits.append("person")
                    bits.extend(labels)
                    if bits:
                        st.caption(("detected: " if item["status"] == "error" else "removed: ") + ", ".join(bits))
                    elif item["status"] == "skipped":
                        st.caption("nothing flagged — left unchanged")
                    if item["status"] == "error" and item["info"].get("preview_dir"):
                        st.caption(f"YOLO/Gemini output saved — check `{item['info']['preview_dir']}`")

        for ph, item in zip(placeholders, queue):
            render_item(ph, item)

        # Only Gemini is required to start — the Qwen URL is optional here on
        # purpose, so YOLO + Gemini can be verified (via mask_previews/) even
        # before the RunPod pod is connected; process_image() will raise a
        # clear "Qwen on RunPod is not connected" error at that point instead.
        disabled = not gemini_key or all(i["status"] != "pending" for i in queue)
        if not api_url:
            st.caption("⚠️ No Qwen API URL set — YOLO + Gemini will still run and save previews to `mask_previews/`, but each image will end in a 'Qwen on RunPod is not connected' error.")
        if st.button("Process Queue", type="primary", disabled=disabled):
            for ph, item in zip(placeholders, queue):
                if item["status"] != "pending":
                    continue
                item["status"] = "processing"
                render_item(ph, item)
                try:
                    img = Image.open(io.BytesIO(item["bytes"]))
                    result, info = process_image(
                        img, gemini_key, api_url, api_key,
                        num_inference_steps=steps, name=item["name"],
                    )
                    buf = io.BytesIO()
                    result.save(buf, format="PNG")
                    item["result_bytes"] = buf.getvalue()
                    item["info"] = info
                    item["status"] = "skipped" if info.get("skipped") else "done"

                    out_path = RESULTS_DIR / f"{Path(item['name']).stem}_cleaned.png"
                    result.save(out_path)
                except Exception as exc:
                    item["status"] = "error"
                    item["error"] = str(exc)
                    item["info"] = getattr(exc, "debug_info", None)
                render_item(ph, item)
            st.success(f"Queue finished. Results saved in `{RESULTS_DIR}`.")

        if not gemini_key:
            st.info("Enter a Gemini API key in the sidebar to enable processing.")
    else:
        st.info("Upload one or more images to build the queue.")

# ─────────────────────────────────────────────────────────────────────────
# Single Test (Phase 1): raw Qwen call, no YOLO/Gemini/masking
# ─────────────────────────────────────────────────────────────────────────
with tab_single:
    st.caption("Sends the image straight to Qwen with your own prompt — no YOLO/Gemini/masking. Useful for testing the RunPod server directly.")

    uploaded = st.file_uploader("Input image", type=["jpg", "jpeg", "png", "webp"], key="single_uploader")
    prompt = st.text_area("Prompt", placeholder="e.g. Remove the cardboard boxes from the floor")

    col_in, col_out = st.columns(2)
    with col_in:
        st.subheader("Input")
        if uploaded:
            st.image(uploaded)

    if st.button("Generate", type="primary", disabled=not (api_url and prompt.strip()), key="single_generate"):
        data = {"prompt": prompt, "num_inference_steps": str(steps)}
        if seed_text.strip():
            data["seed"] = seed_text.strip()
        files = {"image": (uploaded.name, uploaded.getvalue(), uploaded.type)} if uploaded else None
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

        with col_out:
            st.subheader("Result")
            with st.spinner("Generating on RunPod..."):
                try:
                    resp = httpx.post(
                        f"{api_url.rstrip('/')}/generate",
                        data=data,
                        files=files,
                        headers=headers,
                        timeout=REQUEST_TIMEOUT,
                    )
                except Exception as exc:
                    st.error(f"request failed: {exc}")
                    st.stop()

            if resp.status_code != 200:
                st.error(f"server error {resp.status_code}: {resp.text}")
                st.stop()

            st.image(resp.content)
            st.caption(f"mode: {resp.headers.get('X-Mode')} · inference: {resp.headers.get('X-Inference-Seconds')}s")
            st.download_button("Download PNG", resp.content, file_name="qwen_result.png", mime="image/png")
