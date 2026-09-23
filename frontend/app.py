"""
Streamlit frontend for the Qwen-Image-2.1 RunPod server.
Upload an image + prompt -> send to the server -> show the result.

Run:
    streamlit run frontend/app.py
"""
import os

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DEFAULT_API_URL = os.getenv("RUNPOD_BASE_URL", "")   # e.g. https://<pod-id>-8000.proxy.runpod.net
DEFAULT_API_KEY = os.getenv("API_KEY", "")
REQUEST_TIMEOUT = 600.0

st.set_page_config(page_title="Qwen-Image-2.1 Tester", layout="wide")
st.title("Qwen-Image-2.1 Tester")

with st.sidebar:
    st.header("Server")
    api_url = st.text_input("API URL", value=DEFAULT_API_URL, placeholder="https://<POD_ID>-8000.proxy.runpod.net")
    api_key = st.text_input("API key (optional)", value=DEFAULT_API_KEY, type="password")
    if st.button("Check health", disabled=not api_url):
        try:
            r = httpx.get(f"{api_url.rstrip('/')}/health", timeout=15.0)
            st.json(r.json())
        except Exception as exc:
            st.error(f"health check failed: {exc}")

    st.header("Settings")
    steps = st.slider("Inference steps", 10, 60, 40)
    seed_text = st.text_input("Seed (blank = random)", value="")

uploaded = st.file_uploader("Input image", type=["jpg", "jpeg", "png", "webp"])
prompt = st.text_area("Prompt", placeholder="e.g. Remove the cardboard boxes from the floor")

col_in, col_out = st.columns(2)
with col_in:
    st.subheader("Input")
    if uploaded:
        st.image(uploaded)

if st.button("Generate", type="primary", disabled=not (api_url and prompt.strip())):
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
