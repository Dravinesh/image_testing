"""
FastAPI server for Qwen/Qwen-Image-2.1 — runs on a RunPod GPU Pod.

Endpoints:
    GET  /health    -> model load status
    POST /generate  -> multipart form (prompt, optional image, optional params) -> PNG bytes

Run: started on the pod by runpod_server/bootstrap.sh (see README).
"""
import io
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
from diffusers import QwenImage21Pipeline
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen-Image-2.1")
API_KEY = os.getenv("API_KEY", "")                      # empty = no auth
CPU_OFFLOAD = os.getenv("CPU_OFFLOAD", "0") == "1"       # enable on smaller GPUs
DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "40"))
MAX_INPUT_DIM = int(os.getenv("MAX_INPUT_DIM", "2048"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("qwen-image-server")

pipe: Optional[QwenImage21Pipeline] = None
pipe_lock = threading.Lock()  # one generation at a time on a single GPU


def load_pipeline() -> QwenImage21Pipeline:
    log.info("loading %s (cpu_offload=%s)...", MODEL_ID, CPU_OFFLOAD)
    t0 = time.time()
    p = QwenImage21Pipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    if CPU_OFFLOAD:
        p.enable_model_cpu_offload()
    else:
        p.to("cuda")

    # Decode the final image in tiles/slices instead of all at once — cuts peak
    # memory during the VAE decode step, which is where an OOM tends to hit
    # right after the denoising loop finishes (steps complete, then "Killed").
    for method in ("enable_vae_slicing", "enable_vae_tiling"):
        try:
            getattr(p, method)()
        except Exception as exc:
            log.warning("%s not available/failed: %s", method, exc)

    log.info("model loaded in %.1fs", time.time() - t0)
    return p


@asynccontextmanager
async def lifespan(_: FastAPI):
    global pipe
    pipe = load_pipeline()
    yield
    pipe = None
    torch.cuda.empty_cache()


app = FastAPI(title="Qwen-Image-2.1 server", lifespan=lifespan)


def check_api_key(authorization: Optional[str] = Header(default=None)) -> None:
    if not API_KEY:
        return
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="invalid or missing API key")


def _load_input_image(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not read image: {exc}")
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_INPUT_DIM:
        scale = MAX_INPUT_DIM / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if pipe is not None else "loading",
        "model": MODEL_ID,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.post("/generate", dependencies=[Depends(check_api_key)])
def generate(
    prompt: str = Form(...),
    image: Optional[UploadFile] = File(default=None),
    num_inference_steps: int = Form(DEFAULT_STEPS),
    seed: Optional[int] = Form(default=None),
    width: Optional[int] = Form(default=None),
    height: Optional[int] = Form(default=None),
) -> Response:
    if pipe is None:
        raise HTTPException(status_code=503, detail="model is still loading")
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is empty")

    kwargs = {"prompt": prompt, "num_inference_steps": num_inference_steps}
    if image is not None and image.filename:
        kwargs["image"] = _load_input_image(image.file.read())
    if width and height:
        kwargs["width"], kwargs["height"] = width, height
    if seed is not None:
        kwargs["generator"] = torch.Generator("cuda").manual_seed(seed)

    mode = "edit" if "image" in kwargs else "t2i"
    log.info("generate mode=%s steps=%d seed=%s prompt=%r", mode, num_inference_steps, seed, prompt[:120])

    t0 = time.time()
    with pipe_lock:
        try:
            result = pipe(**kwargs).images[0]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise HTTPException(status_code=507, detail="GPU out of memory — try smaller size or CPU_OFFLOAD=1")
    elapsed = time.time() - t0
    log.info("done in %.1fs", elapsed)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={"X-Inference-Seconds": f"{elapsed:.1f}", "X-Mode": mode},
    )
