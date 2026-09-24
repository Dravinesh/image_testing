"""
Phase 2 local orchestration pipeline.

Runs entirely on the local machine, except the single call to the
Qwen-Image-2.1 server on RunPod:

  1. YOLO (yolov8n-seg.pt, local file) detects and masks people.
  2. Gemini detects other removable clutter/trash/temporary objects and
     masks them (bounding boxes -> filled rectangles).
  3. The two masks are merged (union) and dilated slightly.
  4. Qwen-Image-2.1 (on RunPod) is asked, via a text prompt describing what
     to remove, to edit the WHOLE image — it has no mask/inpaint input of
     its own (confirmed against the actual diffusers pipeline source: only
     `prompt` + `image`, no `mask_image` parameter).
  5. Qwen's output is composited back onto the ORIGINAL image using the
     merged mask (with feathered edges), so only the masked regions
     actually change and everything else stays pixel-identical — this is
     what makes step 4's whole-image edit behave like masked inpainting.

If nothing is detected in steps 1-2, the Qwen call is skipped entirely and
the original image is returned unchanged.

Steps 1-3 (YOLO + Gemini + merge) always run and their output is saved to
MASK_PREVIEW_DIR before step 4 is attempted — so you can verify YOLO/Gemini
are working even if the Qwen pod isn't connected yet.
"""
import base64
import io
import json
import time
from pathlib import Path
from typing import Optional

import cv2
import httpx
import numpy as np
from PIL import Image, ImageFilter
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
YOLO_MODEL_PATH = PROJECT_ROOT / "yolov8n-seg.pt"
YOLO_REMOVE_CLASSES = [0]  # COCO class 0 = person

# Where step 1-3 output (masks + an overlay preview) is saved, so YOLO/Gemini
# can be checked independently of whether the Qwen pod is reachable.
MASK_PREVIEW_DIR = PROJECT_ROOT / "mask_previews"


class QwenNotConnectedError(RuntimeError):
    """Raised when the Qwen/RunPod server isn't configured or reachable —
    distinct from other errors (bad response, OOM, etc.) so the frontend can
    show a clear, specific message instead of a raw connection traceback."""
    pass

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
# gemini-2.5-flash is now legacy (restricted to accounts that used it before)
# and 404s for new/unrelated keys — gemini-3.5-flash is its current
# same-tier successor (fast/cheap, matching the original choice of "flash"
# over "pro"). gemini-3.8-flash is available if higher detection accuracy
# is worth the extra cost.
GEMINI_DETECT_MODEL = "gemini-3.5-flash"
GEMINI_MAX_DIM = 1024

# Feather (Gaussian blur) radius applied to the mask before compositing, in
# pixels — smooths the seam between Qwen's edit and the untouched original,
# since Qwen isn't natively mask-conditioned like a true inpainting model.
MASK_FEATHER_PX = 15

GEMINI_DETECT_PROMPT = """
You are analyzing a real estate property photo. Identify every item that does not belong to the property and should be removed to make the listing look clean and professional.
The photo may be an indoor room, an outdoor building exterior, or an open plot of land.

IMPORTANT — size in frame does not matter:
Flag all non-property items regardless of how large they appear. A jute bag or cardboard box filling 30% of the frame must still be flagged — large size does not mean it belongs there.

IMPORTANT — cardboard boxes and packaging are NEVER part of a property:
Cardboard boxes, corrugated boxes, large jute sacks, gunny bags, and packaging materials of any size must always be flagged without exception. They are temporary items regardless of where they are placed or how they are arranged.

Return a JSON array in this exact format:
[{"label": "item description", "box": [y_min, x_min, y_max, x_max]}]

Coordinates are normalized 0 to 1000. Box format: [y_min, x_min, y_max, x_max].

Flag an item if it meets ANY of these criteria:
- It is a person (occupant, worker, visitor — even partially visible)
- It is waste, garbage, or litter (bags, polythene, packaging, bottles)
- It was temporarily placed and is not part of the property (boxes, sacks, tools, cleaning equipment, construction material)
- It is visible dirt, stain, or debris sitting ON a floor, wall, or ground surface
- It makes the photo look untidy or unprofessional for a property listing

Do NOT flag items that are a permanent or semi-permanent part of the property:
- Building structure (walls, ceilings, floors, doors, windows, pillars, staircases, railings, grills, balconies)
- Furniture and staging (sofas, chairs, tables, beds, wardrobes)
- Electronics and appliances (TVs, fans, AC units, refrigerators, lights, switches)
- Permanent fixtures (cabinets, shelves, counters, built-ins)
- Outdoor structures (boundary walls, gates, fences)
- Vegetation (trees, plants, grass)
- The floor or ground surface itself — only flag dirt or debris ON it, not the surface

Return an empty array [] if nothing needs to be removed.
Return ONLY the JSON array, no explanation or other text.
"""

_yolo_model: Optional[YOLO] = None


def _get_yolo_model() -> YOLO:
    global _yolo_model
    if _yolo_model is None:
        if not YOLO_MODEL_PATH.exists():
            raise FileNotFoundError(f"YOLO model not found at {YOLO_MODEL_PATH}")
        _yolo_model = YOLO(str(YOLO_MODEL_PATH))
    return _yolo_model


def yolo_person_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Step 1: YOLO segmentation mask for people."""
    h, w = image_bgr.shape[:2]
    model = _get_yolo_model()
    results = model(image_bgr, verbose=False)
    mask = np.zeros((h, w), dtype=np.uint8)

    for r in results:
        if r.masks is None:
            continue
        for i, seg in enumerate(r.masks.xy):
            if int(r.boxes.cls[i]) in YOLO_REMOVE_CLASSES:
                pts = np.array(seg, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)

    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


def _to_gemini_b64(image_bgr: np.ndarray) -> str:
    h, w = image_bgr.shape[:2]
    img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    if max(w, h) > GEMINI_MAX_DIM:
        scale = GEMINI_MAX_DIM / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _parse_gemini_json(text: str) -> list:
    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


# Status codes worth retrying: 503/500/502/504 are Google's own "temporarily
# unavailable/overloaded" family, and 429 is rate-limiting — both are
# typically transient, unlike e.g. a 400 (bad request) or 403 (bad key).
GEMINI_RETRY_STATUS = {429, 500, 502, 503, 504}
GEMINI_MAX_ATTEMPTS = 4


def gemini_clutter_mask(gemini_key: str, image_bgr: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Step 2: Gemini bounding-box detection of non-property items.
    Returns (mask, detected_labels)."""
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    b64 = _to_gemini_b64(image_bgr)
    payload = {
        "contents": [{
            "parts": [
                {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                {"text": GEMINI_DETECT_PROMPT},
            ]
        }],
        "generationConfig": {"responseMimeType": "application/json"},
    }

    resp = None
    for attempt in range(1, GEMINI_MAX_ATTEMPTS + 1):
        resp = httpx.post(
            f"{GEMINI_BASE_URL}/{GEMINI_DETECT_MODEL}:generateContent",
            params={"key": gemini_key},
            json=payload,
            timeout=120.0,
        )
        if resp.status_code not in GEMINI_RETRY_STATUS or attempt == GEMINI_MAX_ATTEMPTS:
            break
        time.sleep(2 ** attempt)  # 2s, 4s, 8s between attempts 1-2, 2-3, 3-4
    resp.raise_for_status()

    text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    detections = _parse_gemini_json(text)

    labels: list[str] = []
    if not detections:
        return mask, labels

    for item in detections:
        box = item.get("box", [])
        if len(box) != 4:
            continue
        labels.append(str(item.get("label", "object")))
        y0 = max(0, int(box[0] / 1000 * h))
        x0 = max(0, int(box[1] / 1000 * w))
        y1 = min(h, int(box[2] / 1000 * h))
        x1 = min(w, int(box[3] / 1000 * w))
        cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask, labels


def save_mask_previews(
    name: str,
    image: Image.Image,
    person_mask: np.ndarray,
    clutter_mask: np.ndarray,
    merged_mask: np.ndarray,
    labels: list[str],
) -> Path:
    """Saves YOLO/Gemini's raw output — independent of whether the Qwen call
    that follows succeeds — so those two steps can be checked on their own.
    Returns the directory the files were written into."""
    MASK_PREVIEW_DIR.mkdir(exist_ok=True)
    stem = Path(name).stem

    Image.fromarray(person_mask).save(MASK_PREVIEW_DIR / f"{stem}_1_yolo_person_mask.png")
    Image.fromarray(clutter_mask).save(MASK_PREVIEW_DIR / f"{stem}_2_gemini_clutter_mask.png")
    Image.fromarray(merged_mask).save(MASK_PREVIEW_DIR / f"{stem}_3_merged_mask.png")

    # Original image with the merged mask drawn as a translucent red overlay
    # — the fastest way to eyeball whether the right regions got flagged.
    overlay = image.convert("RGBA")
    red_layer = Image.new("RGBA", image.size, (255, 0, 0, 0))
    alpha = Image.fromarray(merged_mask).convert("L").point(lambda p: 120 if p > 0 else 0)
    red_layer.putalpha(alpha)
    Image.alpha_composite(overlay, red_layer).convert("RGB").save(
        MASK_PREVIEW_DIR / f"{stem}_4_overlay.png"
    )

    (MASK_PREVIEW_DIR / f"{stem}_labels.txt").write_text(
        "\n".join(labels) if labels else "(none detected)", encoding="utf-8"
    )

    return MASK_PREVIEW_DIR


# Gemini can independently label a person too (e.g. "person", "man on
# ladder") — when YOLO already flagged people, drop those from Gemini's list
# so the prompt doesn't repeat itself (was producing "any people, person").
_PERSON_SYNONYMS = {"person", "people", "human", "man", "woman", "occupant", "worker", "visitor"}


def build_removal_prompt(gemini_labels: list[str], person_detected: bool) -> str:
    """Step 4 prep: since Qwen has no mask input, tell it in plain language
    what's already been marked for removal. Qwen's only job here is to fill
    those areas in naturally — it doesn't decide what to remove, the mask
    (from steps 1-3) already did, and composite() enforces that boundary
    afterward regardless of what Qwen does elsewhere in the image."""
    other_items = sorted({
        label for label in gemini_labels
        if not (person_detected and label.strip().lower() in _PERSON_SYNONYMS)
    })
    items = (["people"] if person_detected else []) + other_items
    items_text = ", ".join(items) if items else "clutter, trash, and temporary objects"

    return (
        "PROPERTY LISTING PHOTO CLEANUP.\n"
        f"The following have already been marked for removal: {items_text}.\n"
        "Your only task is to fill in those marked areas naturally — match "
        "the exact texture, color, pattern, lighting, and shadow of the "
        "immediately surrounding area, so it looks like they were never "
        "there. Do not add, remove, or restyle anything else.\n"
        "Keep everything else in the photo exactly unchanged: the building "
        "structure, walls, floors, furniture, fixtures, windows, doors, "
        "vegetation, and the overall composition, perspective, and lighting "
        "must remain identical.\n"
        "Also improve overall brightness, sharpness, and color accuracy."
    )


def call_qwen(
    base_url: str,
    api_key: str,
    image: Image.Image,
    prompt: str,
    num_inference_steps: int = 40,
    timeout: float = 600.0,
) -> Image.Image:
    """Step 4: send the full image + removal prompt to the Qwen server."""
    if not base_url or not base_url.strip() or "<pod-id>" in base_url:
        raise QwenNotConnectedError("Qwen on RunPod is not connected")

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    try:
        resp = httpx.post(
            f"{base_url.rstrip('/')}/generate",
            data={"prompt": prompt, "num_inference_steps": str(num_inference_steps)},
            files={"image": ("input.jpg", buf.getvalue(), "image/jpeg")},
            headers=headers,
            timeout=timeout,
        )
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        raise QwenNotConnectedError("Qwen on RunPod is not connected") from exc

    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def composite(original: Image.Image, edited: Image.Image, mask: np.ndarray) -> Image.Image:
    """Step 5: keep `original` pixels outside `mask`; use `edited` pixels
    inside it, with feathered edges for a smoother seam. This is what makes
    Qwen's whole-image edit behave like masked inpainting."""
    if edited.size != original.size:
        edited = edited.resize(original.size, Image.LANCZOS)
    mask_img = Image.fromarray(mask).convert("L").resize(original.size, Image.LANCZOS)
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(MASK_FEATHER_PX))
    return Image.composite(edited, original, mask_img)


def process_image(
    image: Image.Image,
    gemini_key: str,
    runpod_base_url: str,
    runpod_api_key: str,
    num_inference_steps: int = 40,
    name: str = "image",
) -> tuple[Image.Image, dict]:
    """Runs the full pipeline on one already-opened PIL image.
    Returns (result_image, debug_info). `name` is only used to name the
    saved preview files (see MASK_PREVIEW_DIR)."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    person_mask = yolo_person_mask(image_bgr)
    person_found = bool((person_mask > 0).any())

    clutter_mask, labels = gemini_clutter_mask(gemini_key, image_bgr)

    merged_mask = cv2.bitwise_or(person_mask, clutter_mask)

    # Save YOLO/Gemini output BEFORE the Qwen call, so it's there to inspect
    # even if that call fails (e.g. the pod isn't connected yet).
    preview_dir = save_mask_previews(name, image, person_mask, clutter_mask, merged_mask, labels)

    debug_info = {"person_found": person_found, "labels": labels, "preview_dir": str(preview_dir)}

    if not merged_mask.any():
        # Nothing flagged — skip the Qwen call and return the original as-is.
        debug_info["skipped"] = True
        return image, debug_info

    debug_info["skipped"] = False
    prompt = build_removal_prompt(labels, person_found)
    try:
        edited = call_qwen(runpod_base_url, runpod_api_key, image, prompt, num_inference_steps)
    except Exception as exc:
        # YOLO/Gemini already ran and their output is saved — attach that
        # info to the exception so callers (e.g. the frontend) can still show
        # it even though the pipeline didn't finish.
        exc.debug_info = debug_info  # type: ignore[attr-defined]
        raise
    result = composite(image, edited, merged_mask)

    return result, debug_info
