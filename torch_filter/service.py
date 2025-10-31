# torch_filter/service.py
import os
import io
import base64
import logging
import httpx

import torch
import torchvision.transforms as T

from typing import Optional
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer


log = logging.getLogger("torch-filter")
app = FastAPI(title="torch-filter", version="0.2.1")

# -----------------------
# Environment / Defaults
# -----------------------
IMAGE_TS_PATH = os.getenv("IMAGE_TS", "filters/model.ts")
TEXT_TS_PATH = os.getenv("TEXT_TS", "filters/text_model.ts")
TOK_NAME = os.getenv("TEXT_MODEL", "distilroberta-base")
MAXLEN = int(os.getenv("MAXLEN", "256"))

# -------------
# Load models
# -------------
image_model = None
if os.path.exists(IMAGE_TS_PATH):
    try:
        image_model = torch.jit.load(IMAGE_TS_PATH, map_location="cpu").eval()
        log.info("Loaded image TorchScript from %s", IMAGE_TS_PATH)
    except Exception as e:
        log.warning("Failed to load IMAGE_TS %s: %s", IMAGE_TS_PATH, e)

text_model = None
TOKENIZER = AutoTokenizer.from_pretrained(TOK_NAME, use_fast=True)
if os.path.exists(TEXT_TS_PATH):
    try:
        text_model = torch.jit.load(TEXT_TS_PATH, map_location="cpu").eval()
        log.info("Loaded text TorchScript from %s", TEXT_TS_PATH)
    except Exception as e:
        log.warning("Failed to load TEXT_TS %s: %s", TEXT_TS_PATH, e)

# ----------------
# Preprocessing
# ----------------
img_preproc = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def _tensor_from_image_bytes(b: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(b)).convert("RGB")
    x = img_preproc(img).unsqueeze(0)  # [1,3,224,224]
    return x


def _sigmoid_to_score(logits: torch.Tensor) -> float:
    with torch.no_grad():
        p = torch.sigmoid(logits.float()).item()
    return max(0.0, min(1.0, float(p)))


# -----------
# Schemas
# -----------
class TextIn(BaseModel):
    text: str


# -----------
# Routes
# -----------
@app.get("/health")
def health():
    return {
        "ok": True,
        "image_model": bool(image_model),
        "text_model": bool(text_model),
    }


@app.post("/predict/text")
def predict_text(inp: TextIn):
    if text_model is None:
        raise HTTPException(status_code=503, detail="text model not loaded")

    text = (inp.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty text")

    toks = TOKENIZER(
        text,
        max_length=MAXLEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = toks["input_ids"]
    attention_mask = toks["attention_mask"]

    try:
        logits = text_model(input_ids, attention_mask)  # TorchScript forward
        score = _sigmoid_to_score(logits)
        label = "unsafe_text" if score >= 0.5 else "safe_text"
        return {"score": score, "label": label}
    except Exception as e:
        log.exception("text inference failed: %s", e)
        raise HTTPException(status_code=500, detail="text inference error")


@app.post("/predict/image")
async def predict_image(
    file: UploadFile | None = File(None),
    image_b64: str | None = Form(None),
    image_url: str | None = Form(None),
):
    if image_model is None:
        raise HTTPException(status_code=503, detail="image model not loaded")

    # Get bytes from one of the supported inputs
    data: Optional[bytes] = None
    if file is not None:
        data = await file.read()
    elif image_b64:
        try:
            data = base64.b64decode(image_b64)
        except Exception:
            raise HTTPException(status_code=400, detail="invalid base64")
    elif image_url:
        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as cx:
                r = await cx.get(image_url)
                r.raise_for_status()
                data = r.content
        except Exception:
            raise HTTPException(status_code=400, detail="failed to fetch image_url")
    else:
        raise HTTPException(status_code=400, detail="provide file or image_b64 or image_url")

    # Preprocess & run model
    try:
        x = _tensor_from_image_bytes(data)  # [1,3,224,224]
        logits = image_model(x)             # -> [1] or scalar
        if isinstance(logits, torch.Tensor) and logits.ndim > 0:
            logits = logits.squeeze()
        score = _sigmoid_to_score(torch.as_tensor(logits))
        label = "unsafe_image" if score >= 0.5 else "safe_image"
        return {"score": float(score), "label": label}
    except HTTPException:
        raise
    except Exception as e:
        log.exception("image inference failed: %s", e)
        raise HTTPException(status_code=500, detail="image inference error")
