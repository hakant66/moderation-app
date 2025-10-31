# backend/cascade.py
import os, base64, httpx

FILTER_URL = os.getenv("TORCH_FILTER_URL", "http://localhost:9000")

class TorchFilterError(Exception):
    pass

async def filter_text(text: str) -> float:
    """Returns prob_unsafe ∈ [0,1] from torch-filter."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as cx:
            r = await cx.post(f"{FILTER_URL}/predict/text", json={"text": text})
            r.raise_for_status()
            return float(r.json().get("score", 0.0))
    except Exception as e:
        raise TorchFilterError(str(e))

async def filter_image_bytes(data: bytes) -> float:
    """Returns prob_unsafe ∈ [0,1] from torch-filter."""
    try:
        b64 = base64.b64encode(data).decode("utf-8")
        async with httpx.AsyncClient(timeout=5.0) as cx:
            r = await cx.post(f"{FILTER_URL}/predict/image", data={"image_b64": b64})
            r.raise_for_status()
            return float(r.json().get("score", 0.0))
    except Exception as e:
        raise TorchFilterError(str(e))
