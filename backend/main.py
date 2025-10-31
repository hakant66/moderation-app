# backend/main.py
import os
import base64
import hashlib
import asyncio
import time
import threading
import logging
from typing import Any, Dict, List, Literal, Optional, Tuple

import yaml
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI, RateLimitError

from rate_limit import bucket  # token bucket to smooth bursts
from config import settings    # centralized config (.env -> settings)

# -----------------------
# Logging & feature flags
# -----------------------
log = logging.getLogger("uvicorn.error")

# PowerShell: $env:DISABLE_OPENAI = "1"
DISABLE_OPENAI = str(os.getenv("DISABLE_OPENAI", "0")).lower() in {"1", "true", "yes"}
if DISABLE_OPENAI:
    log.warning("OpenAI moderation calls are DISABLED via DISABLE_OPENAI env var.")

# ------------------------
# OpenAI client (resilient)
# ------------------------
if not settings.openai_api_key and not DISABLE_OPENAI:
    raise RuntimeError("OPENAI_API_KEY is required (set in .env or environment).")

client: Optional[OpenAI] = (
    OpenAI(api_key=settings.openai_api_key, max_retries=3, timeout=30.0)
    if (not DISABLE_OPENAI and settings.openai_api_key)
    else None
)

# -----------------
# Policy & Settings
# -----------------
_default_policy = os.path.join(os.path.dirname(__file__), "policy.yaml")
policy_path = settings.policy_file if os.path.exists(settings.policy_file) else _default_policy
with open(policy_path, "r", encoding="utf-8") as f:
    POLICY = yaml.safe_load(f) or {}

FILTER_URL = settings.torch_filter_url
FILTER_BLOCK = float(settings.filter_block)
FILTER_ALLOW = float(settings.filter_allow)
EARLY_ALLOW = bool(settings.filter_early_allow)

# ------------------------
# Tiny in-memory cache (TTL)
# ------------------------
CACHE_TTL = float(os.getenv("CACHE_TTL", "180"))   # seconds
CACHE_MAX = int(os.getenv("CACHE_MAX", "1000"))    # max entries
_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_CACHE_LOCK = threading.Lock()

def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    now = time.time()
    with _CACHE_LOCK:
        item = _CACHE.get(key)
        if not item:
            return None
        exp, val = item
        if exp < now:
            _CACHE.pop(key, None)
            return None
        return val

def _cache_set(key: str, value: Dict[str, Any], ttl: float = CACHE_TTL) -> None:
    exp = time.time() + ttl
    with _CACHE_LOCK:
        if len(_CACHE) >= CACHE_MAX:
            try:
                _CACHE.pop(next(iter(_CACHE)))
            except StopIteration:
                pass
        _CACHE[key] = (exp, value)

# -----
# App
# -----
app = FastAPI(title="Moderation API Demo", version="0.6.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------
# Models
# -------
class TextIn(BaseModel):
    text: str

class DecisionOut(BaseModel):
    decision: Literal["allow", "warn", "block", "ban", "support"]
    categories: List[str]
    scores: Dict[str, float]
    reasons: List[str]
    actions: List[str]
    hash: str

# --------
# Helpers
# --------
def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def sha256_of_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

async def torch_filter_text(text: str) -> Optional[float]:
    """Call PyTorch prefilter for text; returns probability or None on failure."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as cx:
            r = await cx.post(f"{FILTER_URL}/predict/text", json={"text": text})
            r.raise_for_status()
            return float(r.json().get("score", 0.0))
    except Exception:
        return None

async def torch_filter_image_bytes(data: bytes) -> Optional[float]:
    """Call PyTorch prefilter for image; returns probability or None on failure."""
    try:
        b64 = base64.b64encode(data).decode("utf-8")
        async with httpx.AsyncClient(timeout=5.0) as cx:
            r = await cx.post(f"{FILTER_URL}/predict/image", data={"image_b64": b64})
            r.raise_for_status()
            return float(r.json().get("score", 0.0))
    except Exception:
        return None

def normalize_categories(raw: Dict[str, Any]) -> Dict[str, float]:
    """
    Normalize OpenAI Moderation (or neutral) response into {category: score}.
    Supports 'category_scores' (preferred) and boolean 'categories' fallback.
    """
    results = raw.get("results") or []
    if not results:
        return {}
    r0 = results[0]
    scores: Dict[str, float] = {}
    if isinstance(r0.get("category_scores"), dict):
        for k, v in r0["category_scores"].items():
            scores[k] = float(v)
    elif isinstance(r0.get("categories"), dict):
        for k, v in r0["categories"].items():
            scores[k] = 1.0 if v else 0.0
    return scores

def apply_policy(scores: Dict[str, float]) -> Dict[str, Any]:
    """
    Map category scores to a decision and suggested actions using POLICY rules.
    FIX: Only soft-warn when we actually have scores; never soft-warn on empty scores.
    """
    rules = POLICY.get("rules", {})
    triggered: List[str] = []
    highest_action = "allow"
    order = {"allow": 0, "warn": 1, "support": 2, "block": 3, "ban": 4}

    # Hard triggers: only consider categories that actually have a score
    for cat, rule in rules.items():
        if cat not in scores:
            continue
        s = scores[cat]
        th = float(rule.get("threshold", 1.0))
        if s >= th:
            triggered.append(cat)
            act = rule.get("action", "warn")
            if order[act] > order[highest_action]:
                highest_action = act

    # Soft warning if near any threshold (within 0.05) — ONLY if we have any scores
    if not triggered and scores:
        for cat, rule in rules.items():
            if cat not in scores:
                continue
            th = float(rule.get("threshold", 1.0))
            if scores[cat] >= max(th - 0.05, 0.0):
                highest_action = "warn" if order["warn"] > order[highest_action] else highest_action
                break

    reasons = [f"{c}≥{rules[c]['threshold']} ({scores[c]:.2f})" for c in triggered]
    actions: List[str] = []
    if highest_action in ("block", "ban"):
        actions += ["reject_upload", "notify_user"]
        if highest_action == "ban":
            actions += ["escalate_review", "restrict_account"]
    elif highest_action == "support":
        actions += ["show_support_interstitial", "escalate_review"]
    elif highest_action == "warn":
        actions += ["blur_media", "age_gate"]

    return {
        "decision": highest_action,
        "categories": sorted(scores.keys()),
        "scores": scores,
        "reasons": reasons,
        "actions": actions,
    }

async def fetch_bytes_from_url(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as cx:
        r = await cx.get(url)
        r.raise_for_status()
        return r.content

# ---------------------------
# OpenAI Backoff / or Neutral
# ---------------------------
async def call_moderation_with_backoff(text_or_input, attempts: int = 4):
    """
    Calls OpenAI Moderations with exponential backoff on 429s.
    If DISABLE_OPENAI is set, returns a neutral result (no categories).
    """
    if DISABLE_OPENAI:
        # Neutral, empty scores; policy likely returns "allow" unless torch prefilter blocked.
        return {"results": [{"category_scores": {}}]}

    if client is None:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")

    delay = 0.5
    for i in range(attempts):
        try:
            return client.moderations.create(
                model="omni-moderation-latest",
                input=text_or_input,
            )
        except RateLimitError as e:
            if i == attempts - 1:
                raise HTTPException(status_code=429, detail="OpenAI rate limit exceeded, please retry") from e
            await asyncio.sleep(delay)
            delay = delay * 2 + 0.2  # jitter-ish
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"OpenAI moderation failed: {e}") from e

# -------
# Routes
# -------
@app.get("/api/health")
def health():
    return {
        "ok": True,
        "openai_disabled": DISABLE_OPENAI,
        "cache_size": len(_CACHE),
    }

@app.post("/api/moderate/text", response_model=DecisionOut)
async def moderate_text(inp: TextIn):
    text = (inp.text or "").strip()
    if not text:
        raise HTTPException(400, "empty text")

    # ---- Cache (early return)
    text_hash = sha256_of_text(text)
    cache_key = f"text:{text_hash}"
    cached = _cache_get(cache_key)
    if cached:
        return cached  # FastAPI will validate/serialize to DecisionOut

    # 1) Early allow/block via PyTorch prefilter
    p_torch = await torch_filter_text(text)
    if p_torch is not None:
        if p_torch >= FILTER_BLOCK:
            resp = DecisionOut(
                decision="block",
                categories=["torch_filter"],
                scores={"torch_filter": p_torch},
                reasons=[f"torch filter >= {FILTER_BLOCK}"],
                actions=["reject_upload"],
                hash=text_hash,
            )
            _cache_set(cache_key, resp.model_dump())
            return resp
        if EARLY_ALLOW and p_torch <= FILTER_ALLOW:
            resp = DecisionOut(
                decision="allow",
                categories=["torch_filter"],
                scores={"torch_filter": p_torch},
                reasons=[f"torch filter <= {FILTER_ALLOW}"],
                actions=[],
                hash=text_hash,
            )
            _cache_set(cache_key, resp.model_dump())
            return resp

    # 2) OpenAI Moderation (rate-limited + retries/backoff) OR neutral if disabled
    bucket.acquire()  # smooth bursts
    res = await call_moderation_with_backoff(text)

    raw = res.model_dump() if hasattr(res, "model_dump") else res
    scores = normalize_categories(raw)
    decision = apply_policy(scores)
    resp = DecisionOut(**decision, hash=text_hash)
    _cache_set(cache_key, resp.model_dump())
    return resp

@app.post("/api/moderate/image", response_model=DecisionOut)
async def moderate_image(
    file: UploadFile = File(None),
    image_url: Optional[str] = Form(None)
):
    if not file and not image_url:
        raise HTTPException(400, "provide file or image_url")

    # ---- Cache by URL (cheap)
    url_cache_key: Optional[str] = None
    if image_url:
        url_cache_key = f"img:url:{image_url}"
        cached = _cache_get(url_cache_key)
        if cached:
            return cached

    data: Optional[bytes] = None
    if file:
        data = await file.read()
        content_type = file.content_type or "image/jpeg"
        b64 = base64.b64encode(data).decode("utf-8")
        data_url = f"data:{content_type};base64,{b64}"
        oai_input = [{"type": "input_image", "image_url": {"url": data_url}}]
        sample_hash = sha256_of_bytes(data)
        sha_cache_key = f"img:sha:{sample_hash}"
        cached = _cache_get(sha_cache_key)
        if cached:
            return cached
    else:
        oai_input = [{"type": "input_image", "image_url": {"url": image_url}}]
        try:
            data = await fetch_bytes_from_url(image_url)  # for prefilter + robust hashing
            sample_hash = sha256_of_bytes(data)
        except Exception:
            data = None
            sample_hash = sha256_of_text((image_url or "")[:128])
        sha_cache_key = f"img:sha:{sample_hash}"
        cached = _cache_get(sha_cache_key)
        if cached:
            if url_cache_key:
                _cache_set(url_cache_key, cached)
            return cached

    # 1) Early allow/block via PyTorch prefilter
    p_torch = await torch_filter_image_bytes(data) if data is not None else None
    if p_torch is not None:
        if p_torch >= FILTER_BLOCK:
            resp = DecisionOut(
                decision="block",
                categories=["torch_filter"],
                scores={"torch_filter": p_torch},
                reasons=[f"torch filter >= {FILTER_BLOCK}"],
                actions=["reject_upload"],
                hash=sample_hash,
            )
            payload = resp.model_dump()
            _cache_set(sha_cache_key, payload)
            if url_cache_key:
                _cache_set(url_cache_key, payload)
            return resp
        if EARLY_ALLOW and p_torch <= FILTER_ALLOW:
            resp = DecisionOut(
                decision="allow",
                categories=["torch_filter"],
                scores={"torch_filter": p_torch},
                reasons=[f"torch filter <= {FILTER_ALLOW}"],
                actions=[],
                hash=sample_hash,
            )
            payload = resp.model_dump()
            _cache_set(sha_cache_key, payload)
            if url_cache_key:
                _cache_set(url_cache_key, payload)
            return resp

    # 2) OpenAI Moderation (multimodal) OR neutral if disabled
    bucket.acquire()  # smooth bursts
    res = await call_moderation_with_backoff(oai_input)

    raw = res.model_dump() if hasattr(res, "model_dump") else res
    scores = normalize_categories(raw)
    decision = apply_policy(scores)
    resp = DecisionOut(**decision, hash=sample_hash)
    payload = resp.model_dump()
    _cache_set(sha_cache_key, payload)
    if url_cache_key:
        _cache_set(url_cache_key, payload)
    return resp

# -----
# Main
# -----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
