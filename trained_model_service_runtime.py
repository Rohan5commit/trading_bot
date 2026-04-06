from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import tarfile
from typing import Any, Dict, List
import zipfile

import requests
from fastapi import Depends, FastAPI, Header, HTTPException


BASE_MODEL = os.getenv("TRAINED_MODEL_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
MODEL_NAME = os.getenv("TRAINED_MODEL_NAME", "quant-trained-trading-model")
CPU_THREADS = max(1, int(os.getenv("TRAINED_MODEL_CPU_THREADS", os.getenv("TRAINED_MODEL_CPU", "8")) or 8))
ADAPTER_PATH = os.getenv("TRAINED_MODEL_ADAPTER_PATH", "").strip()
ADAPTER_ARCHIVE_URL = os.getenv("TRAINED_MODEL_ADAPTER_ARCHIVE_URL", "").strip()
ADAPTER_ARCHIVE_TOKEN = os.getenv("TRAINED_MODEL_ADAPTER_ARCHIVE_TOKEN", "").strip()
ADAPTER_CACHE_DIR = os.getenv("TRAINED_MODEL_CACHE_DIR", "/tmp/trained_model_service").strip()
SERVICE_API_KEY = os.getenv("TRAINED_MODEL_API_KEY", "").strip()

LABEL_RE = re.compile(r"\b(STRONG_BUY|BUY|NEUTRAL|SELL|STRONG_SELL)\b", re.IGNORECASE)
_MODEL = None
_TOKENIZER = None
_TORCH = None
_ADAPTER_DIR = None
_LOAD_ERROR = None


def _cache_root() -> Path:
    root = Path(ADAPTER_CACHE_DIR or "/tmp/trained_model_service").expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _is_adapter_dir(path: Path) -> bool:
    return path.is_dir() and (path / "adapter_model.safetensors").exists() and (path / "adapter_config.json").exists()


def _safe_extract_tar(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive_path) as tar:
        members = tar.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if not str(target).startswith(str(destination)):
                raise RuntimeError(f"Blocked unsafe tar member path: {member.name}")
        tar.extractall(destination)


def _safe_extract_zip(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive_path) as zf:
        for member in zf.namelist():
            target = (destination / member).resolve()
            if not str(target).startswith(str(destination)):
                raise RuntimeError(f"Blocked unsafe zip member path: {member}")
        zf.extractall(destination)


def _resolve_extracted_adapter_dir(destination: Path) -> Path:
    if _is_adapter_dir(destination):
        return destination
    candidates = sorted({path.parent for path in destination.rglob("adapter_model.safetensors")})
    for candidate in candidates:
        if _is_adapter_dir(candidate):
            return candidate
    raise RuntimeError("Downloaded adapter archive did not contain a valid LoRA adapter directory.")


def _download_adapter_archive() -> Path:
    cache_root = _cache_root()
    archive_path = cache_root / "adapter_archive.bin"
    extract_root = cache_root / "adapter_extract"
    headers = {}
    if ADAPTER_ARCHIVE_TOKEN:
        headers["Authorization"] = f"Bearer {ADAPTER_ARCHIVE_TOKEN}"
    if ADAPTER_ARCHIVE_URL.startswith("https://api.github.com/") and "/releases/assets/" in ADAPTER_ARCHIVE_URL:
        headers.setdefault("Accept", "application/octet-stream")
        headers.setdefault("X-GitHub-Api-Version", "2022-11-28")
    with requests.get(ADAPTER_ARCHIVE_URL, headers=headers, timeout=120, stream=True) as response:
        response.raise_for_status()
        with archive_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    shutil.rmtree(extract_root, ignore_errors=True)
    extract_root.mkdir(parents=True, exist_ok=True)
    if tarfile.is_tarfile(archive_path):
        _safe_extract_tar(archive_path, extract_root)
    elif zipfile.is_zipfile(archive_path):
        _safe_extract_zip(archive_path, extract_root)
    else:
        raise RuntimeError("Adapter archive must be a .tar/.tar.gz or .zip file.")
    return _resolve_extracted_adapter_dir(extract_root)


def _ensure_adapter_dir() -> Path:
    global _ADAPTER_DIR
    if _ADAPTER_DIR is not None and _is_adapter_dir(_ADAPTER_DIR):
        return _ADAPTER_DIR
    if ADAPTER_PATH:
        candidate = Path(ADAPTER_PATH).expanduser().resolve()
        if _is_adapter_dir(candidate):
            _ADAPTER_DIR = candidate
            return _ADAPTER_DIR
    if not ADAPTER_ARCHIVE_URL:
        raise RuntimeError(
            "No trained adapter is available. Set TRAINED_MODEL_ADAPTER_PATH or TRAINED_MODEL_ADAPTER_ARCHIVE_URL."
        )
    _ADAPTER_DIR = _download_adapter_archive()
    return _ADAPTER_DIR


def _extract_json(text: str):
    if not text:
        return None
    text = str(text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


def _parse_plain_label(text: str):
    if not text:
        return None
    match = LABEL_RE.search(str(text))
    if not match:
        return None
    return {"label": match.group(1).upper(), "confidence": 0.5, "reason": str(text).strip()}


def _candidate_prompt(candidate: Dict[str, Any]) -> str:
    symbol = str(candidate.get("symbol") or "UNKNOWN").strip().upper()
    as_of_date = candidate.get("as_of_date") or candidate.get("last_date") or "UNKNOWN"
    lines = [
        f"TICKER: {symbol}",
        f"DATE: {as_of_date}",
        f"LAST_CLOSE: {candidate.get('last_close')}",
        f"CLOSES_TAIL: {candidate.get('closes_tail')}",
        f"RETURN_1D: {candidate.get('return_1d')}",
        f"RETURN_5D: {candidate.get('return_5d')}",
        f"RETURN_10D: {candidate.get('return_10d')}",
        f"VOLATILITY_20D: {candidate.get('volatility_20d')}",
        f"DIST_MA_20: {candidate.get('dist_ma_20')}",
        f"DIST_MA_50: {candidate.get('dist_ma_50')}",
        f"RSI_14: {candidate.get('rsi_14')}",
        f"VOLUME_RATIO: {candidate.get('volume_ratio')}",
        f"NEWS_COUNT_7D: {candidate.get('news_count_7d')}",
        f"NEWS_SENTIMENT_7D: {candidate.get('news_sentiment_7d')}",
        "QUESTION: Classify the expected 5-day return as STRONG_BUY | BUY | NEUTRAL | SELL | STRONG_SELL.",
        'Return only compact JSON: {"label":"BUY","confidence":0.63,"reason":"short english phrase"}',
    ]
    return "\n".join(lines)


def _load_runtime():
    global _MODEL, _TOKENIZER, _TORCH, _LOAD_ERROR
    if _MODEL is not None and _TOKENIZER is not None and _TORCH is not None:
        return _MODEL, _TOKENIZER, _TORCH
    if _LOAD_ERROR is not None:
        raise RuntimeError(_LOAD_ERROR)
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(CPU_THREADS)
        adapter_dir = _ensure_adapter_dir()
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype="auto",
        )
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
        model.eval()
        _MODEL = model
        _TOKENIZER = tokenizer
        _TORCH = torch
        return _MODEL, _TOKENIZER, _TORCH
    except Exception as exc:
        _LOAD_ERROR = str(exc)
        raise


def _predict_one(candidate: Dict[str, Any]) -> Dict[str, Any]:
    model, tokenizer, torch = _load_runtime()
    system = (
        "You are the trained AI trading decision engine. "
        "Return only valid compact JSON with label, confidence, and a very short reason."
    )
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": _candidate_prompt(candidate)},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
    input_len = encoded["input_ids"].shape[-1]
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=24,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(generated[0][input_len:], skip_special_tokens=True).strip()
    parsed = _extract_json(text) or _parse_plain_label(text) or {
        "label": "NEUTRAL",
        "confidence": 0.5,
        "reason": text or "No parsable output.",
    }
    parsed["symbol"] = candidate.get("symbol")
    return parsed


def _require_api_key(authorization: str | None = Header(default=None)) -> None:
    if not SERVICE_API_KEY:
        return
    expected = f"Bearer {SERVICE_API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _normalize_candidates(payload: Dict[str, Any]) -> List[dict]:
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        return [dict(item) for item in candidates if isinstance(item, dict)]
    candidate = payload.get("candidate")
    if isinstance(candidate, dict):
        return [dict(candidate)]
    return []


app = FastAPI(title=MODEL_NAME)


@app.get("/health", dependencies=[Depends(_require_api_key)])
def health() -> dict[str, Any]:
    try:
        adapter_dir = _ensure_adapter_dir()
    except Exception as exc:
        return {
            "ok": False,
            "model": MODEL_NAME,
            "base_model": BASE_MODEL,
            "error": str(exc),
        }
    return {
        "ok": True,
        "model": MODEL_NAME,
        "base_model": BASE_MODEL,
        "adapter_dir": str(adapter_dir),
    }


@app.get("/model-info", dependencies=[Depends(_require_api_key)])
def model_info() -> dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "base_model": BASE_MODEL,
        "cpu_threads": CPU_THREADS,
        "adapter_path": ADAPTER_PATH or None,
        "adapter_archive_url": bool(ADAPTER_ARCHIVE_URL),
    }


@app.post("/predict_trade_candidates", dependencies=[Depends(_require_api_key)])
def predict_trade_candidates(payload: Dict[str, Any]) -> dict[str, Any]:
    candidates = _normalize_candidates(payload)
    if not candidates:
        raise HTTPException(status_code=400, detail="No candidate payload supplied.")
    signals = [_predict_one(candidate) for candidate in candidates]
    return {
        "model": MODEL_NAME,
        "model_used": MODEL_NAME,
        "signals": signals,
    }
