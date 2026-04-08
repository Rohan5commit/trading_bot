from __future__ import annotations

import json
import types
import inspect
import importlib.util
import os
from pathlib import Path
import re
import shutil
import sys
import tarfile
import threading
from typing import Any, Dict, List
import zipfile

import requests
from fastapi import Depends, FastAPI, Header, HTTPException


BASE_MODEL = os.getenv("TRAINED_MODEL_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
MODEL_NAME = os.getenv("TRAINED_MODEL_NAME", "quant-trained-trading-model")
CPU_THREADS = max(1, int(os.getenv("TRAINED_MODEL_CPU_THREADS", os.getenv("TRAINED_MODEL_CPU", "4")) or 4))
ADAPTER_PATH = os.getenv("TRAINED_MODEL_ADAPTER_PATH", "").strip()
ADAPTER_ARCHIVE_URL = os.getenv("TRAINED_MODEL_ADAPTER_ARCHIVE_URL", "").strip()
ADAPTER_ARCHIVE_TOKEN = os.getenv("TRAINED_MODEL_ADAPTER_ARCHIVE_TOKEN", "").strip()
ADAPTER_CACHE_DIR = os.getenv("TRAINED_MODEL_CACHE_DIR", "/tmp/trained_model_service").strip()
SERVICE_API_KEY = os.getenv("TRAINED_MODEL_API_KEY", "").strip()

LABEL_RE = re.compile(r"\b(STRONG_BUY|BUY|NEUTRAL|SELL|STRONG_SELL)\b", re.IGNORECASE)
CLASS_TOKEN_RE = re.compile(r"\b([ABCDE])\b", re.IGNORECASE)
CLASS_TOKEN_TO_LABEL = {
    "A": "STRONG_BUY",
    "B": "BUY",
    "C": "NEUTRAL",
    "D": "SELL",
    "E": "STRONG_SELL",
}
_MODEL = None
_TOKENIZER = None
_TORCH = None
_ADAPTER_DIR = None
_LOAD_ERROR = None
_ADAPTER_LOCK = threading.Lock()
_LOAD_LOCK = threading.Lock()


def _env_flag(name: str, default: bool = False) -> bool:
    value = str(os.getenv(name, "1" if default else "0") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _disable_torchvision_discovery() -> None:
    original_find_spec = importlib.util.find_spec

    if getattr(importlib.util.find_spec, "_trading_bot_torchvision_patched", False):
        return

    def patched_find_spec(name: str, *args, **kwargs):
        if name == "torchvision" or name.startswith("torchvision."):
            return None
        return original_find_spec(name, *args, **kwargs)

    patched_find_spec._trading_bot_torchvision_patched = True
    importlib.util.find_spec = patched_find_spec
    sys.modules.pop("torchvision", None)
    sys.modules.pop("torchvision.transforms", None)

    class _InterpolationMode:
        NEAREST = "nearest"
        BOX = "box"
        BILINEAR = "bilinear"
        HAMMING = "hamming"
        BICUBIC = "bicubic"
        LANCZOS = "lanczos"

    transforms_module = types.ModuleType("torchvision.transforms")
    transforms_module.InterpolationMode = _InterpolationMode
    torchvision_module = types.ModuleType("torchvision")
    torchvision_module.transforms = transforms_module
    sys.modules["torchvision"] = torchvision_module
    sys.modules["torchvision.transforms"] = transforms_module


def _patch_peft_lora_config_compat() -> None:
    from peft.tuners.lora.config import LoraConfig

    if getattr(LoraConfig, "_trading_bot_compat_patched", False):
        return

    original_init = LoraConfig.__init__
    allowed = set(inspect.signature(original_init).parameters)

    def compat_init(self, *args, **kwargs):
        filtered = {key: value for key, value in kwargs.items() if key in allowed}
        return original_init(self, *args, **filtered)

    LoraConfig.__init__ = compat_init
    LoraConfig._trading_bot_compat_patched = True


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
    with _ADAPTER_LOCK:
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
    if match:
        return {"label": match.group(1).upper(), "confidence": 0.5, "reason": str(text).strip()}
    class_match = CLASS_TOKEN_RE.search(str(text))
    if not class_match:
        return None
    token = class_match.group(1).upper()
    label = CLASS_TOKEN_TO_LABEL.get(token)
    if not label:
        return None
    return {"label": label, "confidence": 0.5, "reason": str(text).strip()}


def _candidate_prompt(candidate: Dict[str, Any]) -> str:
    symbol = str(candidate.get("symbol") or "UNKNOWN").strip().upper()
    as_of_date = candidate.get("as_of_date") or candidate.get("last_date") or "UNKNOWN"
    def _fmt(value: Any, digits: int = 4) -> str:
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return "na"

    return (
        "A=STRONG_BUY B=BUY C=NEUTRAL D=SELL E=STRONG_SELL. "
        f"T={symbol} D={as_of_date} "
        f"R1={_fmt(candidate.get('return_1d'))} "
        f"R5={_fmt(candidate.get('return_5d'))} "
        f"R10={_fmt(candidate.get('return_10d'))} "
        f"V20={_fmt(candidate.get('volatility_20d'))} "
        f"M20={_fmt(candidate.get('dist_ma_20'))} "
        f"M50={_fmt(candidate.get('dist_ma_50'))} "
        f"RSI={_fmt(candidate.get('rsi_14'), 2)} "
        f"VR={_fmt(candidate.get('volume_ratio'), 2)} "
        f"NC={int(candidate.get('news_count_7d') or 0)} "
        f"NS={_fmt(candidate.get('news_sentiment_7d'), 3)} "
        "Class:"
    )


def _reason_from_candidate(label: str, candidate: Dict[str, Any]) -> str:
    label = str(label or "NEUTRAL").strip().upper()
    ret_5d = candidate.get("return_5d")
    rsi_14 = candidate.get("rsi_14")
    sentiment = candidate.get("news_sentiment_7d")

    def _fmt(value: Any) -> str | None:
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return None

    ret_text = _fmt(ret_5d)
    rsi_text = _fmt(rsi_14)
    sentiment_text = _fmt(sentiment)

    if label in {"STRONG_BUY", "BUY"}:
        parts = ["positive momentum bias"]
        if ret_text is not None:
            parts.append(f"5d return {ret_text}")
        if rsi_text is not None:
            parts.append(f"RSI {rsi_text}")
        return ", ".join(parts)
    if label in {"STRONG_SELL", "SELL"}:
        parts = ["negative momentum bias"]
        if ret_text is not None:
            parts.append(f"5d return {ret_text}")
        if sentiment_text is not None:
            parts.append(f"news sentiment {sentiment_text}")
        return ", ".join(parts)
    return "mixed short-term signals"


def _load_runtime():
    global _MODEL, _TOKENIZER, _TORCH, _LOAD_ERROR
    if _MODEL is not None and _TOKENIZER is not None and _TORCH is not None:
        return _MODEL, _TOKENIZER, _TORCH
    if _LOAD_ERROR is not None:
        raise RuntimeError(_LOAD_ERROR)
    with _LOAD_LOCK:
        if _MODEL is not None and _TOKENIZER is not None and _TORCH is not None:
            return _MODEL, _TOKENIZER, _TORCH
        if _LOAD_ERROR is not None:
            raise RuntimeError(_LOAD_ERROR)
        try:
            import torch

            _disable_torchvision_discovery()
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            torch.set_num_threads(CPU_THREADS)
            try:
                torch.set_num_interop_threads(1)
            except Exception:
                pass
            _patch_peft_lora_config_compat()
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
            if _env_flag("TRAINED_MODEL_MERGE_ADAPTER", default=True) and hasattr(model, "merge_and_unload"):
                model = model.merge_and_unload()
            if _env_flag("TRAINED_MODEL_DYNAMIC_QUANTIZE", default=True):
                try:
                    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
                except Exception:
                    pass
            model.eval()
            _MODEL = model
            _TOKENIZER = tokenizer
            _TORCH = torch
            return _MODEL, _TOKENIZER, _TORCH
        except Exception as exc:
            _LOAD_ERROR = str(exc)
            raise


def _prediction_from_text(text: str, candidate: Dict[str, Any]) -> Dict[str, Any]:
    parsed = _extract_json(text) or _parse_plain_label(text) or {}
    label = str(parsed.get("label") or "").strip().upper()
    if label in CLASS_TOKEN_TO_LABEL:
        label = CLASS_TOKEN_TO_LABEL[label]
    if label not in {"STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL"}:
        match = LABEL_RE.search(text or "")
        if match:
            label = match.group(1).upper()
        else:
            class_match = CLASS_TOKEN_RE.search(text or "")
            label = CLASS_TOKEN_TO_LABEL.get(class_match.group(1).upper(), "NEUTRAL") if class_match else "NEUTRAL"
    confidence = {
        "STRONG_BUY": 0.9,
        "BUY": 0.72,
        "NEUTRAL": 0.5,
        "SELL": 0.72,
        "STRONG_SELL": 0.9,
    }[label]
    return {
        "label": label,
        "confidence": confidence,
        "reason": _reason_from_candidate(label, candidate),
        "symbol": candidate.get("symbol"),
    }


def _predict_batch(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    model, tokenizer, torch = _load_runtime()
    prompts = [_candidate_prompt(candidate) for candidate in candidates]
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )
    input_lens = encoded["attention_mask"].sum(dim=1).tolist()
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    outputs: List[Dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        start = int(input_lens[index])
        text = tokenizer.decode(generated[index][start:], skip_special_tokens=True).strip()
        outputs.append(_prediction_from_text(text, candidate))
    return outputs


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


@app.get("/health")
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
        "ready": _MODEL is not None and _TOKENIZER is not None and _TORCH is not None,
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


@app.post("/warmup", dependencies=[Depends(_require_api_key)])
def warmup() -> dict[str, Any]:
    model, tokenizer, _torch = _load_runtime()
    return {
        "ok": True,
        "model": MODEL_NAME,
        "base_model": BASE_MODEL,
        "ready": model is not None and tokenizer is not None,
    }


@app.post("/predict_trade_candidates", dependencies=[Depends(_require_api_key)])
def predict_trade_candidates(payload: Dict[str, Any]) -> dict[str, Any]:
    candidates = _normalize_candidates(payload)
    if not candidates:
        raise HTTPException(status_code=400, detail="No candidate payload supplied.")
    try:
        signals = _predict_batch(candidates)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "model": MODEL_NAME,
        "model_used": MODEL_NAME,
        "signals": signals,
    }
