import json
import os
import re
from typing import Any, Dict, List

import modal

APP_NAME = os.getenv("TRAINED_MODEL_MODAL_APP", "trading-bot-trained-model-inference")
BASE_MODEL = os.getenv("TRAINED_MODEL_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
VOLUME_NAME = os.getenv("TRAINED_MODEL_VOLUME", "train-once-artifacts")
ADAPTER_PATH = os.getenv("TRAINED_MODEL_ADAPTER_PATH", "/artifacts/lora_solid_adapter")
MODEL_NAME = os.getenv("TRAINED_MODEL_NAME", "quant-trained-trading-model")
CPU_COUNT = int(os.getenv("TRAINED_MODEL_CPU", "8"))
MEMORY_MB = int(os.getenv("TRAINED_MODEL_MEMORY_MB", "49152"))

app = modal.App(APP_NAME)
os.environ.setdefault("HF_HOME", "/artifacts/hf_home")
os.environ.setdefault("TRANSFORMERS_CACHE", "/artifacts/hf_home/transformers")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/artifacts/hf_home/hub")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install(
        "fastapi>=0.115.0",
        "pydantic>=2.9.2",
        "transformers>=4.46.0",
        "peft>=0.13.2",
        "accelerate>=1.0.1",
        "sentencepiece>=0.2.0",
    )
)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

_MODEL = None
_TOKENIZER = None
_TORCH = None


def _load_runtime():
    global _MODEL, _TOKENIZER, _TORCH
    if _MODEL is not None and _TOKENIZER is not None and _TORCH is not None:
        return _MODEL, _TOKENIZER, _TORCH

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(max(1, CPU_COUNT))
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, ADAPTER_PATH, is_trainable=False)
    model.eval()

    _MODEL = model
    _TOKENIZER = tokenizer
    _TORCH = torch
    return _MODEL, _TOKENIZER, _TORCH


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


LABEL_RE = re.compile(r"\\b(STRONG_BUY|BUY|NEUTRAL|SELL|STRONG_SELL)\\b", re.IGNORECASE)


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
            pass
    return None


def _parse_plain_label(text: str):
    if not text:
        return None
    match = LABEL_RE.search(str(text))
    if not match:
        return None
    return {"label": match.group(1).upper(), "confidence": 0.5, "reason": str(text).strip()}


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
    parsed = _extract_json(text) or _parse_plain_label(text) or {"label": "NEUTRAL", "confidence": 0.5, "reason": text or "No parsable output."}
    parsed["symbol"] = candidate.get("symbol")
    return parsed


@app.function(
    image=image,
    cpu=CPU_COUNT,
    memory=MEMORY_MB,
    scaledown_window=300,
    timeout=7200,
    startup_timeout=1800,
    volumes={"/artifacts": volume},
)
@modal.fastapi_endpoint(method="POST")
def predict_trade_candidates(payload: Dict[str, Any]):
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        candidate = payload.get("candidate")
        candidates = [candidate] if isinstance(candidate, dict) else []
    else:
        candidates = list(candidates or [])
    signals = [_predict_one(c) for c in candidates if isinstance(c, dict)]
    return {
        "model": MODEL_NAME,
        "model_used": MODEL_NAME,
        "signals": signals,
    }
