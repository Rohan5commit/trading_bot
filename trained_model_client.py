import json
import logging
import os
import re
import threading
from typing import Any, Dict, Optional

import requests

from llm_sentiment import _extract_json

logger = logging.getLogger(__name__)

LABEL_TO_SCORE = {
    "STRONG_BUY": 2.0,
    "BUY": 1.0,
    "NEUTRAL": 0.0,
    "SELL": -1.0,
    "STRONG_SELL": -2.0,
}

_LABEL_RE = re.compile(r"\b(STRONG_BUY|BUY|NEUTRAL|SELL|STRONG_SELL)\b", re.IGNORECASE)


class TrainedModelTradeClient:
    _runtime_lock = threading.Lock()
    _runtime_cache: Dict[str, Dict[str, Any]] = {}

    def __init__(self, ai_cfg: Optional[dict] = None):
        ai_cfg = dict(ai_cfg or {})
        model_cfg = dict(ai_cfg.get("trained_model") or {})
        self.backend = str(model_cfg.get("backend", "http") or "http").strip().lower()
        self.inference_url_env = str(model_cfg.get("inference_url_env", "") or "").strip()
        self.inference_url = str(model_cfg.get("inference_url", "") or "").strip()
        if not self.inference_url and self.inference_url_env and os.getenv(self.inference_url_env):
            self.inference_url = os.getenv(self.inference_url_env).strip()
        self.api_key_env = str(model_cfg.get("api_key_env", "") or "").strip()
        self.api_key = os.getenv(self.api_key_env).strip() if self.api_key_env and os.getenv(self.api_key_env) else ""
        self.timeout_seconds = int(model_cfg.get("timeout_seconds", 60) or 60)
        self.base_model = str(model_cfg.get("base_model", "Qwen/Qwen2.5-7B-Instruct") or "Qwen/Qwen2.5-7B-Instruct").strip()
        self.adapter_dir = str(model_cfg.get("adapter_dir", "./models/lora_solid_adapter") or "./models/lora_solid_adapter").strip()
        self.max_new_tokens = int(model_cfg.get("max_new_tokens", 64) or 64)
        self.temperature = float(model_cfg.get("temperature", 0.0) or 0.0)
        self.cpu_threads = int(model_cfg.get("cpu_threads", 4) or 4)
        self.last_error = None
        self.last_model_used = None

    @property
    def model_identifier(self) -> str:
        if self.backend == "http":
            return self.inference_url or "trained-model-http"
        adapter_name = os.path.basename(os.path.normpath(self.adapter_dir or "adapter")) or "adapter"
        return f"{self.base_model}+{adapter_name}"

    def is_ready(self) -> bool:
        if self.backend == "http":
            if not self.inference_url:
                self.last_error = "trained_model.inference_url is not configured"
                return False
            return True
        if self.backend == "local":
            if not self.base_model or not self.adapter_dir:
                self.last_error = "trained_model.base_model or trained_model.adapter_dir is missing"
                return False
            return True
        self.last_error = f"Unsupported trained model backend: {self.backend}"
        return False

    def predict_candidate(self, candidate: dict) -> Optional[dict]:
        if not self.is_ready():
            return None
        try:
            raw = self._predict_http(candidate) if self.backend == "http" else self._predict_local(candidate)
        except Exception as exc:
            self.last_error = str(exc)
            logger.warning("Trained model inference failed for %s: %s", candidate.get("symbol"), exc)
            return None
        return self._normalize_prediction(raw)

    def _predict_http(self, candidate: dict):
        payload = {
            "candidate": candidate,
            "task": "trade_signal_classification",
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(self.inference_url, json=payload, headers=headers, timeout=self.timeout_seconds)
        response.raise_for_status()
        data = response.json()
        self.last_model_used = data.get("model") or data.get("model_used") or self.model_identifier
        if isinstance(data.get("signal"), dict):
            return data["signal"]
        return data

    def _predict_local(self, candidate: dict):
        runtime = self._ensure_local_runtime()
        tokenizer = runtime["tokenizer"]
        model = runtime["model"]
        torch = runtime["torch"]
        messages = self._build_messages(candidate)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(prompt, return_tensors="pt")
        input_len = encoded["input_ids"].shape[-1]
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=bool(self.temperature and self.temperature > 0.0),
                temperature=max(self.temperature, 1e-5) if self.temperature and self.temperature > 0 else 1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(generated[0][input_len:], skip_special_tokens=True).strip()
        self.last_model_used = self.model_identifier
        return text

    def _ensure_local_runtime(self):
        cache_key = f"{self.base_model}|{self.adapter_dir}"
        with self._runtime_lock:
            cached = self._runtime_cache.get(cache_key)
            if cached is not None:
                return cached
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            torch.set_num_threads(max(1, int(self.cpu_threads or 1)))
            tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            model = PeftModel.from_pretrained(model, self.adapter_dir, is_trainable=False)
            model.eval()
            runtime = {"tokenizer": tokenizer, "model": model, "torch": torch}
            self._runtime_cache[cache_key] = runtime
            return runtime

    def _build_messages(self, candidate: dict):
        symbol = str(candidate.get("symbol") or "UNKNOWN").strip().upper()
        as_of_date = candidate.get("as_of_date") or candidate.get("last_date") or "UNKNOWN"
        lines = [
            f"TICKER: {symbol}",
            f"DATE: {as_of_date}",
            "PRICE_ACTION:",
            f"- last_close: {candidate.get('last_close')}",
            f"- closes_tail: {candidate.get('closes_tail')}",
            f"- volume_1d: {candidate.get('volume_1d')}",
            f"- volume_20d_avg: {candidate.get('volume_20d_avg')}",
            "INDICATORS:",
            f"- return_1d: {candidate.get('return_1d')}",
            f"- return_5d: {candidate.get('return_5d')}",
            f"- return_10d: {candidate.get('return_10d')}",
            f"- volatility_20d: {candidate.get('volatility_20d')}",
            f"- dist_ma_20: {candidate.get('dist_ma_20')}",
            f"- dist_ma_50: {candidate.get('dist_ma_50')}",
            f"- rsi_14: {candidate.get('rsi_14')}",
            f"- volume_ratio: {candidate.get('volume_ratio')}",
            "NEWS_CONTEXT:",
            f"- news_count_7d: {candidate.get('news_count_7d')}",
            f"- news_sentiment_7d: {candidate.get('news_sentiment_7d')}",
            "",
            "QUESTION: Classify the expected 5-day return as STRONG_BUY | BUY | NEUTRAL | SELL | STRONG_SELL.",
            "Return ONLY JSON using this schema:",
            '{"label":"BUY","confidence":0.63,"reason":"..."}',
        ]
        system = (
            "You are the trained AI trading decision engine. "
            "Return only valid JSON with label, confidence, and reason. "
            "Use the provided market snapshot to classify the next 5-day return."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(lines)}]

    def _normalize_prediction(self, raw) -> Optional[dict]:
        parsed = raw
        raw_text = None
        if isinstance(raw, str):
            raw_text = raw
            parsed = _extract_json(raw) or self._parse_plain_label(raw)
        elif isinstance(raw, dict):
            raw_text = json.dumps(raw)
        else:
            raw_text = str(raw)
            parsed = self._parse_plain_label(raw_text)
        if not isinstance(parsed, dict):
            self.last_error = "Trained model response could not be parsed"
            return None
        label = str(parsed.get("label") or parsed.get("signal") or "").strip().upper()
        if label not in LABEL_TO_SCORE:
            self.last_error = f"Unsupported trained model label: {label or 'missing'}"
            return None
        confidence = parsed.get("confidence")
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.9 if label.startswith("STRONG_") else (0.65 if label != "NEUTRAL" else 0.5)
        confidence = max(0.0, min(1.0, confidence))
        reason = str(parsed.get("reason") or parsed.get("notes") or f"Model classified {label}.").strip()
        return {
            "label": label,
            "score": LABEL_TO_SCORE[label],
            "confidence": confidence,
            "reason": reason,
            "raw_text": raw_text,
        }

    @staticmethod
    def _parse_plain_label(text: str) -> Optional[dict]:
        if not text:
            return None
        match = _LABEL_RE.search(str(text))
        if not match:
            return None
        return {"label": match.group(1).upper(), "reason": str(text).strip()}
