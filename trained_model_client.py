import json
import logging
import os
import re
from typing import Optional

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
        self.model_name = str(model_cfg.get("model_name", "trained-trading-model") or "trained-trading-model").strip()
        self.last_error = None
        self.last_model_used = None

    @property
    def model_identifier(self) -> str:
        return self.model_name or self.inference_url or "trained-model-http"

    def is_ready(self) -> bool:
        if self.backend != "http":
            self.last_error = f"Unsupported trained model backend: {self.backend}. Use remote HTTP inference only."
            return False
        if not self.inference_url:
            self.last_error = "trained_model.inference_url is not configured"
            return False
        return True

    def predict_candidate(self, candidate: dict) -> Optional[dict]:
        if not self.is_ready():
            return None
        try:
            raw = self._predict_http(candidate)
        except Exception as exc:
            self.last_error = str(exc)
            logger.warning("Trained model inference failed for %s: %s", candidate.get("symbol"), exc)
            return None
        return self._normalize_prediction(raw)

    def _predict_http(self, candidate: dict):
        payload = {
            "candidate": candidate,
            "task": "trade_signal_classification",
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
