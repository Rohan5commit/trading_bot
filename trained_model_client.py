import json
import logging
import os
import re
from typing import List, Optional

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
        self.model_name = str(model_cfg.get("model_name", "quant-trained-trading-model") or "quant-trained-trading-model").strip()
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
        results = self.predict_candidates([candidate])
        return results[0] if results else None

    def predict_candidates(self, candidates: List[dict]) -> List[Optional[dict]]:
        if not self.is_ready():
            return [None for _ in list(candidates or [])]
        payload_candidates = [dict(c or {}) for c in list(candidates or []) if isinstance(c, dict)]
        if not payload_candidates:
            return []
        try:
            raw_signals = self._predict_batch_http(payload_candidates)
        except Exception as exc:
            self.last_error = str(exc)
            logger.warning("Trained model batch inference failed: %s", exc)
            return [None for _ in payload_candidates]

        out = []
        for signal in raw_signals:
            out.append(self._normalize_prediction(signal))
        while len(out) < len(payload_candidates):
            out.append(None)
        return out[: len(payload_candidates)]

    def _predict_batch_http(self, candidates: List[dict]):
        payload = {
            "candidates": candidates,
            "task": "trade_signal_classification",
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(self.inference_url, json=payload, headers=headers, timeout=self.timeout_seconds)
        response.raise_for_status()
        data = response.json()
        self.last_model_used = data.get("model") or data.get("model_used") or self.model_identifier
        signals = data.get("signals")
        if isinstance(signals, list):
            return signals
        signal = data.get("signal")
        if signal is not None:
            return [signal]
        return []

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
        fallback_text = str(parsed.get("reason") or parsed.get("notes") or raw_text or "").strip()
        fallback_match = _LABEL_RE.search(fallback_text)
        if label not in LABEL_TO_SCORE and fallback_match:
            label = fallback_match.group(1).upper()
        elif label == "NEUTRAL" and fallback_match and fallback_match.group(1).upper() != "NEUTRAL":
            label = fallback_match.group(1).upper()
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
