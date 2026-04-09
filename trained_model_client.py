import json
import logging
import os
import re
import time
from typing import List, Optional
from urllib.parse import urlparse

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
        timeout_override = os.getenv("TRAINED_MODEL_TIMEOUT_SECONDS")
        retries_override = os.getenv("TRAINED_MODEL_MAX_RETRIES")
        backoff_override = os.getenv("TRAINED_MODEL_BACKOFF_SECONDS")
        batch_size_override = os.getenv("TRAINED_MODEL_BATCH_SIZE")
        self.timeout_seconds = int(timeout_override or model_cfg.get("timeout_seconds", 60) or 60)
        self.max_retries = max(0, int(retries_override or model_cfg.get("max_retries", 2) or 2))
        self.backoff_seconds = max(0.0, float(backoff_override or model_cfg.get("backoff_seconds", 5.0) or 5.0))
        self.batch_size = max(1, int(batch_size_override or model_cfg.get("batch_size", 1) or 1))
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
        out = []
        for start in range(0, len(payload_candidates), self.batch_size):
            batch = payload_candidates[start : start + self.batch_size]
            try:
                raw_signals = self._predict_batch_http(batch)
            except Exception as exc:
                self.last_error = str(exc)
                logger.warning(
                    "Trained model batch inference failed for batch %s-%s: %s",
                    start,
                    start + len(batch) - 1,
                    exc,
                )
                out.extend([None for _ in batch])
                continue

            normalized = [self._normalize_prediction(signal) for signal in raw_signals]
            while len(normalized) < len(batch):
                normalized.append(None)
            out.extend(normalized[: len(batch)])
        return out[: len(payload_candidates)]

    def _predict_batch_http(self, candidates: List[dict]):
        payload = {
            "candidates": candidates,
            "task": "trade_signal_classification",
        }
        headers = self._request_headers()
        data = None
        last_exc = None
        prediction_url = self._prediction_url()
        for attempt in range(self.max_retries + 1):
            attempt_started = time.time()
            logger.info(
                "Trained model batch request: size=%s attempt=%s timeout=%ss",
                len(candidates),
                attempt + 1,
                self.timeout_seconds,
            )
            try:
                response = requests.post(
                    prediction_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
                if response.status_code >= 500:
                    detail = self._error_detail(response)
                    raise requests.HTTPError(
                        f"{response.status_code} Server Error: {detail or response.reason or 'remote inference failed'} for url: {response.url}",
                        response=response,
                    )
                response.raise_for_status()
                data = response.json()
                logger.info(
                    "Trained model batch response: size=%s status=%s elapsed=%.2fs",
                    len(candidates),
                    response.status_code,
                    time.time() - attempt_started,
                )
                break
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    raise
                sleep_seconds = self.backoff_seconds * (attempt + 1)
                logger.warning(
                    "Trained model HTTP attempt %s/%s failed: %s; retrying in %.1fs",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)
        if data is None and last_exc is not None:
            raise last_exc
        self.last_model_used = data.get("model") or data.get("model_used") or self.model_identifier
        signals = data.get("signals")
        if isinstance(signals, list):
            return signals
        signal = data.get("signal")
        if signal is not None:
            return [signal]
        return []

    def health(self) -> dict:
        response = requests.get(
            self._health_url(),
            headers=self._request_headers(),
            timeout=min(self.timeout_seconds, 30),
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Trained model health response was not a JSON object.")
        return payload

    def wait_until_ready(self, timeout_seconds: int = 600, poll_seconds: float = 10.0) -> dict:
        timeout_seconds = max(1, int(timeout_seconds or 1))
        poll_seconds = max(0.5, float(poll_seconds or 0.5))
        deadline = time.time() + timeout_seconds
        last_error = "trained model readiness probe did not start"
        while time.time() < deadline:
            try:
                payload = self.health()
                if payload.get("ok") is True:
                    self.last_error = None
                    self.last_model_used = payload.get("model") or self.model_identifier
                    return payload
                last_error = str(payload.get("error") or payload)
            except Exception as exc:
                last_error = str(exc)
            self.last_error = last_error
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            time.sleep(min(poll_seconds, remaining))
        raise RuntimeError(f"Trained model endpoint did not become ready within {timeout_seconds}s: {last_error}")

    def _prediction_url(self) -> str:
        url = (self.inference_url or "").strip()
        if not url:
            return url
        parsed = urlparse(url)
        path = (parsed.path or "").rstrip("/")
        if path.endswith("/predict_trade_candidates") or path == "/predict_trade_candidates":
            return url
        if not path or path == "/":
            return url.rstrip("/") + "/predict_trade_candidates"
        return url

    def _health_url(self) -> str:
        url = (self.inference_url or "").strip()
        if not url:
            return url
        parsed = urlparse(url)
        path = (parsed.path or "").rstrip("/")
        if path.endswith("/health") or path == "/health":
            return url
        if path.endswith("/predict_trade_candidates") or path == "/predict_trade_candidates":
            return url[: -len("/predict_trade_candidates")] + "/health"
        if not path or path == "/":
            return url.rstrip("/") + "/health"
        return url.rstrip("/") + "/health"

    def _request_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _error_detail(response: requests.Response) -> str:
        try:
            payload = response.json()
        except Exception:
            payload = None
        if isinstance(payload, dict):
            detail = payload.get("detail") or payload.get("error") or payload.get("message")
            if detail:
                return str(detail)
        text = (response.text or "").strip()
        if text:
            return text[:500]
        return ""

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
        class_probabilities = parsed.get("class_probabilities")
        if not isinstance(class_probabilities, dict):
            class_probabilities = {}
        else:
            cleaned = {}
            for key, value in class_probabilities.items():
                try:
                    cleaned[str(key).strip().upper()] = float(value)
                except (TypeError, ValueError):
                    continue
            class_probabilities = cleaned
        return {
            "label": label,
            "score": LABEL_TO_SCORE[label],
            "confidence": confidence,
            "reason": reason,
            "raw_text": raw_text,
            "class_probabilities": class_probabilities,
        }

    @staticmethod
    def _parse_plain_label(text: str) -> Optional[dict]:
        if not text:
            return None
        match = _LABEL_RE.search(str(text))
        if not match:
            return None
        return {"label": match.group(1).upper(), "reason": str(text).strip()}
