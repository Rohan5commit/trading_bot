import json
import logging
import os
import re
import time

import requests

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    # Load .env from repo root reliably (scheduler may run with a different cwd).
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
except Exception:
    pass


def _strip_code_fences(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned)
    return cleaned.strip()


def _extract_json(text):
    if not text:
        return None
    cleaned = _strip_code_fences(text)
    
    # 1. Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 2. Try finding the outermost braces/brackets
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = cleaned.find(start_char)
        while start != -1:
            end = cleaned.rfind(end_char)
            while end != -1 and end > start:
                candidate = cleaned[start:end + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # Shrink from the right and try again
                    end = cleaned.rfind(end_char, start, end)
            # Try next start character
            start = cleaned.find(start_char, start + 1)

    logger.warning("Failed to extract JSON from LLM response. Raw text snippet: %s", (text[:500] + "...") if len(text) > 500 else text)
    return None


def _clamp(value, low, high):
    return max(low, min(high, value))


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _chunked(items, size):
    size = max(1, size)
    for idx in range(0, len(items), size):
        yield items[idx:idx + size]


class NvidiaChatClient:
    def __init__(self, llm_cfg):
        self.enabled = llm_cfg.get("enabled", False)
        self.model = llm_cfg.get("model", "stockmark-2-100b-instruct")
        fallback_models = llm_cfg.get("fallback_models", [])
        if isinstance(fallback_models, str):
            fallback_models = [fallback_models]
        if not isinstance(fallback_models, list):
            fallback_models = []
        self.fallback_models = [str(m).strip() for m in fallback_models if str(m).strip()]
        self.base_url = llm_cfg.get(
            "base_url",
            "https://integrate.api.nvidia.com/v1/chat/completions"
        )
        self.api_key_env = llm_cfg.get("api_key_env", "NVIDIA_API_KEY")
        self.api_key = os.getenv(self.api_key_env)
        if isinstance(self.api_key, str):
            self.api_key = self.api_key.strip().strip('"').strip("'")
        self.temperature = llm_cfg.get("temperature", 0.1)
        self.top_p = llm_cfg.get("top_p", 0.9)
        self.max_tokens = llm_cfg.get("max_tokens", 512)
        self.timeout_seconds = llm_cfg.get("timeout_seconds", 20)
        self.max_retries = llm_cfg.get("max_retries", 2)
        self.backoff_seconds = llm_cfg.get("backoff_seconds", 1.0)
        self.last_error = None
        self.last_model_used = None

        # Some NVIDIA endpoints accept different model naming conventions. We'll try a few.
        self.model_candidates = self._build_model_candidates(self.model, self.fallback_models)

    @staticmethod
    def _build_model_candidates(raw_model, fallback_models=None):
        preferred = []
        raw = (raw_model or "").strip()
        if raw:
            preferred.append(raw)
        for fb in fallback_models or []:
            fb = str(fb or "").strip()
            if fb:
                preferred.append(fb)

        candidates = []
        for model_id in preferred:
            mid = str(model_id or "").strip()
            if not mid:
                continue

            # Guardrail: nvidia/stockmark-* is not a valid NVIDIA model id.
            # Rewrite to stockmark/<model> so fallback doesn't fail on this known bad variant.
            if mid.startswith("nvidia/stockmark-"):
                mid = "stockmark/" + mid.split("/", 1)[1]

            candidates.append(mid)

            if "/" not in mid:
                # Plain stockmark models use stockmark/<model>; other plain ids may require nvidia/<model>.
                if mid.startswith("stockmark-"):
                    candidates.append(f"stockmark/{mid}")
                else:
                    candidates.append(f"nvidia/{mid}")
                continue

            namespace, name = mid.split("/", 1)
            namespace = namespace.strip().lower()
            name = name.strip()
            if name and namespace in {"nvidia", "meta", "stockmark"}:
                candidates.append(name)

        # De-dupe while preserving order.
        seen = set()
        out = []
        for c in candidates:
            if c and c not in seen:
                out.append(c)
                seen.add(c)
        return out

    def is_ready(self):
        if not self.enabled:
            return False
        if not self.api_key:
            logger.warning(
                "LLM enabled but %s is not set. Skipping LLM calls.",
                self.api_key_env
            )
            return False
        return True

    def chat(self, messages):
        if not self.is_ready():
            return None
        self.last_model_used = None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "trading_bot/1.0",
        }

        last_exc = None
        for attempt in range(self.max_retries + 1):
            for model in self.model_candidates or [self.model]:
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "max_tokens": self.max_tokens,
                    "stream": False,
                }
                try:
                    response = requests.post(
                        self.base_url,
                        json=payload,
                        headers=headers,
                        timeout=self.timeout_seconds
                    )
                    if response.status_code in (400, 401, 403, 404, 422):
                        self.last_error = f"HTTP {response.status_code} for model={model}: {response.text[:500]}"
                        logger.warning("LLM rejected request for model=%s (HTTP %s). Trying next model.", model, response.status_code)
                        continue
                    if response.status_code == 429:
                        self.last_error = f"HTTP 429 rate limit for model={model}: {response.text[:500]}"
                        logger.warning("LLM rate limited (model=%s).", model)
                        continue

                    response.raise_for_status()
                    data = response.json()
                    content = self._extract_content(data)
                    if not content:
                        self.last_error = f"Empty response content for model={model}"
                        logger.warning("LLM returned empty content for model=%s. Trying next model.", model)
                        continue
                    self.last_error = None
                    self.last_model_used = model
                    return content
                except requests.exceptions.RequestException as exc:
                    last_exc = exc
                    # If we got a structured response, include it.
                    try:
                        status = getattr(getattr(exc, "response", None), "status_code", None)
                        text = getattr(getattr(exc, "response", None), "text", "")
                        if status is not None:
                            self.last_error = f"HTTP {status} for model={model}: {str(text)[:500]}"
                        else:
                            self.last_error = str(exc)
                    except Exception:
                        self.last_error = str(exc)
                    logger.warning("LLM request failed (attempt %d/%d): %s", attempt + 1, self.max_retries + 1, self.last_error)
                    continue

            if attempt < self.max_retries:
                time.sleep(self.backoff_seconds * (attempt + 1))

        if last_exc is not None:
            return None
        return None

    @staticmethod
    def _extract_content(data):
        if not isinstance(data, dict):
            return None
        choices = data.get("choices", [])
        if choices:
            choice = choices[0]
            message = choice.get("message", {})
            # Some NVIDIA reasoning-capable models return output in `reasoning_content`
            # and may set `content` to null or an empty string.
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            
            reasoning = message.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning.strip():
                return reasoning.strip()
                
            if isinstance(content, str):
                return content.strip()

            if "text" in choice:
                return choice["text"].strip()
        if "text" in data:
            return str(data["text"]).strip()
        return None


class NewsSentimentScorer:
    def __init__(self, config):
        llm_cfg = config.get("llm", {})
        sentiment_cfg = llm_cfg.get("news_sentiment", {})
        self.enabled = llm_cfg.get("enabled", False) and sentiment_cfg.get("enabled", False)
        self.max_articles = sentiment_cfg.get("max_articles_per_symbol", 30)
        self.batch_size = sentiment_cfg.get("batch_size", 8)
        self.min_confidence = sentiment_cfg.get("min_confidence", 0.15)
        self.confidence_weighted = sentiment_cfg.get("confidence_weighted", False)
        self.client = NvidiaChatClient(llm_cfg)
        self.stats = {
            "enabled": self.enabled,
            "attempts": 0,
            "batches": 0,
            "errors": 0,
            "skipped": 0,
            "last_error": None,
        }

    def score(self, symbol, items):
        if not self.enabled:
            return items
        self.stats["attempts"] += 1
        if not self.client.is_ready():
            self.stats["errors"] += 1
            self.stats["skipped"] += 1
            self.stats["last_error"] = "LLM unavailable or NVIDIA_API_KEY missing."
            logger.warning(self.stats["last_error"])
            return items
        if not items:
            return items

        limited_items = items[:self.max_articles] if self.max_articles else items
        if self.max_articles and len(items) > self.max_articles:
            logger.info(
                "Limiting sentiment scoring to %d articles for %s.",
                self.max_articles,
                symbol
            )

        for batch in _chunked(limited_items, self.batch_size):
            self.stats["batches"] += 1
            success = self._score_batch(symbol, batch)
            if not success:
                self.stats["errors"] += 1
                self.stats["last_error"] = f"LLM scoring failed for {symbol}."
        return items

    def _score_batch(self, symbol, batch):
        prompt_lines = []
        for idx, item in enumerate(batch, start=1):
            headline = item.get("title", "").strip()
            prompt_lines.append(f"{idx}. {headline}")

        system_msg = (
            "You are a financial news sentiment analyst. "
            "Return only valid JSON with numeric sentiment and confidence."
        )
        user_msg = (
            "Score the sentiment of each headline for the given symbol.\n"
            "Return JSON exactly in this format:\n"
            "{\"results\": [{\"id\": 1, \"sentiment\": -1.0, \"confidence\": 0.0}]}\n"
            "Where sentiment is -1 (very negative) to 1 (very positive), "
            "confidence is 0 to 1.\n"
            f"Symbol: {symbol}\n"
            "Headlines:\n"
            + "\n".join(prompt_lines)
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        response_text = self.client.chat(messages)
        parsed = _extract_json(response_text)
        if parsed is None:
            logger.warning("LLM response missing JSON for %s.", symbol)
            return False

        results = parsed.get("results") if isinstance(parsed, dict) else parsed
        if not isinstance(results, list):
            logger.warning("Unexpected LLM response format for %s.", symbol)
            return False

        for entry in results:
            if not isinstance(entry, dict):
                continue
            raw_id = entry.get("id")
            try:
                idx = int(raw_id) - 1
            except (TypeError, ValueError):
                continue
            if idx < 0 or idx >= len(batch):
                continue

            sentiment = _safe_float(entry.get("sentiment"), 0.0)
            confidence = _safe_float(entry.get("confidence"), 0.0)
            sentiment = _clamp(sentiment, -1.0, 1.0)
            confidence = _clamp(confidence, 0.0, 1.0)

            if confidence < self.min_confidence:
                sentiment = 0.0
            elif self.confidence_weighted:
                sentiment *= confidence

            batch[idx]["sentiment_score"] = sentiment
        return True

    def get_status(self):
        return dict(self.stats)
