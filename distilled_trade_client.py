from __future__ import annotations

import math
from typing import List, Optional

from ai_manager_memory import AIManagerMemory


LABEL_TO_SCORE = {
    "STRONG_SELL": -2.0,
    "SELL": -1.0,
    "NEUTRAL": 0.0,
    "BUY": 1.0,
    "STRONG_BUY": 2.0,
}


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


class DistilledTradeClient:
    """Lightweight local fallback that preserves manager continuity.

    This is intentionally deterministic and uses only the feature snapshot that
    the AI manager already receives, plus the shared SQLite memory journal.
    """

    def __init__(self, config: dict | None = None, manager_memory: AIManagerMemory | None = None):
        cfg = dict(config or {})
        ai_cfg = dict(cfg.get("ai_trading") or {})
        router_cfg = dict(ai_cfg.get("runtime_router") or {})
        self.backend = "distilled_local"
        self.model_name = str(router_cfg.get("distilled_model_name") or "distilled-feature-manager").strip()
        self.last_error = None
        self.last_model_used = self.model_name
        self.manager_memory = manager_memory or AIManagerMemory.from_config(cfg)
        self.lookback_days = max(1, int(router_cfg.get("memory_lookback_days", 120) or 120))

    @property
    def model_identifier(self) -> str:
        return self.model_name

    def is_ready(self) -> bool:
        self.last_error = None
        return True

    def wait_until_ready(self, timeout_seconds: int = 600, poll_seconds: float = 10.0) -> dict:
        self.last_error = None
        self.last_model_used = self.model_name
        return {"ok": True, "model": self.model_name, "backend": self.backend}

    def predict_candidate(self, candidate: dict) -> Optional[dict]:
        rows = self.predict_candidates([candidate])
        return rows[0] if rows else None

    def predict_candidates(self, candidates: List[dict]) -> List[Optional[dict]]:
        biases = self.manager_memory.symbol_side_bias()
        out = []
        for candidate in list(candidates or []):
            if not isinstance(candidate, dict):
                out.append(None)
                continue
            out.append(self._predict_one(candidate, biases))
        return out

    def _predict_one(self, candidate: dict, biases: dict[tuple[str, str], dict[str, float]]) -> dict:
        symbol = str(candidate.get("symbol") or "").strip().upper()
        components = self._component_scores(candidate)
        long_bias = float((biases.get((symbol, "LONG"), {}) or {}).get("bias", 0.0) or 0.0)
        short_bias = float((biases.get((symbol, "SHORT"), {}) or {}).get("bias", 0.0) or 0.0)
        memory_edge = long_bias - short_bias
        components["shared_memory"] = _clamp(memory_edge, -0.45, 0.45)

        current_side = str(candidate.get("current_position_side") or "").strip().upper()
        pre_score = sum(components.values())
        if current_side == "LONG" and pre_score > 0:
            components["position_continuity"] = 0.08
        elif current_side == "SHORT" and pre_score < 0:
            components["position_continuity"] = -0.08
        elif current_side in {"LONG", "SHORT"}:
            components["position_continuity"] = -0.05 if pre_score > 0 else 0.05
        else:
            components["position_continuity"] = 0.0

        score = sum(components.values())
        label = self._label_from_score(score)
        class_probabilities = self._class_probabilities(score)
        volatility_penalty = abs(float(candidate.get("volatility_20d") or 0.0) or 0.0)
        memory_confidence = max(abs(long_bias), abs(short_bias))
        confidence = _clamp(0.30 + abs(score) * 0.22 + memory_confidence * 0.18 - min(volatility_penalty * 2.0, 0.18), 0.18, 0.97)
        reason = self._reason_from_components(label, components)
        return {
            "label": label,
            "score": LABEL_TO_SCORE[label],
            "confidence": confidence,
            "reason": reason,
            "class_probabilities": class_probabilities,
        }

    def _component_scores(self, candidate: dict) -> dict[str, float]:
        def f(name: str) -> float:
            try:
                value = candidate.get(name)
                if value is None:
                    return 0.0
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        return_1d = f("return_1d")
        return_5d = f("return_5d")
        return_10d = f("return_10d")
        dist_ma_20 = f("dist_ma_20")
        dist_ma_50 = f("dist_ma_50")
        rsi_14 = f("rsi_14")
        volume_ratio = f("volume_ratio")
        news_sentiment = f("news_sentiment_7d")
        news_count = f("news_count_7d")

        components = {
            "momentum_5d": _clamp(return_5d * 14.0, -0.85, 0.85),
            "momentum_10d": _clamp(return_10d * 10.0, -0.90, 0.90),
            "fresh_day": _clamp(return_1d * 6.0, -0.35, 0.35),
            "trend_ma20": _clamp(dist_ma_20 * 5.5, -0.70, 0.70),
            "trend_ma50": _clamp(dist_ma_50 * 4.0, -0.55, 0.55),
            "volume_confirmation": _clamp((volume_ratio - 1.0) * 0.35, -0.30, 0.30),
            "news_tone": _clamp(news_sentiment * min(max(news_count, 1.0), 4.0) * 0.14, -0.35, 0.35),
        }

        if rsi_14 >= 72.0:
            components["rsi_regime"] = -0.18
        elif rsi_14 <= 28.0:
            components["rsi_regime"] = 0.18
        elif rsi_14 >= 58.0 and (dist_ma_20 > 0.0 or return_5d > 0.0):
            components["rsi_regime"] = 0.09
        elif 0.0 < rsi_14 <= 42.0 and (dist_ma_20 < 0.0 or return_5d < 0.0):
            components["rsi_regime"] = -0.09
        else:
            components["rsi_regime"] = 0.0

        return components

    @staticmethod
    def _label_from_score(score: float) -> str:
        if score >= 1.05:
            return "STRONG_BUY"
        if score >= 0.30:
            return "BUY"
        if score <= -1.05:
            return "STRONG_SELL"
        if score <= -0.30:
            return "SELL"
        return "NEUTRAL"

    @staticmethod
    def _class_probabilities(score: float) -> dict[str, float]:
        centers = {
            "STRONG_SELL": -1.6,
            "SELL": -0.7,
            "NEUTRAL": 0.0,
            "BUY": 0.7,
            "STRONG_BUY": 1.6,
        }
        weights = {}
        for label, center in centers.items():
            weights[label] = math.exp(-abs(float(score) - center) * 1.6)
        total = sum(weights.values()) or 1.0
        return {label: float(value / total) for label, value in weights.items()}

    @staticmethod
    def _reason_from_components(label: str, components: dict[str, float]) -> str:
        direction = 1 if LABEL_TO_SCORE[label] > 0 else (-1 if LABEL_TO_SCORE[label] < 0 else 0)
        ordered = sorted(components.items(), key=lambda item: abs(float(item[1] or 0.0)), reverse=True)
        positive = [name for name, value in ordered if float(value or 0.0) > 0][:3]
        negative = [name for name, value in ordered if float(value or 0.0) < 0][:3]

        phrase_map = {
            "momentum_5d": "recent momentum",
            "momentum_10d": "multi-day trend",
            "fresh_day": "latest session strength",
            "trend_ma20": "20-day trend",
            "trend_ma50": "50-day trend",
            "volume_confirmation": "volume confirmation",
            "news_tone": "news tone",
            "rsi_regime": "RSI regime",
            "shared_memory": "shared manager memory",
            "position_continuity": "current book context",
        }

        def humanize(names: list[str]) -> str:
            labels = [phrase_map.get(name, name.replace("_", " ")) for name in names if name]
            if not labels:
                return "feature balance"
            if len(labels) == 1:
                return labels[0]
            return ", ".join(labels[:-1]) + f" and {labels[-1]}"

        if direction > 0:
            return f"Distilled manager favors a long because {humanize(positive)} outweigh {humanize(negative)}."
        if direction < 0:
            return f"Distilled manager favors a short because {humanize(negative)} outweigh {humanize(positive)}."
        return f"Distilled manager stays neutral because {humanize(positive)} and {humanize(negative)} are balanced."
