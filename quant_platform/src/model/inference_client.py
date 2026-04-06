from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import xgboost as xgb

from pydantic import BaseModel

from data.feature_store import FeatureStore, FeatureStoreConfig
from data.corpus_builder import LABELS
from .ensemble import combine_scores
from .nim_client import NIMClient
from .schema import SignalResult, SupportingFactors


LABEL_TO_SCORE = {
    "STRONG_SELL": -1.0,
    "SELL": -0.5,
    "NEUTRAL": 0.0,
    "BUY": 0.5,
    "STRONG_BUY": 1.0,
}


class InferenceConfig(BaseModel):
    artifacts_path: str = "artifacts"
    backend: str = "nim_api"  # nim_api or local
    nim_model: str = "meta/llama-3.1-8b-instruct"


class ModelInferenceClient:
    def __init__(self, config: InferenceConfig | None = None, feature_config: FeatureStoreConfig | None = None) -> None:
        self.config = config or InferenceConfig()
        self.feature_store = FeatureStore(feature_config)
        self.artifacts_path = Path(self.config.artifacts_path)
        self.tabular_model = self._load_tabular_model()
        self.scaler = self._load_optional("feature_scaler.pkl")
        self.label_encoder = self._load_optional("label_encoder.pkl")
        self.nim = None
        if self.config.backend == "nim_api":
            try:
                self.nim = NIMClient(self.config.nim_model)
            except Exception:
                self.config.backend = "local"

    def predict(self, ticker: str, date: str) -> SignalResult:
        features = self.feature_store.get_features(ticker, date)
        tabular_score = self._tabular_score(features)
        llm_score = self._llm_score(ticker, date, features)
        technical_score = self._technical_score(features)
        final_score = combine_scores(tabular_score, llm_score, technical_score)
        direction = "LONG" if final_score > 0.1 else "SHORT" if final_score < -0.1 else "FLAT"
        confidence = float(min(1.0, max(0.0, abs(final_score))))
        return SignalResult(
            ticker=ticker,
            date=str(date),
            score=float(final_score),
            direction=direction,
            confidence=confidence,
            factors=SupportingFactors(
                tabular_score=float(tabular_score),
                llm_score=float(llm_score),
                technical_score=float(technical_score),
            ),
        )

    def _tabular_score(self, features: Dict) -> float:
        if self.tabular_model is None:
            return 0.0
        feature_keys = sorted([k for k in features.keys() if k not in ("ticker", "date")])
        vector = np.array([features.get(k, np.nan) for k in feature_keys], dtype=float)
        vector = np.nan_to_num(vector, nan=0.0)
        if self.scaler is not None:
            vector = self.scaler.transform([vector])[0]
        dmatrix = xgb.DMatrix([vector])
        preds = self.tabular_model.predict(dmatrix)
        if preds.ndim == 2:
            scores = np.linspace(-1, 1, preds.shape[1])
            return float((preds[0] * scores).sum())
        return float(preds[0])

    def _llm_score(self, ticker: str, date: str, features: Dict) -> float:
        prompt = self._build_prompt(ticker, date, features)
        if self.config.backend == "nim_api" and self.nim:
            response = self.nim.classify(prompt)
            label = response.get("label", "NEUTRAL")
            return LABEL_TO_SCORE.get(label, 0.0)
        return 0.0

    def _technical_score(self, features: Dict) -> float:
        rsi = features.get("rsi_14") or 50
        macd = features.get("macd") or 0
        ema9 = features.get("ema_9") or 0
        ema21 = features.get("ema_21") or 0
        score = 0.0
        if rsi < 30:
            score += 0.3
        if rsi > 70:
            score -= 0.3
        if macd > 0:
            score += 0.2
        if ema9 > ema21:
            score += 0.2
        return score

    def _build_prompt(self, ticker: str, date: str, features: Dict) -> str:
        return f"""TICKER: {ticker}
DATE: {date}
PRICE_ACTION: close={features.get('close')}, volume={features.get('volume')}
INDICATORS: rsi_14={features.get('rsi_14')}, macd={features.get('macd')}, atr_14={features.get('atr_14')}
FUNDAMENTALS: pe={features.get('pe')}, pb={features.get('pb')}, roe={features.get('roe')}
RECENT_NEWS: count={features.get('news_count')}, sentiment={features.get('news_sentiment')}
EARNINGS_CONTEXT: surprise={features.get('surprise_delta')}
SENTIMENT: social={features.get('social_sentiment')}
MACRO: vix={features.get('vix')}, fedfunds={features.get('fedfunds')}
QUESTION: Given all of the above, classify the expected 5-day return as: STRONG_BUY | BUY | NEUTRAL | SELL | STRONG_SELL
ANSWER:"""

    def _load_tabular_model(self):
        path = self.artifacts_path / "tabular_model.ubj"
        if not path.exists():
            return None
        return xgb.Booster(model_file=str(path))

    def _load_optional(self, name: str):
        path = self.artifacts_path / name
        if not path.exists():
            return None
        return joblib.load(path)
