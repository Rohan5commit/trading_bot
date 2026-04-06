from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from pydantic import BaseModel

from .feature_store import FeatureStore, FeatureStoreConfig
from .utils import ensure_dir

LABELS = ["STRONG_SELL", "SELL", "NEUTRAL", "BUY", "STRONG_BUY"]


class CorpusConfig(BaseModel):
    start: str = "2010-01-01"
    end: str = "2024-12-31"
    forward_windows: List[int] = [1, 5, 20]
    output_tabular: str = "data/corpus/tabular.parquet"
    output_text: str = "data/corpus/text_corpus.jsonl"
    generate_labels: bool = True
    generate_text: bool = True
    label_window: int = 5


class CorpusBuilder:
    def __init__(self, feature_config: FeatureStoreConfig | None = None) -> None:
        self.feature_store = FeatureStore(feature_config)

    def build(self, tickers: Iterable[str], config: CorpusConfig) -> None:
        features = self.feature_store.build_feature_frame(
            tickers, config.start, config.end, normalize=True
        )
        if features.empty:
            raise ValueError("No features available")
        self.build_from_frame(features, config)

    def build_from_frame(self, features: pd.DataFrame, config: CorpusConfig) -> None:
        if features.empty:
            raise ValueError("No features available")
        enriched = self._add_forward_returns(features, config.forward_windows)
        if config.generate_labels:
            enriched = self._add_labels(enriched, window=config.label_window)
        self._write_tabular(enriched, config.output_tabular)
        if config.generate_text:
            if not config.generate_labels:
                raise ValueError("Text corpus requires labels. Set generate_labels=True.")
            self._write_text(enriched, config.output_text)

    def _add_forward_returns(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        df = df.sort_values(["ticker", "date"])
        close = pd.to_numeric(df["close"], errors="coerce").replace(0, np.nan)
        for window in windows:
            shifted = df.groupby("ticker")["close"].shift(-window)
            df[f"forward_{window}d"] = shifted / close - 1.0
        return df

    def _add_labels(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        df = df.copy()
        target = f"forward_{window}d"

        def label_group(group: pd.DataFrame) -> pd.Series:
            if group[target].isna().all():
                return pd.Series([np.nan] * len(group), index=group.index)
            return pd.qcut(group[target], 5, labels=range(5))

        df["label"] = df.groupby("date", group_keys=False).apply(label_group)
        df["label"] = df["label"].map(lambda x: LABELS[int(x)] if pd.notna(x) else np.nan)
        return df

    def _write_tabular(self, df: pd.DataFrame, path: str) -> None:
        ensure_dir(Path(path).parent)
        df.to_parquet(path, index=False)

    def _write_text(self, df: pd.DataFrame, path: str) -> None:
        ensure_dir(Path(path).parent)
        with open(path, "w", encoding="utf-8") as handle:
            for _, row in df.dropna(subset=["label"]).iterrows():
                record = {"text": self._format_prompt(row), "label": row["label"]}
                handle.write(json.dumps(record) + "\n")

    def _format_prompt(self, row: pd.Series) -> str:
        return f"""TICKER: {row['ticker']}
DATE: {row['date'].date()}
PRICE_ACTION: close={row.get('close', np.nan):.4f}, volume={row.get('volume', np.nan):.2f}
INDICATORS: rsi_14={row.get('rsi_14', np.nan):.2f}, macd={row.get('macd', np.nan):.4f}, atr_14={row.get('atr_14', np.nan):.4f}
FUNDAMENTALS: pe={row.get('pe', np.nan)}, pb={row.get('pb', np.nan)}, roe={row.get('roe', np.nan)}
RECENT_NEWS: count={row.get('news_count', 0)}, sentiment={row.get('news_sentiment', 0.0)}
EARNINGS_CONTEXT: surprise={row.get('surprise_delta', np.nan)}
SENTIMENT: social={row.get('social_sentiment', np.nan)}
MACRO: vix={row.get('vix', np.nan)}, fedfunds={row.get('fedfunds', np.nan)}
QUESTION: Given all of the above, classify the expected 5-day return as: STRONG_BUY | BUY | NEUTRAL | SELL | STRONG_SELL
ANSWER: {row['label']}"""
