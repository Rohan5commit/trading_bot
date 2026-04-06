from __future__ import annotations

import os
from io import StringIO
import pandas as pd
import requests


def _load_csv(path_or_url: str) -> pd.DataFrame:
    if path_or_url.startswith("http"):
        resp = requests.get(path_or_url, timeout=60)
        resp.raise_for_status()
        return pd.read_csv(StringIO(resp.text))
    return pd.read_csv(path_or_url)


class TradingViewAdapter:
    def __init__(self, path_env: str = "TRADINGVIEW_CSV_PATH", url_env: str = "TRADINGVIEW_CSV_URL") -> None:
        self.path = os.getenv(path_env)
        self.url = os.getenv(url_env)

    def get_tradingview_features(self, ticker: str) -> pd.DataFrame:
        source = self.url or self.path
        if not source:
            return pd.DataFrame()
        try:
            df = _load_csv(source)
        except Exception:
            return pd.DataFrame()
        if df.empty:
            return df
        if "ticker" not in df.columns or "date" not in df.columns:
            return pd.DataFrame()
        df = df[df["ticker"] == ticker].copy()
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        return df.drop(columns=["ticker"], errors="ignore")
