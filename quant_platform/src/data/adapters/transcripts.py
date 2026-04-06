from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from .hf_yahoo_finance import HfYahooFinanceAdapter


class TranscriptsAdapter:
    def __init__(
        self,
        base_path: str = "data/raw/transcripts",
        hf_enabled: bool = True,
        hf_cache_dir: str = "data/raw/hf_yahoo_finance",
        allow_fallbacks: bool = True,
    ) -> None:
        self.base_path = Path(base_path)
        self.hf = HfYahooFinanceAdapter(enabled=hf_enabled, cache_dir=hf_cache_dir)
        self.allow_fallbacks = allow_fallbacks

    def get_transcript_features(self, ticker: str) -> pd.DataFrame:
        hf_frame = self.hf.get_transcript_features(ticker)
        if not hf_frame.empty:
            return hf_frame
        if not self.allow_fallbacks:
            return pd.DataFrame()
        path = self.base_path / f"{ticker}.json"
        if not path.exists():
            return pd.DataFrame()
        with path.open() as f:
            data = json.load(f)
        rows = []
        for item in data:
            rows.append({
                "date": item.get("date"),
                "mgmt_sentiment": item.get("mgmt_sentiment"),
                "guidance_keywords": item.get("guidance_keywords"),
                "uncertainty": item.get("uncertainty"),
                "analyst_sentiment": item.get("analyst_sentiment"),
                "surprise_delta": item.get("surprise_delta"),
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date").sort_index()

    def get_transcript_features_batch(self, tickers: list[str]) -> dict[str, pd.DataFrame]:
        hf_frames = self.hf.get_transcript_features_batch(tickers)
        if not self.allow_fallbacks:
            return hf_frames

        out: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            frame = hf_frames.get(ticker, pd.DataFrame())
            out[ticker] = frame if not frame.empty else self.get_transcript_features(ticker)
        return out
