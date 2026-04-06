from __future__ import annotations

from typing import Dict, List

import pandas as pd
import yfinance as yf

from .hf_yahoo_finance import HfYahooFinanceAdapter


def _format(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


class OhlcvAdapter:
    def __init__(
        self,
        hf_enabled: bool = True,
        hf_cache_dir: str = "data/raw/hf_yahoo_finance",
        allow_fallbacks: bool = True,
    ) -> None:
        self.hf = HfYahooFinanceAdapter(enabled=hf_enabled, cache_dir=hf_cache_dir)
        self.allow_fallbacks = allow_fallbacks

    def get_ohlcv(self, ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
        if interval == "1d":
            frame = self.hf.get_stock_prices(ticker, start, end)
            if not frame.empty:
                return frame
        if not self.allow_fallbacks:
            return pd.DataFrame()
        df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
        if df.empty:
            return df
        return _format(df)

    def get_ohlcv_batch(
        self, tickers: List[str], start: str, end: str, interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        if not tickers:
            return {}
        if interval == "1d":
            hf_frames = self.hf.get_stock_prices_batch(tickers, start, end)
            if not self.allow_fallbacks:
                return hf_frames
        else:
            hf_frames = {ticker: pd.DataFrame() for ticker in tickers}

        data = yf.download(
            " ".join(tickers),
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            group_by="ticker",
            threads=True,
            progress=False,
        )
        if data.empty:
            return {ticker: pd.DataFrame() for ticker in tickers}

        out: Dict[str, pd.DataFrame] = {}
        if isinstance(data.columns, pd.MultiIndex):
            level0 = set(data.columns.get_level_values(0))
            for ticker in tickers:
                if not hf_frames[ticker].empty:
                    out[ticker] = hf_frames[ticker]
                elif ticker in level0:
                    frame = data[ticker]
                    out[ticker] = _format(frame) if not frame.empty else pd.DataFrame()
                else:
                    out[ticker] = pd.DataFrame()
        else:
            if not hf_frames[tickers[0]].empty:
                out[tickers[0]] = hf_frames[tickers[0]]
            else:
                out[tickers[0]] = _format(data)
        return out

    def get_multi_timeframes(self, ticker: str, start: str, end: str) -> Dict[str, pd.DataFrame]:
        frames = {}
        for interval in ["1m", "5m", "15m", "1h", "1d", "1wk"]:
            try:
                frames[interval] = self.get_ohlcv(ticker, start, end, interval=interval)
            except Exception:
                frames[interval] = pd.DataFrame()
        return frames
