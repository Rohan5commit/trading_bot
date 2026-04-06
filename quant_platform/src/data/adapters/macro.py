from __future__ import annotations

import os
import pandas as pd
import yfinance as yf

try:
    from fredapi import Fred
except Exception:  # pragma: no cover
    Fred = None


def _to_series(df: pd.DataFrame, name: str) -> pd.Series | None:
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(-1):
            close = df.xs("Close", axis=1, level=-1)
            if isinstance(close, pd.DataFrame) and close.shape[1] >= 1:
                close = close.iloc[:, 0]
            return close.rename(name)
    if "Close" in df.columns:
        return df["Close"].rename(name)
    if "close" in df.columns:
        return df["close"].rename(name)
    return None


class MacroAdapter:
    def get_macro_frame(self, start: str, end: str) -> pd.DataFrame:
        frames = []
        if Fred is not None and os.getenv("FRED_API_KEY"):
            fred = Fred(api_key=os.getenv("FRED_API_KEY"))
            series = {
                "fedfunds": "FEDFUNDS",
                "cpi": "CPIAUCSL",
                "unrate": "UNRATE",
                "dgs2": "DGS2",
                "dgs10": "DGS10",
            }
            for name, code in series.items():
                try:
                    data = fred.get_series(code, observation_start=start, observation_end=end)
                    frame = data.to_frame(name)
                    frames.append(frame)
                except Exception:
                    pass

        market_series = {
            "vix": "^VIX",
            "dxy": "DX-Y.NYB",
            "gold": "GC=F",
            "crude": "CL=F",
        }
        for name, ticker in market_series.items():
            try:
                df = yf.download(ticker, start=start, end=end, interval="1d", progress=False)
                series = _to_series(df, name)
                if series is not None:
                    frames.append(series.to_frame())
            except Exception:
                pass

        if not frames:
            return pd.DataFrame()
        frame = pd.concat(frames, axis=1).sort_index().ffill()
        return frame