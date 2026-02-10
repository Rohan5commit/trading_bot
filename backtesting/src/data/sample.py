from __future__ import annotations

import hashlib
from datetime import datetime

import numpy as np
import pandas as pd


def _base_price(symbol: str) -> float:
    digest = hashlib.md5(symbol.encode("utf-8")).hexdigest()
    return 50 + (int(digest[:6], 16) % 150)


def generate_sample_bars(symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
    freq = interval.lower()
    idx = pd.date_range(start=start, end=end, freq=freq, tz="UTC")
    if idx.empty:
        return pd.DataFrame()
    base = _base_price(symbol)
    trend = np.linspace(0, 5, len(idx))
    noise = np.sin(np.linspace(0, 6.28, len(idx)))
    close = base + trend + noise
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    volume = np.full(len(idx), 1_000_000)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )
