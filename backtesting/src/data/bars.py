from __future__ import annotations

from datetime import time
from typing import Iterable

import pandas as pd


def filter_market_hours(frame: pd.DataFrame, timezone: str, open_time: str, close_time: str) -> pd.DataFrame:
    market_open = time.fromisoformat(open_time)
    market_close = time.fromisoformat(close_time)
    localized = frame.tz_convert(timezone)
    within = (localized.index.time >= market_open) & (localized.index.time <= market_close)
    return localized.loc[within]


def resample_to_interval(frame: pd.DataFrame, interval: str, offset: str | None = None) -> pd.DataFrame:
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    if offset:
        return frame.resample(interval, offset=offset).agg(agg).dropna()
    return frame.resample(interval).agg(agg).dropna()


def align_to_market_calendar(frame: pd.DataFrame, timezone: str) -> pd.DataFrame:
    import pandas_market_calendars as mcal

    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=frame.index.min().date(), end_date=frame.index.max().date())
    trading_days = schedule.index.tz_localize(timezone)
    return frame[frame.index.normalize().isin(trading_days.normalize())]
