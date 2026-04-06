from __future__ import annotations

import pandas as pd

try:
    import pandas_market_calendars as mcal
except Exception:  # pragma: no cover
    mcal = None


def trading_days(start: str, end: str, calendar: str = "NYSE") -> pd.DatetimeIndex:
    if mcal is None:
        return pd.bdate_range(start=start, end=end)
    cal = mcal.get_calendar(calendar)
    schedule = cal.schedule(start_date=start, end_date=end)
    return schedule.index
