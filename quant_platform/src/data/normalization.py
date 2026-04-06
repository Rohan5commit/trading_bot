from __future__ import annotations

import pandas as pd


def rolling_zscore(df: pd.DataFrame, window: int = 252, min_periods: int = 20) -> pd.DataFrame:
    mean = df.rolling(window=window, min_periods=min_periods).mean()
    std = df.rolling(window=window, min_periods=min_periods).std()
    return (df - mean) / std.replace(0.0, pd.NA)
