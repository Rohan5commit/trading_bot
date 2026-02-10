import pandas as pd

from src.data.bars import filter_market_hours, resample_to_interval


def test_market_hours_filter():
    idx = pd.date_range("2024-01-01 08:00", periods=5, freq="1h", tz="UTC")
    frame = pd.DataFrame({
        "open": [1, 2, 3, 4, 5],
        "high": [1, 2, 3, 4, 5],
        "low": [1, 2, 3, 4, 5],
        "close": [1, 2, 3, 4, 5],
        "volume": [10, 10, 10, 10, 10],
    }, index=idx)
    filtered = filter_market_hours(frame, "America/New_York", "09:30", "16:00")
    assert all(filtered.index.tz_convert("America/New_York").time >= pd.Timestamp("09:30").time())


def test_resample_to_interval():
    idx = pd.date_range("2024-01-01 09:30", periods=4, freq="1h", tz="UTC")
    frame = pd.DataFrame({
        "open": [1, 2, 3, 4],
        "high": [2, 3, 4, 5],
        "low": [1, 1, 2, 3],
        "close": [2, 3, 4, 5],
        "volume": [10, 10, 10, 10],
    }, index=idx)
    resampled = resample_to_interval(frame, "4h", offset="9h30min")
    assert len(resampled) == 1
    assert resampled.iloc[0]["open"] == 1
    assert resampled.iloc[0]["close"] == 5
