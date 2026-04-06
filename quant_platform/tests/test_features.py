import pandas as pd

from data.indicators import compute_indicators


def test_compute_indicators_basic():
    df = pd.DataFrame({
        "open": [1, 2, 3, 4, 5],
        "high": [2, 3, 4, 5, 6],
        "low": [0.5, 1, 2, 3, 4],
        "close": [1.5, 2.5, 3.5, 4.5, 5.5],
        "volume": [100, 110, 120, 130, 140],
    })
    out = compute_indicators(df)
    assert "close" in out.columns
