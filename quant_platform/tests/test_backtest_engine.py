import pandas as pd

from backtest.metrics import cagr, max_drawdown


def test_metrics():
    equity = pd.Series([100, 110, 105, 120], index=pd.date_range("2023-01-01", periods=4))
    assert cagr(equity) != 0
    assert max_drawdown(equity) <= 0
