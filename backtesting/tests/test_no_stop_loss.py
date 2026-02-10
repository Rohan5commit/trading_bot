from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from src.engine.backtest import _execute_signals
from src.engine.portfolio import Portfolio
from src.engine.signals import Signal


def test_no_stop_loss_guardrail():
    idx = pd.date_range("2024-01-01", periods=2, freq="4h", tz="UTC")
    bars = {
        "AAA": pd.DataFrame({
            "open": [10, 10],
            "high": [10, 10],
            "low": [10, 10],
            "close": [10, 10],
            "volume": [100, 100],
        }, index=idx)
    }
    portfolio = Portfolio(1000)
    signal = Signal("AAA", datetime(2024, 1, 1), "BUY", "next_open", 0.5, 0.5, ["stop_loss"], "4 bars")
    config = {
        "risk_rails": {"safety_switch": False, "max_position_count": 1, "max_daily_turnover": 1.0, "max_gap_exposure": 1.0},
    }
    with pytest.raises(ValueError):
        _execute_signals([signal], bars, portfolio, idx[1], exit_rules=type("E", (), {"time_based_days": 1, "take_profit_tiers": []})(), config=config)
