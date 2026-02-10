from pathlib import Path

import pandas as pd

from src.engine.backtest import run_backtest


def make_bars(symbol: str):
    idx = pd.date_range("2024-01-01", periods=4, freq="4h", tz="UTC")
    frame = pd.DataFrame({
        "open": [10, 11, 11, 11],
        "high": [10, 11, 11, 11],
        "low": [10, 11, 11, 11],
        "close": [10, 11, 11, 11],
        "volume": [100] * 4,
    }, index=idx)
    frame.attrs["symbol"] = symbol
    return frame


def base_config():
    return {
        "portfolio": {"rebalancing_cadence": "daily", "max_positions": 1},
        "risk_rails": {"safety_switch": False, "max_position_count": 1, "max_daily_turnover": 1.0, "max_gap_exposure": 1.0},
        "strategy": {"lookback_days": 1, "take_profit": {"enabled": False}, "exit_rules": {"time_based_days": 10}},
        "backtest": {"initial_capital": 1000},
    }


def test_dividend_applied_in_backtest(tmp_path: Path):
    bars = {"AAA": make_bars("AAA")}
    actions = {"AAA": pd.DataFrame({"dividends": [1.0], "stock splits": [0.0]}, index=[bars["AAA"].index[2]])}
    benchmark = pd.DataFrame()

    result = run_backtest(bars, actions, benchmark, base_config(), output_dir=tmp_path)
    assert result.equity.iloc[-1]["equity"] > 1000
