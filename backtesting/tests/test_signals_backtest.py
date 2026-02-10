from pathlib import Path

import pandas as pd

from src.engine.backtest import run_backtest
from src.engine.signals import Signal, normalize_signals


def make_bars(symbol: str):
    idx = pd.date_range("2024-01-01", periods=6, freq="4h", tz="UTC")
    frame = pd.DataFrame({
        "open": [10, 10, 10, 10, 12, 12],
        "high": [10, 10, 10, 10, 12, 12],
        "low": [10, 10, 10, 10, 12, 12],
        "close": [10, 10, 10, 10, 12, 12],
        "volume": [100] * 6,
    }, index=idx)
    frame.attrs["symbol"] = symbol
    return frame


def base_config():
    return {
        "portfolio": {"rebalancing_cadence": "daily", "max_positions": 2},
        "risk_rails": {"safety_switch": False, "max_position_count": 2, "max_daily_turnover": 1.0, "max_gap_exposure": 1.0},
        "strategy": {"lookback_days": 3, "take_profit": {"enabled": False}, "exit_rules": {"time_based_days": 10}},
        "backtest": {"initial_capital": 1000},
    }


def test_equal_weight_sizing():
    signals = [
        Signal("A", pd.Timestamp("2024-01-01").to_pydatetime(), "BUY", "next_open", 0.5, 0.3, ["momentum"], "4 bars"),
        Signal("B", pd.Timestamp("2024-01-01").to_pydatetime(), "BUY", "next_open", 0.2, 0.3, ["momentum"], "4 bars"),
    ]
    normalized = normalize_signals(signals, 2)
    assert all(abs(s.position_size - 0.5) < 1e-6 for s in normalized if s.action == "BUY")


def test_no_lookahead_trade_timing(tmp_path: Path):
    bars = {"AAA": make_bars("AAA")}
    actions = {"AAA": pd.DataFrame()}
    benchmark = pd.DataFrame()
    result = run_backtest(bars, actions, benchmark, base_config(), output_dir=tmp_path)
    assert not result.trades.empty
    first_trade_time = pd.to_datetime(result.trades.iloc[0]["timestamp"])
    assert first_trade_time >= bars["AAA"].index[4].to_pydatetime()
