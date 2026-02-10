import json
from pathlib import Path

import pandas as pd

from src.engine.backtest import run_backtest


def make_bars(symbol: str, closes):
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="4h", tz="UTC")
    frame = pd.DataFrame({
        "open": closes,
        "high": closes,
        "low": closes,
        "close": closes,
        "volume": [100] * len(closes),
    }, index=idx)
    frame.attrs["symbol"] = symbol
    return frame


def base_config():
    return {
        "portfolio": {"rebalancing_cadence": "daily", "max_positions": 2},
        "risk_rails": {"safety_switch": False, "max_position_count": 2, "max_daily_turnover": 1.0, "max_gap_exposure": 1.0},
        "strategy": {"lookback_days": 2, "take_profit": {"enabled": False}, "exit_rules": {"time_based_days": 10}},
        "backtest": {"initial_capital": 1000},
    }


def test_regression_snapshot(tmp_path: Path):
    bars = {
        "AAA": make_bars("AAA", [10, 10, 11, 12, 13, 14]),
        "BBB": make_bars("BBB", [10, 10, 9, 8, 7, 6]),
    }
    actions = {"AAA": pd.DataFrame(), "BBB": pd.DataFrame()}
    benchmark = pd.DataFrame()

    result = run_backtest(bars, actions, benchmark, base_config(), output_dir=tmp_path)
    expected = json.loads((Path("tests/fixtures/expected_snapshot.json")).read_text())

    assert len(result.trades) == expected["trade_count"]
    final_equity = result.equity.iloc[-1]["equity"]
    assert abs(final_equity - expected["final_equity"]) < 1e-2

    metrics = result.metrics
    for key, value in expected["metrics"].items():
        assert abs(metrics[key] - value) < 1e-2
