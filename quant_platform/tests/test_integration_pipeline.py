import pandas as pd
from types import SimpleNamespace

from backtest.engine import BacktestEngine, BacktestConfig


def test_integration_backtest():
    config = BacktestConfig(
        universe=["AAA", "BBB"],
        start="2023-01-01",
        end="2023-01-10",
        benchmark="AAA",
        strategy={"name": "ai_ensemble", "top_k": 1, "bottom_k": 0},
        risk={"max_position": 1.0},
        costs={"commission_bps": 0.0, "slippage_bps": 0.0},
        portfolio={"initial_capital": 1000},
    )
    engine = BacktestEngine(config)

    def fake_features(*args, **kwargs):
        dates = pd.date_range("2023-01-02", periods=5, freq="B")
        data = []
        for d in dates:
            for t in ["AAA", "BBB"]:
                data.append({"date": d, "ticker": t, "close": 100, "rsi_14": 50, "macd": 0, "atr_14": 1})
        return pd.DataFrame(data)

    engine.feature_store.build_feature_frame = fake_features
    engine.inference.predict = lambda ticker, date: SimpleNamespace(score=0.1)

    metrics = engine.run()
    assert "CAGR" in metrics
