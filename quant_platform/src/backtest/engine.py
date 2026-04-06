from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from pydantic import BaseModel

from data.feature_store import FeatureStore, FeatureStoreConfig
from model.inference_client import ModelInferenceClient, InferenceConfig
from .metrics import (
    cagr,
    sharpe,
    sortino,
    max_drawdown,
    volatility,
    win_rate,
    profit_factor,
    turnover,
    alpha_beta,
)
from .portfolio import allocate_weights
from .strategies import momentum_breakout, mean_reversion_pullback, factor_ranking
from .plotting import plot_equity_curve, plot_drawdown


class BacktestConfig(BaseModel):
    universe: List[str]
    start: str
    end: str
    benchmark: str = "SPY"
    strategy: Dict
    risk: Dict
    costs: Dict
    portfolio: Dict
    output_dir: str = "reports/backtest"


class BacktestEngine:
    def __init__(self, config: BacktestConfig, feature_config: FeatureStoreConfig | None = None) -> None:
        self.config = config
        self.feature_store = FeatureStore(feature_config)
        self.inference = ModelInferenceClient(InferenceConfig())

    def run(self) -> Dict:
        cfg = self.config
        df = self.feature_store.build_feature_frame(cfg.universe, cfg.start, cfg.end, normalize=True)
        if df.empty:
            raise ValueError("No feature data")

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["date", "ticker"])
        df = df.set_index(["date", "ticker"]).sort_index()
        dates = df.index.get_level_values(0).unique()

        prices = df["close"].unstack("ticker")
        returns = prices.pct_change().shift(-1)

        signal_frame = self._precompute_signals(df)

        weights = []
        equity = [cfg.portfolio.get("initial_capital", 1_000_000.0)]
        trade_log = []
        prev_weights = pd.Series(0.0, index=cfg.universe)
        entry_price = {}

        max_gross = cfg.risk.get("max_gross", 1.0)
        max_net = cfg.risk.get("max_net", 1.0)
        stop_loss = cfg.risk.get("stop_loss")
        take_profit = cfg.risk.get("take_profit")

        for date in dates[:-1]:
            scores = signal_frame.loc[date]
            w = allocate_weights(scores, cfg.strategy.get("top_k", 3), cfg.strategy.get("bottom_k", 0), cfg.risk.get("max_position", 0.2))
            w = w.reindex(cfg.universe).fillna(0.0)

            gross = w.abs().sum()
            if gross > max_gross and gross > 0:
                w = w * (max_gross / gross)
            net = w.sum()
            if abs(net) > max_net and abs(net) > 0:
                w = w * (max_net / abs(net))

            for ticker in cfg.universe:
                price = prices.loc[date, ticker]
                if prev_weights.get(ticker, 0.0) == 0.0 and w[ticker] != 0.0:
                    entry_price[ticker] = price
                if prev_weights.get(ticker, 0.0) != 0.0 and w[ticker] == 0.0:
                    entry_price.pop(ticker, None)

                if ticker in entry_price and w[ticker] != 0.0 and stop_loss is not None and take_profit is not None:
                    direction = 1 if w[ticker] > 0 else -1
                    ret = direction * (price - entry_price[ticker]) / entry_price[ticker]
                    if ret <= -stop_loss or ret >= take_profit:
                        w[ticker] = 0.0
                        entry_price.pop(ticker, None)

            weights.append(w)

            day_ret = (returns.loc[date] * w).sum()
            turnover_cost = (w - prev_weights).abs().sum() * (cfg.costs.get("commission_bps", 1.0) + cfg.costs.get("slippage_bps", 2.0)) / 10000
            equity.append(equity[-1] * (1 + day_ret - turnover_cost))

            for ticker in cfg.universe:
                if w[ticker] != prev_weights.get(ticker, 0.0):
                    trade_log.append({"date": date, "ticker": ticker, "weight": w[ticker]})

            prev_weights = w

        equity_series = pd.Series(equity[1:], index=dates[:-1])
        returns_series = equity_series.pct_change().fillna(0)
        weights_df = pd.DataFrame(weights, index=dates[:-1])
        trade_df = pd.DataFrame(trade_log)

        benchmark_prices = prices[cfg.benchmark] if cfg.benchmark in prices.columns else prices.mean(axis=1)
        benchmark_returns = benchmark_prices.pct_change().shift(-1).reindex(returns_series.index).fillna(0)

        alpha, beta = alpha_beta(returns_series, benchmark_returns)

        metrics = {
            "CAGR": cagr(equity_series),
            "Sharpe": sharpe(returns_series),
            "Sortino": sortino(returns_series),
            "MaxDrawdown": max_drawdown(equity_series),
            "Volatility": volatility(returns_series),
            "WinRate": win_rate(trade_df),
            "ProfitFactor": profit_factor(trade_df),
            "Turnover": turnover(weights_df),
            "Alpha": alpha,
            "Beta": beta,
        }

        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        equity_series.to_csv(output_dir / "equity_curve.csv")
        trade_df.to_csv(output_dir / "trade_log.csv", index=False)
        pd.Series(metrics).to_json(output_dir / "metrics.json")
        plot_equity_curve(equity_series, str(output_dir / "equity_curve.png"))
        plot_drawdown(equity_series, str(output_dir / "drawdown.png"))

        return metrics

    def _precompute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        strategy = self.config.strategy.get("name", "ai_ensemble")
        if strategy == "ai_ensemble":
            return self._ai_signals(df)

        frame = df.reset_index()
        if strategy == "momentum_breakout":
            frame["signal"] = momentum_breakout(frame)
        elif strategy == "mean_reversion_pullback":
            frame["signal"] = mean_reversion_pullback(frame)
        elif strategy == "factor_ranking":
            cols = self.config.strategy.get("factors", ["rsi_14", "macd", "atr_14"])
            frame["signal"] = factor_ranking(frame, cols)
        else:
            raise ValueError(f"Unknown strategy {strategy}")

        return frame.pivot(index="date", columns="ticker", values="signal")

    def _ai_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        records = []
        for (date, ticker), row in df.iterrows():
            result = self.inference.predict(ticker, str(date))
            records.append({"date": date, "ticker": ticker, "score": result.score})
        frame = pd.DataFrame(records)
        return frame.pivot(index="date", columns="ticker", values="score")


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/backtest.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    engine = BacktestEngine(BacktestConfig(**cfg))
    engine.run()
