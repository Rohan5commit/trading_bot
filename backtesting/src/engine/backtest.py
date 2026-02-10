from __future__ import annotations

from dataclasses import dataclass
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from src.engine.portfolio import Portfolio
from src.engine.signals import Signal, normalize_signals
from src.engine.strategy import build_exit_rules, generate_signals
from src.utils.config import ensure_dirs


@dataclass(frozen=True)
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.DataFrame
    metrics: dict


def _is_rebalance_time(timestamp: pd.Timestamp, cadence: str) -> bool:
    if cadence == "daily":
        return True
    if cadence == "weekly":
        return timestamp.weekday() == 0
    if cadence == "monthly":
        return timestamp.day == 1
    return False


def _compute_metrics(equity: pd.DataFrame, trades: pd.DataFrame, config: dict, benchmark: pd.DataFrame | None = None) -> dict:
    equity_curve = equity.set_index("timestamp")["equity"].resample("1D").last().dropna()
    returns = equity_curve.pct_change().dropna()
    turnover = float(trades["shares"].abs().sum()) if not trades.empty else 0.0
    win_rate = float((trades["pnl"] > 0).mean()) if not trades.empty else 0.0
    if returns.empty:
        return {"CAGR": 0.0, "drawdown": 0.0, "volatility": 0.0, "turnover": turnover, "win_rate": win_rate}
    total_years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / total_years) - 1 if total_years > 0 else 0.0
    drawdown = ((equity_curve / equity_curve.cummax()) - 1).min()
    volatility = returns.std() * np.sqrt(252)
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0.0
    best_trade = float(trades["pnl"].max()) if not trades.empty else 0.0
    worst_trade = float(trades["pnl"].min()) if not trades.empty else 0.0
    total_pnl = float(trades["pnl"].sum()) if not trades.empty else 0.0
    metrics = {
        "CAGR": float(cagr),
        "drawdown": float(drawdown),
        "max_drawdown": float(drawdown),
        "volatility": float(volatility),
        "turnover": turnover,
        "win_rate": win_rate,
        "total_return": float(total_return),
        "total_pnl": total_pnl,
        "best_trade_pnl": best_trade,
        "worst_trade_pnl": worst_trade,
        "sharpe_ratio": float(sharpe_ratio),
    }
    if benchmark is not None and not benchmark.empty:
        bench_curve = benchmark["close"].resample("1D").last().dropna()
        aligned = pd.concat([equity_curve, bench_curve], axis=1).dropna()
        if not aligned.empty:
            bench = aligned.iloc[:, 1]
            bench_total_return = bench.iloc[-1] / bench.iloc[0] - 1
            bench_years = (bench.index[-1] - bench.index[0]).days / 365.25
            bench_cagr = (bench.iloc[-1] / bench.iloc[0]) ** (1 / bench_years) - 1 if bench_years > 0 else 0.0
            metrics["benchmark_cagr"] = float(bench_cagr)
            metrics["benchmark_total_return"] = float(bench_total_return)
            metrics["cagr_improvement"] = float(cagr - bench_cagr)
            metrics["total_return_improvement"] = float(total_return - bench_total_return)
    mc_cfg = config.get("backtest", {}).get("monte_carlo", {})
    if mc_cfg.get("enabled", False):
        metrics["monte_carlo"] = _monte_carlo_metrics(returns, mc_cfg)
    return metrics


def _monte_carlo_metrics(returns: pd.Series, mc_cfg: dict) -> dict:
    simulations = int(mc_cfg.get("simulations", 500))
    seed = mc_cfg.get("seed")
    if seed is not None:
        np.random.seed(int(seed))
    samples = []
    for _ in range(simulations):
        boot = returns.sample(n=len(returns), replace=True).reset_index(drop=True)
        equity = (1 + boot).cumprod()
        if equity.empty:
            continue
        years = max(1e-9, len(equity) / 252)
        cagr = float(equity.iloc[-1] ** (1 / years) - 1)
        drawdown = float(((equity / equity.cummax()) - 1).min())
        samples.append({"cagr": cagr, "drawdown": drawdown})

    if not samples:
        return {}
    cagr_vals = np.array([s["cagr"] for s in samples])
    dd_vals = np.array([s["drawdown"] for s in samples])
    return {
        "cagr_p05": float(np.percentile(cagr_vals, 5)),
        "cagr_p50": float(np.percentile(cagr_vals, 50)),
        "cagr_p95": float(np.percentile(cagr_vals, 95)),
        "drawdown_p05": float(np.percentile(dd_vals, 5)),
        "drawdown_p50": float(np.percentile(dd_vals, 50)),
        "drawdown_p95": float(np.percentile(dd_vals, 95)),
    }


def run_backtest(
    bars: Dict[str, pd.DataFrame],
    actions: Dict[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    config: dict,
    output_dir: Path,
    write_outputs: bool = True,
    trade_start: pd.Timestamp | None = None,
) -> BacktestResult:
    ensure_dirs(output_dir)
    cadence = config["portfolio"]["rebalancing_cadence"]
    max_positions = int(config["portfolio"]["max_positions"])
    initial_capital = float(config["backtest"]["initial_capital"])
    exit_rules = build_exit_rules(config["strategy"])

    index = sorted(set().union(*[frame.index for frame in bars.values()]))
    portfolio = Portfolio(initial_capital)
    equity_rows: List[dict] = []

    for i, timestamp in enumerate(index[:-1]):
        if trade_start is not None and timestamp < trade_start:
            continue
        next_timestamp = index[i + 1]
        if _is_rebalance_time(timestamp, cadence):
            signals = generate_signals({s: f.loc[:timestamp] for s, f in bars.items()}, config["strategy"])
            signals = normalize_signals(signals, max_positions)
            _execute_signals(signals, bars, portfolio, next_timestamp, exit_rules, config)
        _apply_corporate_actions(actions, portfolio, timestamp)
        prices = {symbol: frame.loc[timestamp, "close"] for symbol, frame in bars.items() if timestamp in frame.index}
        equity_rows.append(portfolio.mark_to_market(timestamp.to_pydatetime(), prices))

    equity = pd.DataFrame(equity_rows)
    trades = pd.DataFrame([trade.__dict__ for trade in portfolio.trades])
    metrics = _compute_metrics(equity, trades, config, benchmark=benchmark)

    if write_outputs:
        trades.to_csv(output_dir / "trades.csv", index=False)
        daily_equity = (
            equity.set_index("timestamp")["equity"].resample("1D").last().dropna().reset_index()
            if not equity.empty
            else pd.DataFrame(columns=["timestamp", "equity"])
        )
        if not benchmark.empty and not daily_equity.empty:
            benchmark_daily = benchmark.set_index(benchmark.index)["close"].resample("1D").last().dropna().reset_index()
            benchmark_daily.columns = ["timestamp", "spy_close"]
            daily_equity = daily_equity.merge(benchmark_daily, on="timestamp", how="left")
        daily_equity.to_csv(output_dir / "daily_equity.csv", index=False)
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, indent=2))

    return BacktestResult(trades=trades, equity=equity, metrics=metrics)


def _execute_signals(
    signals: Iterable[Signal],
    bars: Dict[str, pd.DataFrame],
    portfolio: Portfolio,
    timestamp: pd.Timestamp,
    exit_rules,
    config: dict,
) -> None:
    if config["risk_rails"]["safety_switch"]:
        return
    if any("stop" in tag.lower() for signal in signals for tag in signal.rationale_tags):
        raise ValueError("Stop-loss signals are not allowed")

    signals = list(signals)
    max_positions = int(config["risk_rails"]["max_position_count"])
    buy_signals = [s for s in signals if s.action == "BUY"]
    sell_signals = [s for s in signals if s.action == "SELL"]

    prices = {symbol: bars[symbol].loc[timestamp, "open"] for symbol in bars if timestamp in bars[symbol].index}
    equity = portfolio.total_value(prices)

    buy_signals = sorted(buy_signals, key=lambda s: s.confidence, reverse=True)[:max_positions]

    for signal in sell_signals:
        if signal.symbol in portfolio.positions:
            shares = portfolio.positions[signal.symbol].shares
            portfolio.execute_trade(signal.symbol, timestamp.to_pydatetime(), "SELL", shares, prices.get(signal.symbol, 0))

    max_turnover = float(config["risk_rails"]["max_daily_turnover"])
    total_turnover = 0.0
    for signal in buy_signals:
        if signal.symbol not in prices:
            continue
        gap_limit = float(config["risk_rails"]["max_gap_exposure"])
        prev_close = bars[signal.symbol].loc[timestamp - pd.Timedelta(hours=4), "close"] if timestamp - pd.Timedelta(hours=4) in bars[signal.symbol].index else prices[signal.symbol]
        gap = abs(prices[signal.symbol] / prev_close - 1) if prev_close else 0.0
        if gap > gap_limit:
            continue
        target_value = equity * signal.position_size
        total_turnover += target_value / equity if equity else 0.0
        if total_turnover > max_turnover:
            break
        shares = target_value / prices[signal.symbol] if prices[signal.symbol] > 0 else 0
        portfolio.execute_trade(signal.symbol, timestamp.to_pydatetime(), "BUY", shares, prices[signal.symbol])

    _apply_exit_rules(portfolio, prices, timestamp, exit_rules)


def _apply_exit_rules(portfolio: Portfolio, prices: Dict[str, float], timestamp: pd.Timestamp, exit_rules) -> None:
    to_sell = []
    for symbol, position in portfolio.positions.items():
        holding_days = (timestamp.to_pydatetime() - position.entry_time).days
        if holding_days >= exit_rules.time_based_days:
            to_sell.append(symbol)
            continue
        for tier in exit_rules.take_profit_tiers:
            if prices.get(symbol, 0) >= position.avg_cost * (1 + tier["pct"]):
                to_sell.append(symbol)
                break

    for symbol in to_sell:
        if symbol in prices:
            shares = portfolio.positions[symbol].shares
            portfolio.execute_trade(symbol, timestamp.to_pydatetime(), "SELL", shares, prices[symbol])


def _apply_corporate_actions(actions: Dict[str, pd.DataFrame], portfolio: Portfolio, timestamp: pd.Timestamp) -> None:
    for symbol, frame in actions.items():
        if frame.empty:
            continue
        ts = timestamp.tz_convert("UTC") if timestamp.tzinfo else timestamp.tz_localize("UTC")
        if frame.index.tz is None:
            frame_index = frame.index.tz_localize("UTC")
        else:
            frame_index = frame.index.tz_convert("UTC")
        frame_utc = frame.copy()
        frame_utc.index = frame_index
        if ts in frame_utc.index:
            row = frame_utc.loc[ts]
            dividend = float(row.get("dividends", 0))
            split = float(row.get("stock splits", 0))
            if dividend:
                portfolio.apply_dividend(symbol, dividend)
            if split:
                portfolio.apply_split(symbol, split)


def run_walk_forward(
    bars: Dict[str, pd.DataFrame],
    actions: Dict[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    config: dict,
    output_dir: Path,
) -> BacktestResult:
    wf_cfg = config.get("backtest", {}).get("walk_forward", {})
    train_days = int(wf_cfg.get("train_days", 60))
    test_days = int(wf_cfg.get("test_days", 20))
    step_days = int(wf_cfg.get("step_days", test_days))

    start_date = pd.Timestamp(config["backtest"]["start_date"])
    end_date = pd.Timestamp(config["backtest"]["end_date"])

    trades_list: list[pd.DataFrame] = []
    equity_list: list[pd.DataFrame] = []

    current = start_date
    while current + pd.Timedelta(days=train_days + test_days) <= end_date:
        train_start = current
        train_end = train_start + pd.Timedelta(days=train_days)
        test_start = train_end
        test_end = test_start + pd.Timedelta(days=test_days)

        best_lookback = _select_lookback(bars, actions, benchmark, config, train_start, train_end)

        config_override = json.loads(json.dumps(config))
        config_override["strategy"]["lookback_days"] = best_lookback

        bars_slice = _slice_bars(bars, train_start, test_end)
        actions_slice = _slice_frames(actions, train_start, test_end)
        benchmark_slice = _slice_frame(benchmark, train_start, test_end)

        trade_start_ts = test_start
        if bars_slice:
            sample_frame = next(iter(bars_slice.values()))
            trade_start_ts, _ = _coerce_range(sample_frame, test_start, test_start)

        result = run_backtest(
            bars_slice,
            actions_slice,
            benchmark_slice,
            config_override,
            output_dir,
            write_outputs=False,
            trade_start=trade_start_ts,
        )

        if not result.trades.empty:
            trades_list.append(result.trades)
        if not result.equity.empty:
            equity_list.append(result.equity)

        current += pd.Timedelta(days=step_days)

    trades = pd.concat(trades_list, ignore_index=True) if trades_list else pd.DataFrame()
    equity = pd.concat(equity_list, ignore_index=True) if equity_list else pd.DataFrame()

    metrics = _compute_metrics(equity, trades, config)

    trades.to_csv(output_dir / "trades.csv", index=False)
    daily_equity = (
        equity.set_index("timestamp")["equity"].resample("1D").last().dropna().reset_index()
        if not equity.empty
        else pd.DataFrame(columns=["timestamp", "equity"])
    )
    if not benchmark.empty and not daily_equity.empty:
        benchmark_daily = benchmark.set_index(benchmark.index)["close"].resample("1D").last().dropna().reset_index()
        benchmark_daily.columns = ["timestamp", "spy_close"]
        daily_equity = daily_equity.merge(benchmark_daily, on="timestamp", how="left")
    daily_equity.to_csv(output_dir / "daily_equity.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, indent=2))

    return BacktestResult(trades=trades, equity=equity, metrics=metrics)


def _select_lookback(
    bars: Dict[str, pd.DataFrame],
    actions: Dict[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    config: dict,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
) -> int:
    nested_cfg = config.get("backtest", {}).get("nested_cv", {})
    if not nested_cfg.get("enabled", False):
        return int(config["strategy"]["lookback_days"])

    grid = nested_cfg.get("parameter_grid", {}).get("lookback_days", [config["strategy"]["lookback_days"]])
    folds = int(nested_cfg.get("folds", 3))
    fold_size = max(1, int((train_end - train_start).days / (folds + 1)))

    best_score = -float("inf")
    best_lookback = int(config["strategy"]["lookback_days"])

    for lookback in grid:
        scores: list[float] = []
        for fold in range(folds):
            inner_train_end = train_start + pd.Timedelta(days=fold_size * (fold + 1))
            inner_test_end = inner_train_end + pd.Timedelta(days=fold_size)
            if inner_test_end > train_end:
                break
            bars_slice = _slice_bars(bars, train_start, inner_test_end)
            actions_slice = _slice_frames(actions, train_start, inner_test_end)
            benchmark_slice = _slice_frame(benchmark, train_start, inner_test_end)
            config_override = json.loads(json.dumps(config))
            config_override["strategy"]["lookback_days"] = int(lookback)
            trade_start_ts = inner_train_end
            if bars_slice:
                sample_frame = next(iter(bars_slice.values()))
                trade_start_ts, _ = _coerce_range(sample_frame, inner_train_end, inner_train_end)
            result = run_backtest(
                bars_slice,
                actions_slice,
                benchmark_slice,
                config_override,
                output_dir=Path("/tmp"),
                write_outputs=False,
                trade_start=trade_start_ts,
            )
            scores.append(float(result.metrics.get("CAGR", 0.0)))
        avg = float(np.mean(scores)) if scores else -float("inf")
        if avg > best_score:
            best_score = avg
            best_lookback = int(lookback)

    return best_lookback


def _slice_bars(bars: Dict[str, pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, pd.DataFrame]:
    sliced: Dict[str, pd.DataFrame] = {}
    for symbol, frame in bars.items():
        start_ts, end_ts = _coerce_range(frame, start, end)
        sliced_frame = frame.loc[start_ts:end_ts]
        if not sliced_frame.empty:
            sliced[symbol] = sliced_frame
    return sliced


def _slice_frames(frames: Dict[str, pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, pd.DataFrame]:
    sliced: Dict[str, pd.DataFrame] = {}
    for symbol, frame in frames.items():
        if frame.empty:
            sliced[symbol] = frame
            continue
        start_ts, end_ts = _coerce_range(frame, start, end)
        sliced_frame = frame.loc[start_ts:end_ts]
        sliced[symbol] = sliced_frame
    return sliced


def _slice_frame(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if frame.empty:
        return frame
    start_ts, end_ts = _coerce_range(frame, start, end)
    return frame.loc[start_ts:end_ts]


def _coerce_range(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    if frame.index.tz is not None:
        start_ts = start.tz_localize(frame.index.tz) if start.tzinfo is None else start.tz_convert(frame.index.tz)
        end_ts = end.tz_localize(frame.index.tz) if end.tzinfo is None else end.tz_convert(frame.index.tz)
    else:
        start_ts = start.tz_convert("UTC").tz_localize(None) if start.tzinfo is not None else start
        end_ts = end.tz_convert("UTC").tz_localize(None) if end.tzinfo is not None else end
    return start_ts, end_ts
