from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import json
import pandas as pd
import yfinance as yf

from src.engine.backtest import BacktestResult, _compute_metrics
from src.engine.portfolio import Portfolio
from src.utils.config import ensure_dirs


@dataclass(frozen=True)
class EarningsEvent:
    symbol: str
    earnings_ts: pd.Timestamp
    entry_ts: pd.Timestamp


def _load_cached_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_cached_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _get_market_caps(symbols: Iterable[str], cache_dir: Path) -> Dict[str, float]:
    cache_path = cache_dir / "market_caps.json"
    cache = _load_cached_json(cache_path)
    updated = False
    results: Dict[str, float] = {}

    for symbol in symbols:
        if symbol in cache:
            results[symbol] = float(cache[symbol])
            continue
        try:
            ticker = yf.Ticker(symbol)
            cap = None
            fast_info = getattr(ticker, "fast_info", None)
            if fast_info and "market_cap" in fast_info:
                cap = fast_info.get("market_cap")
            if cap is None:
                info = ticker.get_info()
                cap = info.get("marketCap")
            if cap is None:
                cap = 0.0
            cache[symbol] = float(cap)
            results[symbol] = float(cap)
            updated = True
        except Exception:
            cache[symbol] = 0.0
            results[symbol] = 0.0
            updated = True

    if updated:
        _write_cached_json(cache_path, cache)
    return results


def _get_earnings_dates(symbol: str, start: pd.Timestamp, end: pd.Timestamp, cache_dir: Path) -> List[pd.Timestamp]:
    cache_path = cache_dir / f"earnings_{symbol}.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if "earnings_date" in cached.columns:
            dates = pd.to_datetime(cached["earnings_date"], errors="coerce").dropna()
            return [d for d in dates if start <= d <= end]

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.get_earnings_dates(limit=60)
    except Exception:
        df = pd.DataFrame()

    dates: List[pd.Timestamp] = []
    if df is not None and not df.empty:
        if "Earnings Date" in df.columns:
            raw = df["Earnings Date"]
        else:
            raw = df.index
        dates = [pd.Timestamp(d).tz_localize(None) for d in raw]

    if dates:
        pd.DataFrame({"earnings_date": dates}).to_csv(cache_path, index=False)
    return [d for d in dates if start <= d <= end]


def _build_earnings_events(
    bars: Dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_market_cap: float,
    entry_days_before: int,
    cache_dir: Path,
) -> List[EarningsEvent]:
    symbols = list(bars.keys())
    market_caps = _get_market_caps(symbols, cache_dir)
    eligible = [s for s, cap in market_caps.items() if cap >= min_market_cap]
    events: List[EarningsEvent] = []

    for symbol in eligible:
        frame = bars.get(symbol)
        if frame is None or frame.empty:
            continue
        frame_index = frame.index.tz_localize(None)
        earnings_dates = _get_earnings_dates(symbol, start, end, cache_dir)
        if not earnings_dates:
            continue
        for earnings_ts in earnings_dates:
            prior = frame_index[frame_index < earnings_ts]
            if prior.empty:
                continue
            entry_ts = prior.max()
            if entry_ts < start or entry_ts > end:
                continue
            if entry_days_before > 1:
                pos = frame_index.get_indexer([entry_ts])[0]
                target_idx = pos - (entry_days_before - 1)
                if target_idx >= 0:
                    entry_ts = frame_index[target_idx]
                else:
                    continue
            events.append(EarningsEvent(symbol=symbol, earnings_ts=earnings_ts, entry_ts=entry_ts))
    return events


def run_earnings_backtest(
    bars: Dict[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    config: dict,
    output_dir: Path,
) -> BacktestResult:
    ensure_dirs(output_dir)
    strategy_cfg = config.get("strategy", {}).get("earnings", {})
    backtest_cfg = config.get("backtest", {})
    portfolio_cfg = config.get("portfolio", {})

    initial_capital = float(backtest_cfg.get("initial_capital", 100000))
    take_profit = float(strategy_cfg.get("take_profit_pct", 0.05))
    stop_loss = float(strategy_cfg.get("stop_loss_pct", 0.05))
    min_market_cap = float(strategy_cfg.get("min_market_cap", 5_000_000_000))
    entry_days_before = int(strategy_cfg.get("entry_days_before", 1))
    max_hold_days = int(strategy_cfg.get("max_hold_days", 10))
    position_size_pct = float(strategy_cfg.get("position_size_pct", 0.1))
    max_positions = int(portfolio_cfg.get("max_positions", 20))
    intraday_fill = str(strategy_cfg.get("intraday_fill", "stop_first"))

    start = pd.Timestamp(backtest_cfg.get("start_date")).tz_localize(None)
    end = pd.Timestamp(backtest_cfg.get("end_date")).tz_localize(None)

    # Normalize bar indices to tz-naive so comparisons/unions are stable across providers.
    norm_bars: Dict[str, pd.DataFrame] = {}
    for sym, frame in (bars or {}).items():
        if frame is None or frame.empty:
            continue
        f = frame.copy()
        idx = pd.to_datetime(f.index, errors="coerce")
        if hasattr(idx, "tz") and idx.tz is not None:
            idx = idx.tz_convert(None)
        else:
            try:
                idx = idx.tz_localize(None)
            except Exception:
                pass
        f.index = idx
        norm_bars[sym] = f
    bars = norm_bars

    if isinstance(benchmark, pd.DataFrame) and (not benchmark.empty):
        b = benchmark.copy()
        bidx = pd.to_datetime(b.index, errors="coerce")
        if hasattr(bidx, "tz") and bidx.tz is not None:
            bidx = bidx.tz_convert(None)
        else:
            try:
                bidx = bidx.tz_localize(None)
            except Exception:
                pass
        b.index = bidx
        benchmark = b

    data_dir = Path("data")
    cache_dir = data_dir / "cache"
    ensure_dirs(cache_dir)

    events = _build_earnings_events(
        bars=bars,
        start=start,
        end=end,
        min_market_cap=min_market_cap,
        entry_days_before=entry_days_before,
        cache_dir=cache_dir,
    )

    entries_by_date: Dict[pd.Timestamp, List[str]] = {}
    for event in events:
        entries_by_date.setdefault(event.entry_ts.normalize(), []).append(event.symbol)

    index = sorted(set().union(*[frame.index for frame in bars.values()]))
    portfolio = Portfolio(initial_capital)
    equity_rows: List[dict] = []
    meta: Dict[str, dict] = {}

    for timestamp in index:
        if timestamp < start or timestamp > end:
            continue

        day_key = timestamp.normalize()
        # Open entries at open price
        if day_key in entries_by_date:
            symbols_today = entries_by_date[day_key]
            available_slots = max(0, max_positions - len(portfolio.positions))
            for symbol in symbols_today:
                if available_slots <= 0:
                    break
                if symbol in portfolio.positions:
                    continue
                if symbol not in bars or timestamp not in bars[symbol].index:
                    continue
                price = float(bars[symbol].loc[timestamp, "open"])
                if price <= 0:
                    continue
                equity = portfolio.total_value({s: float(bars[s].loc[timestamp, "close"]) for s in portfolio.positions if timestamp in bars[s].index})
                target_value = equity * position_size_pct
                shares = target_value / price
                if shares <= 0:
                    continue
                portfolio.execute_trade(symbol, timestamp.to_pydatetime(), "BUY", shares, price)
                meta[symbol] = {
                    "entry_price": price,
                    "target_price": price * (1 + take_profit),
                    "stop_price": price * (1 - stop_loss),
                    "entry_time": timestamp.to_pydatetime(),
                }
                available_slots -= 1

        # Check exits
        for symbol in list(portfolio.positions.keys()):
            if symbol not in bars or timestamp not in bars[symbol].index:
                continue
            bar = bars[symbol].loc[timestamp]
            meta_row = meta.get(symbol)
            if not meta_row:
                continue
            high = float(bar.get("high", bar.get("close", 0)))
            low = float(bar.get("low", bar.get("close", 0)))
            close = float(bar.get("close", 0))
            tp_hit = high >= meta_row["target_price"]
            sl_hit = low <= meta_row["stop_price"]

            exit_price = None
            if tp_hit and sl_hit:
                exit_price = meta_row["stop_price"] if intraday_fill == "stop_first" else meta_row["target_price"]
            elif tp_hit:
                exit_price = meta_row["target_price"]
            elif sl_hit:
                exit_price = meta_row["stop_price"]
            else:
                holding_days = (timestamp.to_pydatetime() - meta_row["entry_time"]).days
                if max_hold_days and holding_days >= max_hold_days:
                    exit_price = close

            if exit_price is not None:
                shares = portfolio.positions[symbol].shares
                portfolio.execute_trade(symbol, timestamp.to_pydatetime(), "SELL", shares, exit_price)
                meta.pop(symbol, None)

        prices = {symbol: float(frame.loc[timestamp, "close"]) for symbol, frame in bars.items() if timestamp in frame.index}
        equity_rows.append(portfolio.mark_to_market(timestamp.to_pydatetime(), prices))

    # Close remaining positions at final close
    if equity_rows:
        last_ts = pd.Timestamp(equity_rows[-1]["timestamp"])
        for symbol in list(portfolio.positions.keys()):
            frame = bars.get(symbol)
            if frame is None:
                continue
            if last_ts in frame.index:
                price = float(frame.loc[last_ts, "close"])
            else:
                price = float(frame.iloc[-1]["close"])
            shares = portfolio.positions[symbol].shares
            portfolio.execute_trade(symbol, last_ts.to_pydatetime(), "SELL", shares, price)
        prices = {symbol: float(frame.loc[last_ts, "close"]) for symbol, frame in bars.items() if last_ts in frame.index}
        equity_rows.append(portfolio.mark_to_market(last_ts.to_pydatetime(), prices))

    equity = pd.DataFrame(equity_rows)
    trades = pd.DataFrame([trade.__dict__ for trade in portfolio.trades])
    metrics = _compute_metrics(equity, trades, config, benchmark=benchmark)

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
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    notes = config.get("report", {}).get("notes", "")
    earnings_notes = (
        "Earnings strategy: buy before earnings (entry_days_before={entry_days_before}), "
        "TP={take_profit:.0%}, SL={stop_loss:.0%}, min_market_cap=${min_cap:,.0f}. "
        "If TP and SL hit same day, fill={intraday_fill}. "
        "Positions close after {max_hold_days} days if neither hit.".format(
            entry_days_before=entry_days_before,
            take_profit=take_profit,
            stop_loss=stop_loss,
            min_cap=min_market_cap,
            intraday_fill=intraday_fill,
            max_hold_days=max_hold_days,
        )
    )
    if notes:
        notes = f"{notes}\n{earnings_notes}"
    else:
        notes = earnings_notes
    (output_dir / "strategy_notes.txt").write_text(notes, encoding="utf-8")

    return BacktestResult(trades=trades, equity=equity, metrics=metrics)
