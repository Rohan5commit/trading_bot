#!/usr/bin/env python3
"""
Earnings backtest runner using mock earnings dates (quarterly pattern).
Strategy: Buy before earnings for companies with market cap > $5B,
with 5% take profit and 5% stop loss.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import json
from datetime import timedelta

# Add backtesting directory to the path
sys.path.insert(0, str(Path(__file__).parent / "backtesting"))

from src.engine.backtest import BacktestResult, _compute_metrics
from src.engine.portfolio import Portfolio
from src.engine.email_report import build_email_report, send_email_report, write_email_report
from src.utils.config import ensure_dirs


def load_cached_bars(cache_dir: Path, start_date: str, end_date: str) -> dict:
    """Load cached bar data from parquet files."""
    bars = {}
    cache_path = cache_dir / "cache" / "yfinance_bars"
    
    if not cache_path.exists():
        print(f"Cache directory not found: {cache_path}")
        return bars
    
    # Use UTC timestamps to match cached data
    start_ts = pd.Timestamp(start_date, tz='UTC')
    end_ts = pd.Timestamp(end_date, tz='UTC')
    
    # Load all parquet files
    for file_path in cache_path.glob("*.parquet"):
        # Extract symbol from filename (format: SYMBOL_YYYY-MM-DD_YYYY-MM-DD_4H.parquet)
        parts = file_path.stem.split('_')
        if len(parts) >= 4:
            symbol = parts[0]
            file_start = pd.Timestamp(parts[1], tz='UTC')
            file_end = pd.Timestamp(parts[2], tz='UTC')
            
            # Check if file overlaps with our date range
            if file_end >= start_ts and file_start <= end_ts:
                try:
                    df = pd.read_parquet(file_path)
                    # Filter to date range
                    df = df[(df.index >= start_ts) & (df.index <= end_ts)]
                    if not df.empty:
                        bars[symbol] = df
                        print(f"Loaded {symbol}: {len(df)} bars from {file_path.name}")
                except Exception as e:
                    print(f"Error loading {file_path.name}: {e}")
    
    return bars


def load_cached_benchmark(cache_dir: Path, start_date: str, end_date: str) -> pd.DataFrame:
    """Load cached benchmark data from parquet files."""
    cache_path = cache_dir / "cache" / "yfinance_benchmark"
    
    if not cache_path.exists():
        print(f"Benchmark cache directory not found: {cache_path}")
        return pd.DataFrame()
    
    # Use UTC timestamps to match cached data
    start_ts = pd.Timestamp(start_date, tz='UTC')
    end_ts = pd.Timestamp(end_date, tz='UTC')
    
    # Try to find SPY benchmark file
    for file_path in cache_path.glob("SPY_*.parquet"):
        try:
            df = pd.read_parquet(file_path)
            # Filter to date range
            df = df[(df.index >= start_ts) & (df.index <= end_ts)]
            if not df.empty:
                print(f"Loaded SPY benchmark: {len(df)} bars from {file_path.name}")
                return df
        except Exception as e:
            print(f"Error loading benchmark {file_path.name}: {e}")
    
    return pd.DataFrame()


def generate_mock_earnings_dates(bars: dict, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Generate mock earnings dates based on quarterly pattern."""
    earnings_dates = {}
    
    # Ensure start/end are timezone-aware
    if start.tz is None:
        start = start.tz_localize('UTC')
    if end.tz is None:
        end = end.tz_localize('UTC')
    
    for symbol, df in bars.items():
        if df.empty:
            continue
        
        # Get first date and make it timezone-naive for comparison
        first_date = df.index[0]
        if first_date.tz is not None:
            first_date = first_date.tz_localize(None)
        
        # Generate quarterly earnings dates (every ~90 days)
        # Start from first available date + 30 days
        current_date = first_date + timedelta(days=30)
        
        dates = []
        while pd.Timestamp(current_date, tz='UTC') <= end:
            ts = pd.Timestamp(current_date, tz='UTC')
            if ts >= start and ts <= end:
                dates.append(ts)
            current_date += timedelta(days=90)  # Quarterly earnings
        
        if dates:
            earnings_dates[symbol] = dates
            print(f"{symbol}: {len(dates)} mock earnings dates")
    
    return earnings_dates


def run_earnings_backtest(
    bars: dict,
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

    start = pd.Timestamp(backtest_cfg.get("start_date"))
    end = pd.Timestamp(backtest_cfg.get("end_date"))

    # Generate mock earnings dates
    earnings_dates = generate_mock_earnings_dates(bars, start, end)
    
    # Build earnings events
    events = []
    for symbol, dates in earnings_dates.items():
        if symbol not in bars:
            continue
        frame = bars[symbol]
        frame_index = frame.index
        
        for earnings_ts in dates:
            # Find entry date (entry_days_before days before earnings)
            prior = frame_index[frame_index < earnings_ts]
            if prior.empty:
                continue
            entry_ts = prior.max()
            if entry_ts < start or entry_ts > end:
                continue
            
            events.append({
                "symbol": symbol,
                "earnings_ts": earnings_ts,
                "entry_ts": entry_ts
            })
    
    print(f"\nTotal earnings events: {len(events)}")
    
    # Group entries by date
    entries_by_date = {}
    for event in events:
        day_key = pd.Timestamp(event["entry_ts"]).normalize()
        entries_by_date.setdefault(day_key, []).append(event["symbol"])
    
    print(f"Unique entry dates: {len(entries_by_date)}")
    
    # Get all unique timestamps
    index = sorted(set().union(*[frame.index for frame in bars.values()]))
    portfolio = Portfolio(initial_capital)
    equity_rows = []
    meta = {}

    for timestamp in index:
        if timestamp < start or timestamp > end:
            continue

        day_key = pd.Timestamp(timestamp).normalize()
        
        # Open entries at open price
        if day_key in entries_by_date:
            symbols_today = entries_by_date[day_key]
            available_slots = max(0, max_positions - len(portfolio.positions))
            print(f"{day_key.date()}: Entering {len(symbols_today)} positions (slots: {available_slots})")
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
                print(f"  BUY {symbol}: {shares:.2f} shares @ ${price:.2f}")

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
            exit_reason = None
            if tp_hit and sl_hit:
                exit_price = meta_row["stop_price"] if intraday_fill == "stop_first" else meta_row["target_price"]
                exit_reason = "both"
            elif tp_hit:
                exit_price = meta_row["target_price"]
                exit_reason = "take_profit"
            elif sl_hit:
                exit_price = meta_row["stop_price"]
                exit_reason = "stop_loss"
            else:
                holding_days = (timestamp.to_pydatetime() - meta_row["entry_time"]).days
                if max_hold_days and holding_days >= max_hold_days:
                    exit_price = close
                    exit_reason = "time_exit"

            if exit_price is not None:
                shares = portfolio.positions[symbol].shares
                pnl = (exit_price - meta_row["entry_price"]) * shares
                portfolio.execute_trade(symbol, timestamp.to_pydatetime(), "SELL", shares, exit_price)
                meta.pop(symbol, None)
                print(f"  SELL {symbol}: {shares:.2f} shares @ ${exit_price:.2f} ({exit_reason}) P&L: ${pnl:.2f}")

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
            print(f"  FINAL SELL {symbol}: {shares:.2f} shares @ ${price:.2f}")
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
        "Earnings strategy (MOCK EARNINGS): buy before earnings (entry_days_before={entry_days_before}), "
        "TP={take_profit:.0%}, SL={stop_loss:.0%}, min_market_cap=${min_cap:,.0f}. "
        "If TP and SL hit same day, fill={intraday_fill}. "
        "Positions close after {max_hold_days} days if neither hit. "
        "NOTE: Using mock quarterly earnings dates (every 90 days) for demonstration.".format(
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


def main():
    # Setup directories
    data_dir = Path("backtesting/data")
    report_dir = Path("backtesting/reports/latest")
    ensure_dirs(data_dir, report_dir)

    # Load cached data
    print("Loading cached bar data...")
    # Use 2022 period which has more data
    start_date = "2022-01-01"
    end_date = "2022-06-30"
    
    bars = load_cached_bars(data_dir, start_date, end_date)
    print(f"\nLoaded bars for {len(bars)} symbols")

    print("\nLoading cached benchmark data...")
    benchmark = load_cached_benchmark(data_dir, start_date, end_date)
    print(f"Loaded benchmark with {len(benchmark)} bars")

    if not bars:
        print("ERROR: No bar data loaded. Cannot run backtest.")
        return

    # Build config for earnings strategy
    config = {
        "strategy": {
            "enabled": True,
            "type": "earnings_event",
            "earnings": {
                "take_profit_pct": 0.05,  # 5% take profit
                "stop_loss_pct": 0.05,  # 5% stop loss
                "min_market_cap": 5_000_000_000,  # $5B market cap
                "entry_days_before": 1,  # Buy 1 day before earnings
                "max_hold_days": 10,  # Max hold 10 days
                "position_size_pct": 0.1,  # 10% position size
                "intraday_fill": "stop_first",  # If both TP and SL hit, fill stop first
            }
        },
        "backtest": {
            "start_date": pd.Timestamp(start_date, tz='UTC').isoformat(),
            "end_date": pd.Timestamp(end_date, tz='UTC').isoformat(),
            "initial_capital": 100000,
        },
        "portfolio": {
            "max_positions": 20,
        },
        "report": {
            "title": "Earnings Strategy Backtest (Mock Earnings)",
            "notes": "Buy before earnings for companies with market cap > $5B, 5% TP/SL",
        },
        "email": {
            "enabled": True,
            "smtp_host": "smtp.gmail.com",
            "smtp_port": 587,
            "from": "rohan.santhoshkumar1@gmail.com",
            "to": ["rohan.santhoshkumar@gmail.com"],
            "subject_prefix": "Earnings Backtest",
            "username_env": "SMTP_USERNAME",
            "password_env": "SMTP_PASSWORD",
            "timeout_seconds": 20,
        }
    }

    print("\nRunning earnings backtest with mock earnings dates...")
    result = run_earnings_backtest(bars, benchmark, config, report_dir)

    print(f"\nBacktest completed!")
    print(f"Total trades: {len(result.trades)}")
    print(f"Total P&L: ${result.metrics.get('total_pnl', 0):.2f}")
    print(f"Win rate: {result.metrics.get('win_rate', 0):.2%}")
    print(f"CAGR: {result.metrics.get('CAGR', 0):.2%}")
    print(f"Max drawdown: {result.metrics.get('max_drawdown', 0):.2%}")
    print(f"Sharpe ratio: {result.metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Total return: {result.metrics.get('total_return', 0):.2%}")

    # Generate and send email report
    print("\nGenerating email report...")
    subject, body = build_email_report(config, report_dir)
    write_email_report(subject, body, report_dir / "email_report.txt")

    print("Sending email report...")
    send_email_report(config, subject, body)
    print("Email sent successfully!")

    print(f"\nReport saved to: {report_dir}")
    print(f"Email report: {report_dir / 'email_report.txt'}")
    print(f"Trades: {report_dir / 'trades.csv'}")
    print(f"Metrics: {report_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
