#!/usr/bin/env python3
"""
Simple backtest runner using cached data.
Strategy: Buy on first day, hold with 5% TP/SL.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import json

# Add the backtesting directory to the path
sys.path.insert(0, str(Path(__file__).parent / "backtesting"))

from src.engine.backtest import run_backtest
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

    # Build config for simple strategy
    config = {
        "strategy": {
            "enabled": True,
            "type": "native",
            "name": "simple_momentum",
            "lookback_days": 20,
            "take_profit": {
                "enabled": True,
                "tiers": [
                    {"pct": 0.05, "trim": 1.0},  # 5% take profit, sell all
                ]
            },
            "exit_rules": {
                "time_based_days": 10,
                "indicator_based": "sma_cross"
            }
        },
        "backtest": {
            "start_date": pd.Timestamp(start_date, tz='UTC').isoformat(),
            "end_date": pd.Timestamp(end_date, tz='UTC').isoformat(),
            "initial_capital": 100000,
        },
        "portfolio": {
            "max_positions": 20,
            "rebalancing_cadence": "weekly",
        },
        "risk_rails": {
            "max_position_count": 20,
            "max_daily_turnover": 0.25,
            "max_gap_exposure": 0.05,
            "safety_switch": False,
        },
        "report": {
            "title": "Simple Momentum Backtest (5% TP/SL)",
            "notes": "Simple momentum strategy with 5% take profit and 10-day max hold",
        },
        "email": {
            "enabled": True,
            "smtp_host": "smtp.gmail.com",
            "smtp_port": 587,
            "from": "rohan.santhoshkumar1@gmail.com",
            "to": ["rohan.santhoshkumar@gmail.com"],
            "subject_prefix": "Backtest Report",
            "username_env": "SMTP_USERNAME",
            "password_env": "SMTP_PASSWORD",
            "timeout_seconds": 20,
        }
    }

    print("\nRunning backtest...")
    result = run_backtest(bars, {}, benchmark, config, report_dir)

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
