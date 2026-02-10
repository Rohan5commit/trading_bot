#!/usr/bin/env python3
"""
One-time earnings backtest runner.
Strategy: Buy before earnings for companies with market cap > $5B,
with 5% take profit and 5% stop loss.
"""

import sys
import os
from pathlib import Path

# Add the backtesting directory to the path
sys.path.insert(0, str(Path(__file__).parent / "backtesting"))

import pandas as pd
from src.data.provenance import ProvenanceLogger
from src.data.yfinance_providers import YFinanceBarsProvider, YFinanceBenchmarkProvider, WikipediaUniverseProvider
from src.engine.earnings_backtest import run_earnings_backtest
from src.engine.email_report import build_email_report, send_email_report, write_email_report
from src.utils.config import ensure_dirs


def main():
    # Setup directories
    data_dir = Path("backtesting/data")
    report_dir = Path("backtesting/reports/latest")
    ensure_dirs(data_dir, report_dir)

    # Setup providers
    cache_dir = data_dir / "cache"
    provenance = ProvenanceLogger(Path("backtesting/logs/provenance.jsonl"))

    bars_provider = YFinanceBarsProvider(
        cache_dir=cache_dir,
        provenance=provenance,
        timezone="America/New_York",
        open_time="09:30",
        close_time="16:00",
        source_interval="1h",
        fallback_to_sample=False,
    )

    benchmark_provider = YFinanceBenchmarkProvider(
        cache_dir=cache_dir,
        provenance=provenance,
        source_interval="1h",
        fallback_to_sample=False,
    )

    universe_provider = WikipediaUniverseProvider(
        cache_dir=cache_dir,
        provenance=provenance,
        source_url="https://en.wikipedia.org/wiki/Nasdaq-100",
        as_of="2024-12-31",
    )

    # Get universe
    universe = universe_provider.get_universe()
    symbols = universe.members

    # Backtest period - use recent data
    start_date = "2024-01-01"
    end_date = "2025-12-31"
    interval = "4H"

    print(f"Fetching bars for {len(symbols)} symbols from {start_date} to {end_date}...")
    bars = bars_provider.get_bars(symbols, start_date, end_date, interval)
    print(f"Got bars for {len(bars)} symbols")

    print("Fetching benchmark (SPY)...")
    benchmark = benchmark_provider.get_benchmark(start_date, end_date, interval)
    print(f"Got benchmark with {len(benchmark)} bars")

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
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": 100000,
        },
        "portfolio": {
            "max_positions": 20,
        },
        "report": {
            "title": "Earnings Strategy Backtest",
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

    print("\nRunning earnings backtest...")
    result = run_earnings_backtest(bars, benchmark, config, report_dir)

    print(f"\nBacktest completed!")
    print(f"Total trades: {len(result.trades)}")
    print(f"Total P&L: ${result.metrics.get('total_pnl', 0):.2f}")
    print(f"Win rate: {result.metrics.get('win_rate', 0):.2%}")
    print(f"CAGR: {result.metrics.get('CAGR', 0):.2%}")
    print(f"Max drawdown: {result.metrics.get('max_drawdown', 0):.2%}")
    print(f"Sharpe ratio: {result.metrics.get('sharpe_ratio', 0):.2f}")

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
