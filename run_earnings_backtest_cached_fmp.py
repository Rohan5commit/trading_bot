#!/usr/bin/env python3
"""
Earnings backtest runner using cached bar data and Financial Modeling Prep API for earnings dates.
Strategy: Buy before earnings for companies with market cap > $5B,
with 5% take profit and 5% stop loss.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import json
from collections import Counter

# Load .env from repo root so SENDER_EMAIL/SENDER_PASSWORD are available.
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
except Exception:
    pass

# Add backtesting directory to the path
sys.path.insert(0, str(Path(__file__).parent / "backtesting"))

from src.engine.earnings_backtest_fmp import run_earnings_backtest
from src.engine.email_report import build_email_report, send_email_report, write_email_report
from src.utils.config import ensure_dirs


def load_cached_bars(cache_dir: Path, start_date: str, end_date: str) -> dict:
    """Load cached bar data from parquet files."""
    bars = {}
    # Prefer TwelveData bars (real prices); yfinance_bars has historically included sample/fallback caches.
    cache_path = cache_dir / "cache" / "twelvedata_bars"
    if not cache_path.exists():
        cache_path = cache_dir / "twelvedata_bars"
    if not cache_path.exists():
        cache_path = cache_dir / "cache" / "yfinance_bars"
    if not cache_path.exists():
        cache_path = cache_dir / "yfinance_bars"
    
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
                    # Skip known sample-like caches (volume constant at 1,000,000).
                    try:
                        vol = df["volume"]
                        if vol.nunique() == 1 and int(float(vol.iloc[0])) == 1_000_000:
                            continue
                    except Exception:
                        pass
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
        cache_path = cache_dir / "yfinance_benchmark"
    
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

def _pick_largest_cached_window(cache_dir: Path) -> tuple[str, str, str]:
    """
    Choose the (start_date, end_date, interval) window that has the most cached bar files,
    so the backtest runs on the largest cached universe without downloading anything.
    """
    # Prefer TwelveData cache windows first.
    bars_dir = cache_dir / "cache" / "twelvedata_bars"
    if not bars_dir.exists():
        bars_dir = cache_dir / "twelvedata_bars"
    if not bars_dir.exists():
        bars_dir = cache_dir / "cache" / "yfinance_bars"
    if not bars_dir.exists():
        bars_dir = cache_dir / "yfinance_bars"
    if not bars_dir.exists():
        return ("2022-01-01", "2022-06-30", "4H")

    counts = Counter()
    for fp in bars_dir.glob("*.parquet"):
        parts = fp.stem.split("_")
        if len(parts) < 4:
            continue
        # SYMBOL_START_END_INTERVAL
        start, end, interval = parts[-3], parts[-2], parts[-1]
        counts[(start, end, interval)] += 1
    if not counts:
        return ("2022-01-01", "2022-06-30", "4H")
    (start, end, interval), _ = counts.most_common(1)[0]
    return (start, end, interval)

def _symbols_with_cached_earnings(cache_dir: Path) -> set[str]:
    earnings_dir = cache_dir / "cache"
    if not earnings_dir.exists():
        earnings_dir = cache_dir
    out = set()
    for fp in earnings_dir.glob("earnings_*.csv"):
        sym = fp.stem.replace("earnings_", "").strip().upper()
        if sym:
            out.add(sym)
    return out


def main():
    # Setup directories
    data_dir = Path("backtesting/data")
    report_dir = Path("backtesting/reports/latest")
    ensure_dirs(data_dir, report_dir)

    # Load cached data
    print("Loading cached bar data...")
    # Optional CLI override: python3 run_earnings_backtest_cached_fmp.py 2025-09-23 2026-01-21 4H
    if len(sys.argv) >= 3:
        start_date = str(sys.argv[1]).strip()
        end_date = str(sys.argv[2]).strip()
        interval = str(sys.argv[3]).strip() if len(sys.argv) >= 4 else "4H"
    else:
        start_date, end_date, interval = _pick_largest_cached_window(data_dir / "cache")
    print(f"Using cached window: {start_date} -> {end_date} interval={interval}")
    
    bars = load_cached_bars(data_dir, start_date, end_date)
    # Restrict to symbols we already have cached earnings for (avoids extra network calls + storage growth).
    allowed = _symbols_with_cached_earnings(data_dir / "cache")
    if allowed:
        bars = {k: v for k, v in bars.items() if k in allowed}
    print(f"\nLoaded bars for {len(bars)} symbols")

    print("\nLoading cached benchmark data...")
    benchmark = load_cached_benchmark(data_dir, start_date, end_date)
    print(f"Loaded benchmark with {len(benchmark)} bars")

    if not bars:
        print("ERROR: No bar data loaded. Cannot run backtest.")
        return

    # Build config for earnings strategy
    # Use timezone-aware timestamps to match cached data
    sender = os.getenv("SENDER_EMAIL") or "rohan.santhoshkumar1@gmail.com"
    recipient = os.getenv("RECIPIENT_EMAIL") or "rohan.santhoshkumar@gmail.com"
    config = {
        "strategy": {
            "enabled": True,
            "type": "earnings_event",
            "earnings": {
                "take_profit_pct": 0.04,  # 4% take profit
                "stop_loss_pct": 0.04,  # 4% stop loss
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
            "title": "Earnings Strategy Backtest",
            "notes": "Buy before earnings for companies with market cap > $5B, 4% TP/SL",
        },
        "email": {
            "enabled": True,
            "smtp_host": "smtp.gmail.com",
            "smtp_port": 587,
            "from": sender,
            "to": [recipient],
            "subject_prefix": "Earnings Backtest (TP=4%, SL=4%)",
            "username_env": "SENDER_EMAIL",
            "password_env": "SENDER_PASSWORD",
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
    print(f"Total return: {result.metrics.get('total_return', 0):.2%}")

    # Generate and send email report
    print("\nGenerating email report...")
    subject, body = build_email_report(config, report_dir)
    write_email_report(subject, body, report_dir / "email_report.txt")

    print("Sending email report...")
    sent = send_email_report(config, subject, body, report_dir=report_dir)
    print("Email sent successfully!" if sent else "Email not sent (missing credentials or SMTP failure).")

    # Cleanup this one-off report to save space (keep caches for future runs).
    if sent:
        for name in ["trades.csv", "metrics.json", "daily_equity.csv", "email_report.txt", "strategy_notes.txt", "signals.json", "data_status.json", "report.html"]:
            p = report_dir / name
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

    print(f"\nReport saved to: {report_dir}")
    print(f"Email report: {report_dir / 'email_report.txt'}")
    print(f"Trades: {report_dir / 'trades.csv'}")
    print(f"Metrics: {report_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
