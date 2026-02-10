#!/usr/bin/env python3
"""
One-time bootstrap for S&P 500:
- Ensures each ticker has price history (limited window) in SQLite
- Builds features in-memory (no feature_store files)
- Trains/updates one OLS model per symbol

This is meant to be run manually (it can take a while the first time).
Daily scheduled runs should use `python3 main.py` (fast daily job).
"""

import os
import sys
import sqlite3
import time
import pandas as pd

from ingest_prices import PriceIngestor
from features import FeatureEngineer
from train import ModelManager


def main():
    config_path = "config.yaml"
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None

    ing = PriceIngestor(config_path)
    fe = FeatureEngineer(config_path)
    mm = ModelManager(config_path)

    universe = pd.read_csv("./universe/sp500.csv")
    tickers = [str(t).strip().upper() for t in universe["ticker"].tolist() if str(t).strip()]
    if limit:
        tickers = tickers[:limit]

    # Conservative throttle similar to daily job.
    key_count = len(ing.twelvedata_keys.keys())
    assumed_rpm = 5
    overall_rps = (key_count * assumed_rpm) / 60.0 if key_count else 0.0
    sleep_s = 1.0 / overall_rps if overall_rps > 0 else 0.6
    sleep_s = max(0.2, min(2.0, sleep_s))

    latest_market_date = ing.get_latest_market_date()

    for idx, sym in enumerate(tickers, 1):
        try:
            last = ing.get_latest_date_for_symbol(sym)
        except Exception:
            last = None

        df_prices = None
        src = str(getattr(ing, "price_source", "stooq") or "stooq").strip().lower()
        if src == "auto":
            src = "twelvedata" if ing.twelvedata_keys.keys() else "stooq"

        try:
            if last is None:
                df_prices = ing.fetch_twelvedata_daily(sym) if src == "twelvedata" else ing.fetch_stooq_data(sym)
            elif latest_market_date and last != latest_market_date:
                df_prices = ing.fetch_twelvedata_daily(sym, outputsize=10) if src == "twelvedata" else ing.fetch_stooq_latest(sym)
        except Exception:
            df_prices = None

        if df_prices is not None and not df_prices.empty:
            conn = sqlite3.connect(ing.db_path)
            try:
                ing._sqlite_upsert(type("Table", (), {"name": "prices"}), conn, df_prices.columns.tolist(), df_prices.values.tolist())
            finally:
                conn.close()

        # Train if missing/stale.
        try:
            if mm._should_retrain(sym):
                df_feat = fe.generate(sym)
                if df_feat is not None and not df_feat.empty:
                    mm.train_ols(sym, features_df=df_feat)
        except Exception:
            pass

        if idx % 25 == 0:
            print(f"Progress: {idx}/{len(tickers)}")

        time.sleep(sleep_s)

    # Keep disk usage small.
    try:
        mm.prune_models_keep_latest_only()
    except Exception:
        pass

    print("Bootstrap complete.")


if __name__ == "__main__":
    main()

