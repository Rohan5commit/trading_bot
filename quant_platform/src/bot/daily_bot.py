from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import schedule
import yaml

from model.inference_client import ModelInferenceClient, InferenceConfig


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _latest_trading_day() -> str:
    today = datetime.utcnow().date()
    if today.weekday() >= 5:
        today -= timedelta(days=today.weekday() - 4)
    return str(today - timedelta(days=1))


def run_once(config_path: str) -> None:
    cfg = load_config(config_path)
    watchlist = cfg.get("watchlist", [])
    backend = cfg.get("backend", "nim_api")
    output_dir = Path(cfg.get("output_dir", "reports/bot"))
    output_dir.mkdir(parents=True, exist_ok=True)

    client = ModelInferenceClient(InferenceConfig(backend=backend))
    date = _latest_trading_day()

    rows = []
    for ticker in watchlist:
        result = client.predict(ticker, date)
        rows.append(result.model_dump())

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / f"signals_{date}.csv", index=False)

    db_path = output_dir / "predictions.db"
    with sqlite3.connect(db_path) as conn:
        df.to_sql("predictions", conn, if_exists="append", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bot.yaml")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if args.once:
        run_once(args.config)
        return

    cfg = load_config(args.config)
    schedule.every().day.at(cfg.get("run_time_sgt", "08:00")).do(run_once, args.config)
    while True:
        schedule.run_pending()


if __name__ == "__main__":
    main()
