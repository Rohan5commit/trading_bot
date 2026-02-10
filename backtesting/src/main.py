from __future__ import annotations

import argparse

from src.engine.pipeline import run_backtest_pipeline, run_eod_pipeline
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Quant model pipeline")
    parser.add_argument("command", choices=["backtest", "report", "run_eod", "email_report", "run_signals", "run_scheduler"])
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.command in {"backtest", "report"}:
        run_backtest_pipeline(config)
    elif args.command == "run_eod":
        run_eod_pipeline(config)
    elif args.command == "email_report":
        from src.engine.email_report import build_email_report, send_email_report, write_email_report
        from pathlib import Path

        report_dir = Path("reports/latest")
        subject, body = build_email_report(config, report_dir)
        write_email_report(subject, body, report_dir / "email_report.txt")
        send_email_report(config, subject, body)
    elif args.command == "run_signals":
        from src.engine.pipeline import run_signals_pipeline

        run_signals_pipeline(config)
    elif args.command == "run_scheduler":
        from src.engine.scheduler import run_daily_signal_scheduler

        run_daily_signal_scheduler(config)


if __name__ == "__main__":
    main()
