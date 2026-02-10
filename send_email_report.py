#!/usr/bin/env python3
"""
Send email report for backtest results.
"""

import sys
import os
from pathlib import Path

# Add the backtesting directory to the path
sys.path.insert(0, str(Path(__file__).parent / "backtesting"))

from src.engine.email_report import send_email_report


def main():
    config = {
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

    # Read the email report
    report_dir = Path("backtesting/reports/latest")
    email_report_path = report_dir / "email_report.txt"

    if not email_report_path.exists():
        print(f"Email report not found at {email_report_path}")
        return

    with open(email_report_path, 'r') as f:
        subject = f.readline().strip()
        # Skip empty line
        if not subject:
            subject = f.readline().strip()
        body = f.read()

    print(f"Sending email report...")
    print(f"Subject: {subject}")
    print(f"Body length: {len(body)} characters")

    send_email_report(config, subject, body)
    print("Email sent successfully!")


if __name__ == "__main__":
    main()
