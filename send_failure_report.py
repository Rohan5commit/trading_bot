from __future__ import annotations

import argparse
import html as _html
import os
import smtplib
from datetime import datetime
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from zoneinfo import ZoneInfo


SGT = ZoneInfo("Asia/Singapore")


def _now_sgt() -> datetime:
    return datetime.now(SGT)


def _smtp_settings() -> tuple[str, int, str, str, str]:
    server = str(os.getenv("SMTP_SERVER") or "smtp.gmail.com").strip()
    port = int(str(os.getenv("SMTP_PORT") or "587").strip())
    sender = str(os.getenv("SENDER_EMAIL") or "").strip()
    password = str(os.getenv("SENDER_PASSWORD") or "").strip()
    recipient = str(os.getenv("RECIPIENT_EMAIL") or "").strip()
    if not all((sender, password, recipient)):
        raise RuntimeError("SMTP credentials are not fully configured.")
    return server, port, sender, password, recipient


def _build_subject(*, strategy_tag: str, report_date: str) -> str:
    return f"Trading Bot Daily Report ({strategy_tag}) - {report_date} [ERROR]"


def _build_body(*, strategy_tag: str, report_date: str, source: str, message: str, model_used: str) -> str:
    now = _now_sgt()
    issue_time = now.strftime("%H:%M:%S")
    body_lines = [
        "Daily Trading Bot Report",
        "========================",
        f"Date: {report_date}",
        "",
        "PORTFOLIO SUMMARY",
        "-----------------",
        "Stocks Scanned Today: 0",
        "Open Positions: 0",
        "Positions Closed Today: 0",
        "New Positions Opened Today: 0",
        "Current Capital Estimate: $100,000.00",
        "Invested Notional: $0.00",
        "Available Cash: $100,000.00",
        "",
        "DAILY PERFORMANCE",
        "-----------------",
        "Realized P&L (Today): 0.00% ($0.00)",
        "",
        "ACCOUNT TOTALS",
        "--------------",
        "Total Realized P&L (Lifetime): 0.00% ($0.00)",
        "Unrealized P&L (Open Positions): 0.00% ($0.00)",
        "----------------------------------------",
        "TOTAL ACCOUNT RETURN: 0.00% (Lifetime Realized + Unrealized)",
        "",
        "PIPELINE SUMMARY",
        "----------------------------------------",
        "Tickers Total: 0",
        "Tickers Processed: 0",
        "Tickers Failed: 0",
        "News/LLM Sentiment: ON",
        "Run Health: ERROR",
        "Run Errors: 1",
        "Run Warnings: 0",
    ]
    if strategy_tag == "AI":
        body_lines.extend(
            [
                f"AI Trading LLM: ERROR (model={model_used or 'quant-trained-trading-model'})",
                f"AI Model Used: {model_used or 'quant-trained-trading-model'}",
                "AI Entries: BLOCKED (LLM unavailable/error)",
                f"AI LLM Error: {message}",
            ]
        )
    body_lines.extend(
        [
            "",
            "ERROR SUMMARY",
            "----------------------------------------",
            f"[{issue_time}] ERROR: {source} - {message}",
            "",
            "EXECUTION STEPS",
            "--------------------",
            f"[{issue_time}] {source}: Failed ({message})",
            "",
            "META-LEARNER INSIGHTS",
            "----------------------------------------",
            f"{strategy_tag} trading engine status: ERROR - {message}",
            "",
            "POSITIONS ENTERED TODAY",
            "----------------------------------------",
            "No new positions opened.",
            "",
            "POSITIONS CLOSED TODAY (Take Profit Hit)",
            "----------------------------------------",
            "No positions closed today.",
            "",
            "OPEN POSITIONS (Unrealized)",
            "----------------------------------------",
            "No open positions.",
            "",
            "---",
            "This is an automated message from your Trading Bot.",
        ]
    )
    return "\n".join(body_lines).strip() + "\n"


def _send_email(*, subject: str, body: str) -> None:
    server, port, sender, password, recipient = _smtp_settings()
    body_html = (
        "<html><body>"
        "<div style=\"font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', "
        "'Courier New', monospace; white-space: pre-wrap; line-height: 1.35;\">"
        + _html.escape(body)
        + "</div></body></html>"
    )
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject
    msg["Date"] = formatdate(localtime=False)
    msg["Message-ID"] = make_msgid()
    msg.set_content(body, charset="utf-8", cte="base64")
    msg.add_alternative(body_html, subtype="html", charset="utf-8", cte="base64")

    with smtplib.SMTP(server, port) as smtp:
        smtp.starttls()
        smtp.login(sender, password)
        smtp.send_message(msg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy-tag", required=True, choices=("Core", "AI"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--date", default="")
    parser.add_argument("--model-used", default="")
    args = parser.parse_args()

    report_date = str(args.date or _now_sgt().date().isoformat()).strip()
    strategy_tag = str(args.strategy_tag).strip()
    source = str(args.source).strip()
    message = str(args.message).strip()
    model_used = str(args.model_used).strip()

    subject = _build_subject(strategy_tag=strategy_tag, report_date=report_date)
    body = _build_body(
        strategy_tag=strategy_tag,
        report_date=report_date,
        source=source,
        message=message,
        model_used=model_used,
    )
    _send_email(subject=subject, body=body)


if __name__ == "__main__":
    main()
