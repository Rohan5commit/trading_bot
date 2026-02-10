from __future__ import annotations

import json
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_signals(path: Path) -> List[Dict[str, Any]]:
    payload = _load_json(path)
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        try:
            keys = sorted(payload.keys(), key=lambda k: int(k))
            return [payload[k] for k in keys]
        except ValueError:
            return list(payload.values())
    return []


def build_email_report(config: dict, report_dir: Path) -> Tuple[str, str]:
    metrics_path = report_dir / "metrics.json"
    trades_path = report_dir / "trades.csv"
    signals_path = report_dir / "signals.json"
    data_status_path = report_dir / "data_status.json"

    metrics = _load_json(metrics_path) or {}
    trades = pd.read_csv(trades_path) if trades_path.exists() else pd.DataFrame()
    signals = _load_signals(signals_path)
    data_status = _load_json(data_status_path) or {}

    timestamp = pd.Timestamp.utcnow().isoformat()
    subject_prefix = config.get("email", {}).get("subject_prefix", "[QuantModel]")
    subject = f"{subject_prefix} Report {timestamp[:10]}"

    lines = [
        f"Report generated: {timestamp}",
        "",
    ]
    if data_status:
        lines.append(f"Data status: {data_status.get('status')} ({data_status.get('reason')})")
        lines.append("")
    strategy_enabled = config.get("strategy", {}).get("enabled", False)
    if not strategy_enabled:
        lines.append("Strategy disabled: backtest results skipped.")
        lines.append("")
        metrics = {}
        trades = pd.DataFrame()

    lines.extend([
        "Metrics:",
    ])
    if metrics:
        for key, value in metrics.items():
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- No metrics available")

    lines.append("")
    lines.append("Recent trades:")
    if not trades.empty:
        tail = trades.tail(5)
        for _, row in tail.iterrows():
            lines.append(
                f"- {row['timestamp']} {row['symbol']} {row['action']} shares={row['shares']} price={row['price']} pnl={row['pnl']}"
            )
    else:
        lines.append("- No trades recorded")

    lines.append("")
    lines.append("Top buy signals:")
    if signals:
        buy_signals = [s for s in signals if s.get("action") == "BUY"]
        buy_signals.sort(key=lambda s: float(s.get("confidence", 0)), reverse=True)
        for signal in buy_signals[:10]:
            entry_price = signal.get("entry_price")
            entry_display = f"{entry_price:.2f}" if isinstance(entry_price, (int, float)) else "N/A"
            lines.append(
                f"- {signal.get('timestamp')} {signal.get('symbol')} {signal.get('action')} entry={entry_display} size={signal.get('position_size')} conf={signal.get('confidence')} tags={signal.get('rationale_tags')}"
            )
    else:
        lines.append("- No signals recorded")

    lines.append("")
    lines.append("Note: This report is informational only; no trades were executed.")

    return subject, "\n".join(lines)


def write_email_report(subject: str, body: str, output_path: Path) -> None:
    output_path.write_text(f"{subject}\n\n{body}", encoding="utf-8")


def send_email_report(config: dict, subject: str, body: str, report_dir: Path | None = None) -> bool:
    if not (config.get("strategy", {}).get("enabled", False) or config.get("signals", {}).get("enabled", False)):
        return False
    email_cfg = config.get("email", {})
    if not email_cfg.get("enabled"):
        return False

    host = email_cfg.get("smtp_host")
    port = int(email_cfg.get("smtp_port", 587))
    timeout = int(email_cfg.get("timeout_seconds", 20))
    sender = email_cfg.get("from")
    recipients = email_cfg.get("to", [])

    if not host or not sender or not recipients:
        return False

    username_env = email_cfg.get("username_env", "SMTP_USERNAME")
    password_env = email_cfg.get("password_env", "SMTP_PASSWORD")
    username = os.getenv(username_env)
    password = os.getenv(password_env)
    if not username or not password:
        return False

    message = EmailMessage()
    message["From"] = sender
    message["To"] = ", ".join(recipients)
    message["Subject"] = subject
    # base64 avoids quoted-printable soft-wrap artifacts on long lines.
    message.set_content(body, charset="utf-8", cte="base64")

    # Attach artifacts if available (keeps body readable and provides a spreadsheet-like export).
    if report_dir is not None and isinstance(report_dir, Path) and report_dir.exists():
        attachments = [
            ("trades.csv", "text", "csv"),
            ("metrics.json", "application", "json"),
            ("daily_equity.csv", "text", "csv"),
        ]
        for fname, maintype, subtype in attachments:
            path = report_dir / fname
            if not path.exists():
                continue
            try:
                data = path.read_bytes()
                message.add_attachment(data, maintype=maintype, subtype=subtype, filename=fname)
            except Exception:
                continue

    with smtplib.SMTP(host, port, timeout=timeout) as server:
        server.starttls()
        server.login(username, password)
        server.send_message(message)
    return True
