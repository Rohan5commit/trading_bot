"""
Email Notification System

Sends daily trading reports via email.
"""
import smtplib
import os
import html as _html
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load .env from repo root reliably (scheduler may run with a different cwd).
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

def _format_table(headers, rows):
    if not rows:
        return ""
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(str(value)))
    sep = "-+-".join("-" * width for width in widths)
    header_line = " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    lines = [header_line, sep]
    for row in rows:
        lines.append(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(lines)


class EmailNotifier:
    def __init__(self):
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', 587))
        self.sender_email = os.getenv('SENDER_EMAIL')
        self.sender_password = os.getenv('SENDER_PASSWORD')  # App password for Gmail
        self.recipient_email = os.getenv('RECIPIENT_EMAIL')
        
    def send_daily_report(
        self,
        report_data,
        unrealized_df=None,
        closed_positions=None,
        new_positions=None,
        meta_insights=None,
        signal_rankings=None,
        pipeline_stats=None,
        backtest_signals=None,
        strategies=None,
        subject_tag=None
    ):
        """Send daily trading report via email"""
        if not all([self.sender_email, self.sender_password, self.recipient_email]):
            logger.warning("Email credentials not configured. Skipping email notification.")
            logger.info("To enable email, set SENDER_EMAIL, SENDER_PASSWORD, and RECIPIENT_EMAIL in .env")
            return False
        
        # Build email content
        subject = f"Trading Bot Daily Report - {report_data['date']}"
        if subject_tag:
            subject = f"Trading Bot Daily Report ({subject_tag}) - {report_data['date']}"
        
        # Calculate combined P&L
        realized = report_data.get('total_realized_pnl', 0)
        realized_today = report_data.get('realized_pnl_today', 0)
        unrealized = report_data.get('total_unrealized_pnl', 0)
        realized_dollars = report_data.get('total_realized_pnl_dollars')
        realized_today_dollars = report_data.get('realized_pnl_today_dollars')
        unrealized_dollars = report_data.get('total_unrealized_pnl_dollars')
        initial_capital = report_data.get('initial_capital')
        total_account_return = report_data.get('total_account_return')
        if total_account_return is None:
            if initial_capital and (realized_dollars is not None or unrealized_dollars is not None):
                total_account_return = ((realized_dollars or 0.0) + (unrealized_dollars or 0.0)) / initial_capital
            else:
                total_account_return = realized + unrealized
        combined = total_account_return
        current_capital = report_data.get('current_capital_estimate', 100000)
        invested_notional = report_data.get("invested_notional")
        available_cash = report_data.get("available_cash")
        invested_notional_str = f"${float(invested_notional):,.2f}" if isinstance(invested_notional, (int, float)) else "N/A"
        available_cash_str = f"${float(available_cash):,.2f}" if isinstance(available_cash, (int, float)) else "N/A"
        stocks_scanned_today = None
        if isinstance(pipeline_stats, dict):
            # Prefer actual processed count; fall back to total universe size if needed.
            stocks_scanned_today = pipeline_stats.get("tickers_processed")
            if stocks_scanned_today is None:
                stocks_scanned_today = pipeline_stats.get("tickers_total")
        stocks_scanned_str = str(stocks_scanned_today) if stocks_scanned_today is not None else "N/A"

        # Build body without leading indentation (some email clients render leading spaces poorly).
        body_lines = [
            "Daily Trading Bot Report",
            "========================",
            f"Date: {report_data['date']}",
            "",
        ]

        if strategies:
            body_lines.append("This report includes multiple strategy accounts. See STRATEGY DETAILS below.")
            body_lines.append(f"Stocks Scanned Today: {stocks_scanned_str}")
            body_lines.append("")
        else:
            body_lines.extend([
                "PORTFOLIO SUMMARY",
                "-----------------",
                f"Stocks Scanned Today: {stocks_scanned_str}",
                f"Open Positions: {report_data.get('open_positions', 0)}",
                f"Positions Closed Today: {report_data.get('positions_closed_at_tp', 0)}",
                f"New Positions Opened Today: {report_data.get('new_positions_opened', 0)}",
                f"Current Capital Estimate: ${current_capital:,.2f}",
                f"Invested Notional: {invested_notional_str}",
                f"Available Cash: {available_cash_str}",
                "",
                "DAILY PERFORMANCE",
                "-----------------",
                "Realized P&L (Today): "
                f"{realized_today:.2%}"
                f"{f' (${realized_today_dollars:,.2f})' if realized_today_dollars is not None else ''}",
                "",
                "ACCOUNT TOTALS",
                "--------------",
                "Total Realized P&L (Lifetime): "
                f"{realized:.2%}"
                f"{f' (${realized_dollars:,.2f})' if realized_dollars is not None else ''}",
                "Unrealized P&L (Open Positions): "
                f"{unrealized:.2%}"
                f"{f' (${unrealized_dollars:,.2f})' if unrealized_dollars is not None else ''}",
                "----------------------------------------",
                f"TOTAL ACCOUNT RETURN: {combined:.2%} (Lifetime Realized + Unrealized)",
            ])
            body_lines.append("")

        if pipeline_stats:
            body_lines.extend([
                "PIPELINE SUMMARY",
                "-" * 40,
            ])
            total = pipeline_stats.get('tickers_total')
            processed = pipeline_stats.get('tickers_processed')
            failed = pipeline_stats.get('tickers_failed')
            news_enabled = pipeline_stats.get('news_enabled')
            models_trained = pipeline_stats.get('models_trained')
            llm_status = pipeline_stats.get('llm_status') if isinstance(pipeline_stats, dict) else None
            ai_trading_llm_status = pipeline_stats.get('ai_trading_llm_status') if isinstance(pipeline_stats, dict) else None
            if total is not None:
                body_lines.append(f"Tickers Total: {total}")
            if processed is not None:
                body_lines.append(f"Tickers Processed: {processed}")
            if failed is not None:
                body_lines.append(f"Tickers Failed: {failed}")
            if models_trained is not None:
                body_lines.append(f"Models Trained: {models_trained}")
            if news_enabled is not None:
                body_lines.append(f"News/LLM Sentiment: {'ON' if news_enabled else 'OFF'}")
            if llm_status is not None:
                errors = llm_status.get('errors', 0)
                attempts = llm_status.get('attempts', 0)
                last_error = llm_status.get('last_error')
                if errors:
                    body_lines.append(f"LLM Status: ERROR (attempts={attempts}, errors={errors})")
                    if last_error:
                        body_lines.append(f"LLM Error: {last_error}")
                else:
                    body_lines.append(f"LLM Status: OK (attempts={attempts})")
            # AI-trading diagnostics must appear only in the AI strategy email.
            include_ai_pipeline = str(subject_tag or "").strip().upper() == "AI"
            if include_ai_pipeline and ai_trading_llm_status is not None:
                ok = bool(ai_trading_llm_status.get("ok"))
                err = ai_trading_llm_status.get("error")
                model = ai_trading_llm_status.get("model")
                model_used = ai_trading_llm_status.get("model_used")
                seen = ai_trading_llm_status.get("candidates_seen") or ai_trading_llm_status.get("candidates_built")
                blocked = ai_trading_llm_status.get("blocked_by_core")
                disallow = ai_trading_llm_status.get("disallow_core_overlap")
                skipped_reason = ai_trading_llm_status.get("skipped_reason")
                suffix = f" (model={model})" if model else ""
                body_lines.append(f"AI Trading LLM: {'OK' if ok else 'ERROR'}{suffix}")
                if model_used:
                    body_lines.append(f"AI Model Used: {model_used}")
                if skipped_reason == "no_capacity":
                    cap = ai_trading_llm_status.get("available_capital")
                    slots = ai_trading_llm_status.get("available_slots")
                    cap_txt = f"${float(cap):,.2f}" if isinstance(cap, (int, float)) else "N/A"
                    slots_txt = str(slots) if slots is not None else "N/A"
                    body_lines.append(f"AI LLM Call: SKIPPED (no available capacity; cash={cap_txt}, slots={slots_txt})")
                elif skipped_reason == "all_neutral":
                    body_lines.append("AI LLM Call: COMPLETED (all model signals neutral; no entries)")
                elif skipped_reason == "no_tradeable_signals":
                    body_lines.append("AI LLM Call: COMPLETED (signals filtered as non-tradeable)")
                if seen is not None:
                    body_lines.append(f"AI Candidates: {seen}")
                if disallow is not None:
                    body_lines.append(f"AI Core-Overlap Block: {'ON' if disallow else 'OFF'}")
                if blocked is not None:
                    body_lines.append(f"AI Core-Overlap Blocked Symbols: {blocked}")
                if ai_trading_llm_status.get("entries_blocked_due_to_llm_error"):
                    body_lines.append("AI Entries: BLOCKED (LLM unavailable/error)")
                if (not ok) and err:
                    body_lines.append(f"AI LLM Error: {err}")
            
            steps = pipeline_stats.get('steps')
            if steps:
                body_lines.append("")
                body_lines.append("EXECUTION STEPS")
                body_lines.append("-" * 20)
                for step in steps:
                    time_str = step.get('time', '')
                    name = step.get('step', '')
                    status = step.get('status', '')
                    details = step.get('details', '')
                    line = f"[{time_str}] {name}: {status}"
                    if details:
                        line += f" ({details})"
                    body_lines.append(line)

            body_lines.append("")
        
        if strategies:
            body_lines.extend([
                "STRATEGY DETAILS",
                "-" * 40,
            ])
            for strat in strategies:
                name = strat.get("name") or "Strategy"
                rep = strat.get("report") or {}
                udf = strat.get("unrealized")
                closed = strat.get("closed") or []
                newp = strat.get("new") or []
                insight = strat.get("meta_insights")

                realized_today = rep.get("realized_pnl_today", 0.0) or 0.0
                realized_today_dollars = rep.get("realized_pnl_today_dollars")
                realized_total = rep.get("total_realized_pnl", 0.0) or 0.0
                realized_total_dollars = rep.get("total_realized_pnl_dollars")
                unreal_total = rep.get("total_unrealized_pnl", 0.0) or 0.0
                unreal_total_dollars = rep.get("total_unrealized_pnl_dollars")
                total_account_return = rep.get("total_account_return")
                if total_account_return is None:
                    total_account_return = realized_total + unreal_total

                body_lines.append("")
                body_lines.append(str(name))
                body_lines.append("-" * 40)
                body_lines.append(f"Open Positions: {rep.get('open_positions', 0)}")
                body_lines.append(f"Positions Closed Today: {rep.get('positions_closed_at_tp', 0)}")
                body_lines.append(f"New Positions Opened Today: {rep.get('new_positions_opened', 0)}")
                cap = rep.get("current_capital_estimate")
                if cap is not None:
                    body_lines.append(f"Current Capital Estimate: ${cap:,.2f}")
                inv = rep.get("invested_notional")
                if inv is not None:
                    body_lines.append(f"Invested Notional: ${float(inv):,.2f}")
                av = rep.get("available_cash")
                if av is not None:
                    body_lines.append(f"Available Cash: ${float(av):,.2f}")
                body_lines.append("")
                body_lines.append(
                    "Realized P&L (Today): "
                    f"{realized_today:.2%}"
                    f"{f' (${realized_today_dollars:,.2f})' if realized_today_dollars is not None else ''}"
                )
                body_lines.append(
                    "Total Realized P&L: "
                    f"{realized_total:.2%}"
                    f"{f' (${realized_total_dollars:,.2f})' if realized_total_dollars is not None else ''}"
                )
                body_lines.append(
                    "Unrealized P&L: "
                    f"{unreal_total:.2%}"
                    f"{f' (${unreal_total_dollars:,.2f})' if unreal_total_dollars is not None else ''}"
                )
                body_lines.append(f"TOTAL ACCOUNT RETURN: {total_account_return:.2%}")

                if insight:
                    body_lines.append("")
                    body_lines.append("INSIGHTS")
                    body_lines.append("-" * 20)
                    body_lines.append(str(insight).strip())

                body_lines.append("")
                body_lines.append("POSITIONS ENTERED TODAY")
                body_lines.append("-" * 20)
                if newp:
                    rows = []
                    for pos in newp:
                        entry_price = pos.get("entry_price")
                        target_price = pos.get("target_price")
                        qty = pos.get("quantity", 0)
                        allocation_pct = pos.get("allocation_pct")
                        allocation_dollars = pos.get("allocation_dollars")
                        rows.append([
                            pos.get("symbol"),
                            pos.get("side", "LONG"),
                            f"{entry_price:.2f}" if isinstance(entry_price, (int, float)) else "N/A",
                            f"{target_price:.2f}" if isinstance(target_price, (int, float)) else "N/A",
                            f"{qty:.0f}",
                            f"{allocation_pct:.1f}%" if allocation_pct is not None else "N/A",
                            f"${allocation_dollars:,.2f}" if allocation_dollars is not None else "N/A",
                            pos.get("reason") or "",
                        ])
                    body_lines.append(_format_table(
                        ["Symbol", "Side", "Entry", "TP", "Qty", "Alloc %", "Alloc $", "Reason"],
                        rows
                    ))
                else:
                    body_lines.append("No new positions opened.")

                body_lines.append("")
                body_lines.append("POSITIONS CLOSED TODAY")
                body_lines.append("-" * 20)
                if closed:
                    rows = []
                    for pos in closed:
                        entry_price = pos.get("entry_price")
                        exit_price = pos.get("exit_price")
                        realized = float(pos.get("realized_pnl", 0.0) or 0.0)
                        rows.append([
                            pos.get("symbol"),
                            pos.get("side", "LONG"),
                            f"{entry_price:.2f}" if isinstance(entry_price, (int, float)) else "N/A",
                            f"{exit_price:.2f}" if isinstance(exit_price, (int, float)) else "N/A",
                            f"{realized:+.2%}",
                            pos.get("reason") or "",
                        ])
                    body_lines.append(_format_table(
                        ["Symbol", "Side", "Entry", "Exit", "P&L %", "Reason"],
                        rows
                    ))
                else:
                    body_lines.append("No positions closed today.")

                body_lines.append("")
                body_lines.append("OPEN POSITIONS (Unrealized)")
                body_lines.append("-" * 20)
                if udf is not None and hasattr(udf, "empty") and not udf.empty:
                    rows = []
                    for _, row in udf.iterrows():
                        pnl_pct = row.get("unrealized_pnl", 0.0) or 0.0
                        pnl_dollars = row.get("unrealized_pnl_dollars")
                        rows.append([
                            row.get("symbol"),
                            row.get("side", "LONG"),
                            row.get("entry_date"),
                            f"{float(row.get('entry_price', 0.0)):.2f}",
                            f"{float(row.get('current_price', 0.0)):.2f}",
                            f"{float(row.get('target_price', 0.0)):.2f}",
                            f"{float(pnl_pct):+.2%}",
                            f"${pnl_dollars:,.2f}" if pnl_dollars is not None else "N/A",
                            f"{float(row.get('distance_to_tp', 0.0)):.1%}" if row.get("distance_to_tp") is not None else "N/A",
                        ])
                    body_lines.append(_format_table(
                        ["Symbol", "Side", "Entry Date", "Entry", "Current", "TP", "P&L %", "P&L $", "Dist to TP"],
                        rows
                    ))
                else:
                    body_lines.append("No open positions.")
            body_lines.append("")

        if meta_insights and not strategies:
            body_lines.extend([
                "META-LEARNER INSIGHTS",
                "-" * 40,
                str(meta_insights).strip(),
                "",
            ])

        if (not strategies) and new_positions:
            body_lines.extend([
                "POSITIONS ENTERED TODAY",
                "-" * 40,
            ])
            rows = []
            for pos in new_positions:
                side = pos.get("side") or "LONG"
                entry_price = pos.get('entry_price')
                target_price = pos.get('target_price')
                qty = pos.get('quantity', 0)
                allocation_pct = pos.get('allocation_pct')
                allocation_dollars = pos.get('allocation_dollars')
                rows.append([
                    pos.get('symbol'),
                    side,
                    f"{entry_price:.2f}" if entry_price is not None else "N/A",
                    f"{target_price:.2f}" if target_price is not None else "N/A",
                    f"{qty:.0f}",
                    f"{allocation_pct:.1f}%" if allocation_pct is not None else "N/A",
                    f"${allocation_dollars:,.2f}" if allocation_dollars is not None else "N/A",
                    pos.get('reason') or ""
                ])
            table = _format_table(
                ["Symbol", "Side", "Entry", "TP", "Qty", "Alloc %", "Alloc $", "Reason"],
                rows
            )
            body_lines.append(table)
            body_lines.append("")
        elif not strategies:
            body_lines.extend([
                "POSITIONS ENTERED TODAY",
                "-" * 40,
                "No new positions opened.",
                "",
            ])
        
        if (not strategies) and closed_positions:
            body_lines.extend([
                "POSITIONS CLOSED TODAY (Take Profit Hit)",
                "-" * 40,
            ])
            rows = []
            for pos in closed_positions:
                side = pos.get("side") or "LONG"
                entry_price = pos.get('entry_price')
                exit_price = pos.get('exit_price')
                realized = pos.get('realized_pnl', 0.0)
                reason = pos.get('reason')
                rows.append([
                    pos.get('symbol'),
                    side,
                    f"{entry_price:.2f}" if entry_price is not None else "N/A",
                    f"{exit_price:.2f}" if exit_price is not None else "N/A",
                    f"{realized:+.2%}",
                    reason or ""
                ])
            table = _format_table(
                ["Symbol", "Side", "Entry", "Exit", "P&L %", "Reason"],
                rows
            )
            body_lines.append(table)
            body_lines.append("")
        elif not strategies:
            body_lines.extend([
                "POSITIONS CLOSED TODAY (Take Profit Hit)",
                "-" * 40,
                "No positions closed today.",
                "",
            ])
        
        if (not strategies) and unrealized_df is not None and not unrealized_df.empty:
            body_lines.extend([
                "OPEN POSITIONS (Unrealized)",
                "-" * 40,
            ])
            rows = []
            for _, row in unrealized_df.iterrows():
                pnl_pct = row.get('unrealized_pnl', 0.0) or 0.0
                pnl_str = f"{pnl_pct:+.2%}"
                pnl_dollars = row.get('unrealized_pnl_dollars')
                dist = row.get("distance_to_tp")
                if dist is None:
                    dist = row.get("dist_to_tp_pct")
                if dist is None:
                    dist = row.get("dist_to_tp")
                dist_str = f"{float(dist):.1%}" if dist is not None else "N/A"
                rows.append([
                    row.get('symbol'),
                    row.get('side', 'LONG'),
                    row.get('entry_date'),
                    f"{float(row.get('entry_price', 0.0)):.2f}",
                    f"{row.get('current_price', 0.0):.2f}",
                    f"{row.get('target_price', 0.0):.2f}",
                    pnl_str,
                    f"${pnl_dollars:,.2f}" if pnl_dollars is not None else "N/A",
                    dist_str,
                ])
            table = _format_table(
                ["Symbol", "Side", "Entry Date", "Entry", "Current", "TP", "P&L %", "P&L $", "Dist to TP"],
                rows
            )
            body_lines.append(table)
            body_lines.append("")
        elif not strategies:
            body_lines.extend([
                "OPEN POSITIONS (Unrealized)",
                "-" * 40,
                "No open positions.",
                "",
            ])

        if signal_rankings is not None:
            body_lines.extend([
                "SIGNAL RANKINGS (T-1)",
                "-" * 40,
            ])
            if hasattr(signal_rankings, "iterrows"):
                rows = signal_rankings.head(10)
                for _, row in rows.iterrows():
                    symbol = row.get('symbol')
                    pred = row.get('predicted_return', 0.0)
                    adj = row.get('adjusted_score', pred)
                    penalty = row.get('penalty', 1.0)
                    rank = row.get('rank', '')
                    body_lines.append(f"{rank}. {symbol}: pred {pred:.2%}, adj {adj:.2%}, penalty {penalty:.2f}")
            else:
                for row in signal_rankings[:10]:
                    body_lines.append(str(row))
            body_lines.append("")

        if backtest_signals is not None:
            body_lines.extend([
                "BACKTEST SIGNALS (Momentum/Mean Reversion)",
                "-" * 40,
            ])
            buy_signals = [s for s in backtest_signals if s.get("action") == "BUY"]
            buy_signals.sort(key=lambda s: float(s.get("confidence", 0)), reverse=True)
            if buy_signals:
                for signal in buy_signals[:10]:
                    entry_price = signal.get("entry_price")
                    entry_display = f"{entry_price:.2f}" if isinstance(entry_price, (int, float)) else "N/A"
                    tags = signal.get("rationale_tags")
                    tags_display = ",".join(tags) if isinstance(tags, list) else ""
                    confidence = float(signal.get("confidence", 0.0) or 0.0)
                    body_lines.append(
                        f"{signal.get('timestamp')} {signal.get('symbol')} BUY "
                        f"entry={entry_display} conf={confidence:.2f} tags={tags_display}\n"
                    )
            else:
                body_lines.append("No backtest signals recorded")
            body_lines.append("")

        body_lines.extend([
            "---",
            "This is an automated message from your Trading Bot.",
        ])

        body = "\n".join([str(x) for x in body_lines if x is not None]).strip() + "\n"

        # Gmail clients occasionally display a blank body for quoted-printable/plain parts and/or <pre>.
        # We force base64 for text/plain and use a simple HTML <div> (not <pre>) so rendering is reliable.
        body_html = (
            "<html><body>"
            "<div style=\"font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', "
            "'Courier New', monospace; white-space: pre-wrap; line-height: 1.35;\">"
            + _html.escape(body)
            + "</div></body></html>"
        )
        
        try:
            msg = EmailMessage()
            msg["From"] = self.sender_email
            msg["To"] = self.recipient_email
            msg["Subject"] = subject
            # Some Gmail clients behave oddly when these headers are missing.
            msg["Date"] = formatdate(localtime=False) # We set it manually in subject/body
            msg["Message-ID"] = make_msgid()

            # base64 avoids quoted-printable soft-wrap artifacts on long lines (tables).
            msg.set_content(body, charset="utf-8", cte="base64")
            # Use base64 for HTML to avoid any quoted-printable edge cases in some clients.
            msg.add_alternative(body_html, subtype="html", charset="utf-8", cte="base64")
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            logger.info(f"Email sent successfully to {self.recipient_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

if __name__ == "__main__":
    notifier = EmailNotifier()
    # Test with dummy data
    test_report = {
        'date': '2026-01-16',
        'open_positions': 9,
        'positions_closed_at_tp': 1,
        'new_positions_opened': 10,
        'total_realized_pnl': 0.03,
        'total_unrealized_pnl': -0.016
    }
    notifier.send_daily_report(test_report)
