from __future__ import annotations

from pathlib import Path

import pandas as pd
from jinja2 import Template


REPORT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{{ title }}</title>
  <style>
    body { font-family: "Helvetica", sans-serif; margin: 24px; color: #222; }
    h1 { margin-bottom: 0; }
    .meta { margin-bottom: 16px; color: #555; }
    table { border-collapse: collapse; width: 100%; margin-top: 16px; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    th { background-color: #f5f5f5; }
    .note { margin-top: 20px; font-size: 0.95em; color: #444; }
  </style>
</head>
<body>
  <h1>{{ title }}</h1>
  <div class="meta">Generated: {{ generated }}</div>

  <h2>Metrics</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    {% for key, value in metrics.items() %}
      <tr><td>{{ key }}</td><td>{{ value }}</td></tr>
    {% endfor %}
  </table>

  <h2>Recent Trades</h2>
  <table>
    <tr><th>Timestamp</th><th>Symbol</th><th>Action</th><th>Shares</th><th>Price</th><th>PnL</th></tr>
    {% for trade in trades %}
      <tr>
        <td>{{ trade.timestamp }}</td>
        <td>{{ trade.symbol }}</td>
        <td>{{ trade.action }}</td>
        <td>{{ trade.shares }}</td>
        <td>{{ trade.price }}</td>
        <td>{{ trade.pnl }}</td>
      </tr>
    {% endfor %}
  </table>

  <div class="note">{{ notes }}</div>
</body>
</html>
"""


def write_report(title: str, metrics: dict, trades: pd.DataFrame, notes: str, output_path: Path) -> None:
    template = Template(REPORT_TEMPLATE)
    rendered = template.render(
        title=title,
        generated=pd.Timestamp.utcnow().isoformat(),
        metrics=metrics,
        trades=trades.tail(10).to_dict(orient="records"),
        notes=notes,
    )
    output_path.write_text(rendered, encoding="utf-8")
