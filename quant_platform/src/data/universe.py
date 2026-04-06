from __future__ import annotations

import os
import re
from typing import List

import pandas as pd
import requests

try:
    import duckdb
except ImportError:  # pragma: no cover
    duckdb = None

from .adapters.hf_yahoo_finance import HfYahooFinanceAdapter

NASDAQ_LISTED = "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED = "https://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"


def _sec_headers() -> dict:
    user_agent = os.getenv("SEC_USER_AGENT")
    if not user_agent:
        user_agent = "train-once-quant-platform/0.1 (Rohan; rohan.santhoshkumar1@gmail.com)"
    return {"User-Agent": user_agent}


def _sanitize(symbols: List[str]) -> List[str]:
    cleaned = []
    for sym in symbols:
        if not sym:
            continue
        sym = str(sym).strip().upper().replace(".", "-")
        if not re.match(r"^[A-Z0-9-]+$", sym):
            continue
        cleaned.append(sym)
    return sorted(set(cleaned))


def _read_pipe(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    lines = resp.text.strip().splitlines()
    data = [line.split("|") for line in lines[1:-1]]
    return pd.DataFrame(data, columns=lines[0].split("|"))


def get_sec_ticker_universe(max_tickers: int | None = None) -> List[str]:
    resp = requests.get(SEC_TICKERS, headers=_sec_headers(), timeout=30)
    resp.raise_for_status()
    data = resp.json()
    symbols = []
    for item in data.values():
        ticker = item.get("ticker")
        if ticker:
            symbols.append(str(ticker))
    symbols = _sanitize(symbols)
    if max_tickers:
        symbols = symbols[:max_tickers]
    return symbols


def get_nasdaq_universe(max_tickers: int | None = None) -> List[str]:
    nasdaq = _read_pipe(NASDAQ_LISTED)
    other = _read_pipe(OTHER_LISTED)

    nasdaq = nasdaq[(nasdaq["Test Issue"] == "N")]
    other = other[(other["Test Issue"] == "N")]

    symbols = pd.concat([nasdaq[["Symbol"]], other[["ACT Symbol"]].rename(columns={"ACT Symbol": "Symbol"})])
    symbols = _sanitize(symbols["Symbol"].dropna().unique().tolist())

    if max_tickers:
        symbols = symbols[:max_tickers]
    return symbols


def get_hf_liquid_universe(
    max_tickers: int | None = None,
    min_rows: int = 3000,
    recent_since: str = "2024-01-01",
    cache_dir: str | None = None,
) -> List[str]:
    if duckdb is None:
        return []
    adapter = HfYahooFinanceAdapter(enabled=True, cache_dir=cache_dir or "data/raw/hf_yahoo_finance")
    source = adapter.dataset_source("stock_prices")
    con = duckdb.connect()
    try:
        for statement in ("INSTALL httpfs", "LOAD httpfs"):
            try:
                con.execute(statement)
            except Exception:
                pass
        limit_sql = f"LIMIT {int(max_tickers)}" if max_tickers else ""
        query = f"""
            WITH base AS (
                SELECT
                    symbol,
                    CAST(report_date AS DATE) AS trade_date,
                    close,
                    volume
                FROM read_parquet({_sec_quote(source)})
                WHERE CAST(report_date AS DATE) BETWEEN DATE '2010-01-01' AND DATE '2024-12-31'
            ),
            symbol_stats AS (
                SELECT
                    symbol,
                    COUNT(*) AS row_count,
                    MAX(trade_date) AS max_date,
                    AVG(CASE WHEN trade_date >= DATE '2023-01-01' THEN close * volume END) AS avg_dollar_volume
                FROM base
                GROUP BY symbol
            )
            SELECT symbol
            FROM symbol_stats
            WHERE row_count >= {int(min_rows)}
              AND max_date >= DATE {_sec_quote(recent_since)}
            ORDER BY avg_dollar_volume DESC NULLS LAST, row_count DESC, symbol
            {limit_sql}
        """
        rows = con.execute(query).fetchall()
    except Exception as exc:
        print(f"hf_liquid universe query failed: {exc}")
        return []
    finally:
        con.close()
    return _sanitize([row[0] for row in rows])


def _sec_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def get_us_equity_universe(
    max_tickers: int | None = None,
    source: str = "nasdaq_all",
    hf_cache_dir: str | None = None,
) -> List[str]:
    if source == "nasdaq_all":
        return get_nasdaq_universe(max_tickers)
    if source == "sec_tickers":
        return get_sec_ticker_universe(max_tickers)
    if source == "hf_liquid":
        return get_hf_liquid_universe(max_tickers=max_tickers, cache_dir=hf_cache_dir)
    raise ValueError(f"Unsupported universe source: {source}")
