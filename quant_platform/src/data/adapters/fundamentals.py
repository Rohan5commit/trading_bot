from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import yfinance as yf

from .hf_yahoo_finance import HfYahooFinanceAdapter


def _as_frame(info: Dict, dates: pd.DatetimeIndex) -> pd.DataFrame:
    data = {
        "pe": info.get("trailingPE"),
        "pb": info.get("priceToBook"),
        "ps": info.get("priceToSalesTrailing12Months"),
        "ev_ebitda": info.get("enterpriseToEbitda"),
        "ev_sales": info.get("enterpriseToRevenue"),
        "roe": info.get("returnOnEquity"),
        "roa": info.get("returnOnAssets"),
        "debt_equity": info.get("debtToEquity"),
        "current_ratio": info.get("currentRatio"),
        "quick_ratio": info.get("quickRatio"),
        "gross_margins": info.get("grossMargins"),
        "operating_margins": info.get("operatingMargins"),
        "profit_margins": info.get("profitMargins"),
        "dividend_yield": info.get("dividendYield"),
        "free_cashflow": info.get("freeCashflow"),
        "market_cap": info.get("marketCap"),
        "shares_out": info.get("sharesOutstanding"),
    }
    frame = pd.DataFrame([data] * len(dates), index=dates)
    if data.get("free_cashflow") and data.get("market_cap"):
        frame["fcf_yield"] = data["free_cashflow"] / data["market_cap"]
    return frame


def _extract_line(df: pd.DataFrame, names: list[str]) -> pd.Series | None:
    if df is None or df.empty:
        return None
    for name in names:
        if name in df.index:
            return df.loc[name]
    return None


def _statement_frame(df: pd.DataFrame, mapping: Dict[str, list[str]]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.columns = pd.to_datetime(df.columns)
    rows = {}
    for key, names in mapping.items():
        series = _extract_line(df, names)
        if series is not None:
            rows[key] = series
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def _align_and_ffill(frame: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(index=dates)
    expanded_index = frame.index.union(dates)
    return frame.reindex(expanded_index).sort_index().ffill().reindex(dates)


def _earnings_frame(ticker: yf.Ticker) -> pd.DataFrame:
    try:
        df = ticker.earnings_dates
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    cols = {"EPS Estimate": "eps_est", "Reported EPS": "eps_actual", "Surprise(%)": "eps_surprise_pct"}
    for col in list(cols.keys()):
        if col not in df.columns:
            cols.pop(col, None)
    df = df.rename(columns=cols)
    if "eps_est" in df.columns and "eps_actual" in df.columns:
        df["eps_surprise"] = df["eps_actual"] - df["eps_est"]
    if "eps_surprise_pct" in df.columns:
        df["eps_beat"] = (df["eps_surprise_pct"] > 0).astype(float)
    return df[[c for c in ["eps_est", "eps_actual", "eps_surprise", "eps_surprise_pct", "eps_beat"] if c in df.columns]]


class FundamentalsAdapter:
    def __init__(
        self,
        hf_enabled: bool = True,
        hf_cache_dir: str = "data/raw/hf_yahoo_finance",
        allow_fallbacks: bool = True,
    ) -> None:
        self.hf = HfYahooFinanceAdapter(enabled=hf_enabled, cache_dir=hf_cache_dir)
        self.allow_fallbacks = allow_fallbacks

    def get_fundamentals(self, ticker: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
        hf_frame = self.hf.get_fundamentals(ticker, dates)
        if not self.allow_fallbacks:
            return hf_frame

        info = {}
        tkr = yf.Ticker(ticker)
        try:
            info = tkr.info
        except Exception:
            info = {}

        frame = _as_frame(info, dates)

        # Quarterly statements
        fin = _statement_frame(
            getattr(tkr, "quarterly_financials", None),
            {
                "revenue": ["Total Revenue", "TotalRevenue"],
                "gross_profit": ["Gross Profit", "GrossProfit"],
                "operating_income": ["Operating Income", "OperatingIncome"],
                "net_income": ["Net Income", "NetIncome"],
                "ebitda": ["EBITDA", "Ebitda"],
            },
        )
        bs = _statement_frame(
            getattr(tkr, "quarterly_balance_sheet", None),
            {
                "total_assets": ["Total Assets", "TotalAssets"],
                "total_liabilities": ["Total Liab", "TotalLiab"],
                "total_equity": ["Total Stockholder Equity", "TotalStockholderEquity"],
                "current_assets": ["Total Current Assets", "TotalCurrentAssets"],
                "current_liabilities": ["Total Current Liabilities", "TotalCurrentLiabilities"],
                "cash": ["Cash", "CashAndCashEquivalents"],
                "long_term_debt": ["Long Term Debt", "LongTermDebt"],
            },
        )
        cf = _statement_frame(
            getattr(tkr, "quarterly_cashflow", None),
            {
                "operating_cashflow": ["Total Cash From Operating Activities", "OperatingCashFlow"],
                "capex": ["Capital Expenditures", "CapitalExpenditures"],
                "free_cashflow_stmt": ["Free Cash Flow", "FreeCashFlow"],
            },
        )

        stmt = pd.concat([fin, bs, cf], axis=1)
        if not stmt.empty:
            stmt = _align_and_ffill(stmt, dates)
            frame = frame.join(stmt, how="left")

        earnings = _earnings_frame(tkr)
        if not earnings.empty:
            earnings = _align_and_ffill(earnings, dates)
            frame = frame.join(earnings, how="left")

        # Derived ratios
        if "revenue" in frame.columns and "gross_profit" in frame.columns:
            frame["gross_margin_stmt"] = frame["gross_profit"] / frame["revenue"].replace(0, np.nan)
        if "revenue" in frame.columns and "operating_income" in frame.columns:
            frame["operating_margin_stmt"] = frame["operating_income"] / frame["revenue"].replace(0, np.nan)
        if "revenue" in frame.columns and "net_income" in frame.columns:
            frame["net_margin_stmt"] = frame["net_income"] / frame["revenue"].replace(0, np.nan)
        if "net_income" in frame.columns and "total_equity" in frame.columns:
            frame["roe_stmt"] = frame["net_income"] / frame["total_equity"].replace(0, np.nan)
        if "net_income" in frame.columns and "total_assets" in frame.columns:
            frame["roa_stmt"] = frame["net_income"] / frame["total_assets"].replace(0, np.nan)
        if "free_cashflow_stmt" in frame.columns and "market_cap" in frame.columns:
            frame["fcf_yield_stmt"] = frame["free_cashflow_stmt"] / frame["market_cap"].replace(0, np.nan)
        if "revenue" in frame.columns:
            frame["revenue_yoy"] = frame["revenue"].pct_change(4)
            frame["revenue_qoq"] = frame["revenue"].pct_change(1)

        if not hf_frame.empty:
            for column in hf_frame.columns:
                if column not in frame.columns:
                    frame[column] = hf_frame[column]
                else:
                    frame[column] = frame[column].where(frame[column].notna(), hf_frame[column])

            if "revenue" in frame.columns and "gross_profit" in frame.columns:
                frame["gross_margin_stmt"] = frame["gross_profit"] / frame["revenue"].replace(0, np.nan)
            if "revenue" in frame.columns and "operating_income" in frame.columns:
                frame["operating_margin_stmt"] = frame["operating_income"] / frame["revenue"].replace(0, np.nan)
            if "revenue" in frame.columns and "net_income" in frame.columns:
                frame["net_margin_stmt"] = frame["net_income"] / frame["revenue"].replace(0, np.nan)
            if "revenue" in frame.columns:
                frame["revenue_yoy"] = frame["revenue"].pct_change(4)
                frame["revenue_qoq"] = frame["revenue"].pct_change(1)
            if "net_income" in frame.columns and "total_equity" in frame.columns:
                frame["roe_stmt"] = frame["net_income"] / frame["total_equity"].replace(0, np.nan)
            if "net_income" in frame.columns and "total_assets" in frame.columns:
                frame["roa_stmt"] = frame["net_income"] / frame["total_assets"].replace(0, np.nan)
            if "free_cashflow_stmt" in frame.columns and "market_cap" in frame.columns:
                frame["fcf_yield_stmt"] = frame["free_cashflow_stmt"] / frame["market_cap"].replace(0, np.nan)

        return frame

    def get_fundamentals_batch(self, tickers: list[str], dates: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
        hf_frames = self.hf.get_fundamentals_batch(tickers, dates)
        if not self.allow_fallbacks:
            return hf_frames

        out: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            out[ticker] = self.get_fundamentals(ticker, dates)
        return out
