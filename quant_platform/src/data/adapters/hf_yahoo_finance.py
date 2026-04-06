from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

try:
    import duckdb
except ImportError:  # pragma: no cover - optional dependency until installed in runtime
    duckdb = None


DEFAULT_BASE_URL = "https://huggingface.co/datasets/defeatbeta/yahoo-finance-data/resolve/main/data"
DATASET_FILES = {
    "stock_prices": "stock_prices.parquet",
    "stock_statement": "stock_statement.parquet",
    "stock_tailing_eps": "stock_tailing_eps.parquet",
    "stock_shares_outstanding": "stock_shares_outstanding.parquet",
    "stock_news": "stock_news.parquet",
    "stock_earning_call_transcripts": "stock_earning_call_transcripts.parquet",
}
POSITIVE_WORDS = {"beat", "growth", "upgrade", "strong", "record", "improve", "guidance", "raised"}
NEGATIVE_WORDS = {"miss", "downgrade", "weak", "lawsuit", "drop", "uncertain", "headwind", "lowered"}
GUIDANCE_WORDS = {"guide", "guidance", "outlook", "forecast", "expect", "target"}
UNCERTAINTY_WORDS = {"uncertain", "uncertainty", "volatile", "cautious", "risk", "challenging", "headwind"}
STATEMENT_LINE_MAP = {
    "revenue": {"total revenue", "totalrevenue", "revenue"},
    "gross_profit": {"gross profit", "grossprofit"},
    "operating_income": {"operating income", "operatingincome"},
    "net_income": {"net income", "netincome"},
    "ebitda": {"ebitda"},
    "total_assets": {"total assets", "totalassets"},
    "total_liabilities": {"total liab", "totalliab", "total liabilities", "totalliabilitiesnetminorityinterest"},
    "total_equity": {"total stockholder equity", "totalstockholderequity", "stockholdersequity"},
    "current_assets": {"total current assets", "totalcurrentassets"},
    "current_liabilities": {"total current liabilities", "totalcurrentliabilities"},
    "cash": {"cash", "cashandcashequivalents"},
    "long_term_debt": {"long term debt", "longtermdebt"},
    "operating_cashflow": {"operating cash flow", "operatingcashflow", "cash from operations"},
    "capex": {"capital expenditure", "capital expenditures", "capitalexpenditures"},
    "free_cashflow_stmt": {"free cash flow", "freecashflow"},
}


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _score_text(text: str) -> float:
    lowered = text.lower()
    pos = sum(word in lowered for word in POSITIVE_WORDS)
    neg = sum(word in lowered for word in NEGATIVE_WORDS)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def _count_terms(text: str, terms: Iterable[str]) -> int:
    lowered = text.lower()
    return sum(term in lowered for term in terms)


def _literal_list(values: Iterable[str]) -> str:
    return ", ".join(_literal(value) for value in values)


def _split_symbols(value: object) -> set[str]:
    if value is None or value is pd.NA:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip().upper().replace(".", "-") for item in value if str(item).strip()}
    text = str(value).strip()
    if not text:
        return set()
    return {part for part in re.split(r"[^A-Z0-9-]+", text.upper().replace(".", "-")) if part}


class HfYahooFinanceAdapter:
    def __init__(self, enabled: bool = True, cache_dir: str = "data/raw/hf_yahoo_finance") -> None:
        self.enabled = enabled and duckdb is not None
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir = self.cache_dir / "_dataset_mirror"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = os.getenv("HF_YAHOO_FINANCE_BASE_URL", DEFAULT_BASE_URL).rstrip("/")

    def dataset_source(self, dataset_name: str) -> str:
        local_path = self._ensure_dataset_local(dataset_name)
        if local_path is not None:
            return str(local_path)
        return self._dataset_url(dataset_name)

    def get_stock_prices(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        return self.get_stock_prices_batch([ticker], start, end).get(ticker, pd.DataFrame())

    def get_stock_prices_batch(self, tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
        tickers = [ticker for ticker in tickers if ticker]
        if not tickers:
            return {}
        sql = f"""
            SELECT symbol, report_date, open, high, low, close, volume
            FROM read_parquet({_literal(self.dataset_source("stock_prices"))})
            WHERE symbol IN ({_literal_list(tickers)})
              AND report_date BETWEEN {_literal(start)} AND {_literal(end)}
            ORDER BY symbol, report_date
        """
        frame = self._query("stock_prices", sql)
        out = {ticker: pd.DataFrame() for ticker in tickers}
        if frame.empty:
            return out
        frame["date"] = pd.to_datetime(frame.pop("report_date"), errors="coerce")
        frame = frame.dropna(subset=["date"])
        for ticker, group in frame.groupby("symbol"):
            group = group.drop(columns=["symbol"]).set_index("date").sort_index()
            group["adj_close"] = group["close"]
            out[ticker] = group[["open", "high", "low", "close", "adj_close", "volume"]]
        return out

    def get_news_features(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        return self.get_news_features_batch([ticker], start, end).get(ticker, pd.DataFrame())

    def get_news_features_batch(self, tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
        tickers = [ticker for ticker in tickers if ticker]
        if not tickers:
            return {}
        escaped = [ticker.replace("'", "''") for ticker in tickers]
        conditions = " OR ".join(
            f"related_symbols ILIKE '%{ticker}%'" for ticker in escaped
        )
        sql = f"""
            SELECT report_date, title, related_symbols
            FROM read_parquet({_literal(self.dataset_source("stock_news"))})
            WHERE report_date BETWEEN {_literal(start)} AND {_literal(end)}
              AND ({conditions})
            ORDER BY report_date
        """
        frame = self._query("stock_news", sql)
        out = {ticker: pd.DataFrame() for ticker in tickers}
        if frame.empty:
            return out

        frame["date"] = pd.to_datetime(frame.pop("report_date"), errors="coerce")
        frame = frame.dropna(subset=["date"])
        if frame.empty:
            return out

        target = set(tickers)
        rows: list[dict[str, object]] = []
        for _, row in frame.iterrows():
            related = _split_symbols(row.get("related_symbols"))
            matches = related & target
            if not matches:
                continue
            sentiment = _score_text(str(row.get("title") or ""))
            date = pd.to_datetime(row["date"]).normalize()
            for ticker in matches:
                rows.append({"ticker": ticker, "date": date, "sentiment": sentiment})

        if not rows:
            return out
        rows_df = pd.DataFrame(rows)
        grouped = rows_df.groupby(["ticker", "date"]).agg(
            news_count=("sentiment", "size"),
            news_sentiment=("sentiment", "mean"),
        )
        for ticker in tickers:
            ticker_frame = grouped.loc[ticker] if ticker in grouped.index.get_level_values(0) else pd.DataFrame()
            if isinstance(ticker_frame, pd.Series):
                ticker_frame = ticker_frame.to_frame().T
            if isinstance(ticker_frame, pd.DataFrame) and not ticker_frame.empty:
                ticker_frame.index.name = "date"
                out[ticker] = ticker_frame.sort_index()
        return out

    def get_transcript_features(self, ticker: str) -> pd.DataFrame:
        return self.get_transcript_features_batch([ticker]).get(ticker, pd.DataFrame())

    def get_transcript_features_batch(self, tickers: list[str]) -> dict[str, pd.DataFrame]:
        tickers = [ticker for ticker in tickers if ticker]
        if not tickers:
            return {}
        sql = f"""
            SELECT symbol, report_date, transcripts
            FROM read_parquet({_literal(self.dataset_source("stock_earning_call_transcripts"))})
            WHERE symbol IN ({_literal_list(tickers)})
            ORDER BY symbol, report_date
        """
        frame = self._query("stock_earning_call_transcripts", sql)
        out = {ticker: pd.DataFrame() for ticker in tickers}
        if frame.empty:
            return out
        for ticker, group in frame.groupby("symbol"):
            out[ticker] = self._transcript_frame(group.drop(columns=["symbol"]))
        return out

    def get_fundamentals(self, ticker: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
        return self.get_fundamentals_batch([ticker], dates).get(ticker, pd.DataFrame(index=dates))

    def get_fundamentals_batch(
        self, tickers: list[str], dates: pd.DatetimeIndex
    ) -> dict[str, pd.DataFrame]:
        tickers = [ticker for ticker in tickers if ticker]
        out = {ticker: pd.DataFrame(index=dates) for ticker in tickers}
        if not tickers:
            return out

        statement = self._query(
            "stock_statement",
            f"""
                SELECT symbol, report_date, item_name, item_value, finance_type, period_type
                FROM read_parquet({_literal(self.dataset_source("stock_statement"))})
                WHERE symbol IN ({_literal_list(tickers)})
                ORDER BY symbol, report_date
            """,
        )
        trailing_eps = self._query(
            "stock_tailing_eps",
            f"""
                SELECT symbol, report_date, tailing_eps
                FROM read_parquet({_literal(self.dataset_source("stock_tailing_eps"))})
                WHERE symbol IN ({_literal_list(tickers)})
                ORDER BY symbol, report_date
            """,
        )
        shares = self._query(
            "stock_shares_outstanding",
            f"""
                SELECT symbol, report_date, shares_outstanding
                FROM read_parquet({_literal(self.dataset_source("stock_shares_outstanding"))})
                WHERE symbol IN ({_literal_list(tickers)})
                ORDER BY symbol, report_date
            """,
        )

        for ticker in tickers:
            statement_frame = statement[statement["symbol"] == ticker].drop(columns=["symbol"]) if not statement.empty else pd.DataFrame()
            eps_frame = trailing_eps[trailing_eps["symbol"] == ticker].drop(columns=["symbol"]) if not trailing_eps.empty else pd.DataFrame()
            shares_frame = shares[shares["symbol"] == ticker].drop(columns=["symbol"]) if not shares.empty else pd.DataFrame()
            out[ticker] = self._fundamental_frame(statement_frame, eps_frame, shares_frame, dates)
        return out

    def _dataset_url(self, dataset_name: str) -> str:
        return f"{self.base_url}/{DATASET_FILES[dataset_name]}"

    def _dataset_path(self, dataset_name: str) -> Path:
        return self.dataset_dir / DATASET_FILES[dataset_name]

    def _ensure_dataset_local(self, dataset_name: str) -> Path | None:
        if not self.enabled:
            return None
        local_path = self._dataset_path(dataset_name)
        if local_path.exists() and local_path.stat().st_size > 0:
            return local_path
        url = self._dataset_url(dataset_name)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")
        try:
            with requests.get(url, stream=True, timeout=(30, 600)) as response:
                response.raise_for_status()
                with tmp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            tmp_path.replace(local_path)
            return local_path
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            return None

    def _cache_path(self, dataset_name: str, sql: str) -> Path:
        digest = hashlib.sha256(sql.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / dataset_name / f"{digest}.parquet"

    def _query(self, dataset_name: str, sql: str) -> pd.DataFrame:
        if not self.enabled:
            return pd.DataFrame()
        cache_path = self._cache_path(dataset_name, sql)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            return pd.read_parquet(cache_path)

        con = duckdb.connect()
        try:
            for statement in ("INSTALL httpfs", "LOAD httpfs"):
                try:
                    con.execute(statement)
                except Exception:
                    pass
            frame = con.execute(sql).fetch_df()
        except Exception:
            return pd.DataFrame()
        finally:
            con.close()

        if not frame.empty:
            frame.to_parquet(cache_path, index=False)
        return frame

    def _transcript_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame()
        rows: list[dict[str, object]] = []
        for _, row in frame.iterrows():
            transcripts = row.get("transcripts")
            if transcripts is None or transcripts is pd.NA:
                transcripts = []
            elif not isinstance(transcripts, list):
                transcripts = list(transcripts)
            mgmt_chunks: list[str] = []
            analyst_chunks: list[str] = []
            guidance_hits = 0
            uncertainty_hits = 0
            for item in transcripts:
                if not isinstance(item, dict):
                    continue
                content = str(item.get("content") or "")
                speaker = str(item.get("speaker") or "").lower()
                guidance_hits += _count_terms(content, GUIDANCE_WORDS)
                uncertainty_hits += _count_terms(content, UNCERTAINTY_WORDS)
                if "analyst" in speaker:
                    analyst_chunks.append(content)
                else:
                    mgmt_chunks.append(content)
            total_chunks = max(len(transcripts), 1)
            rows.append(
                {
                    "date": row.get("report_date"),
                    "mgmt_sentiment": _score_text(" ".join(mgmt_chunks)),
                    "guidance_keywords": guidance_hits,
                    "uncertainty": uncertainty_hits / total_chunks,
                    "analyst_sentiment": _score_text(" ".join(analyst_chunks)),
                    "surprise_delta": pd.NA,
                }
            )
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date"]).set_index("date").sort_index()
        return out

    def _fundamental_frame(
        self,
        statement: pd.DataFrame,
        trailing_eps: pd.DataFrame,
        shares: pd.DataFrame,
        dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        if not statement.empty:
            statement = statement.copy()
            statement["report_date"] = pd.to_datetime(statement["report_date"], errors="coerce")
            statement["item_name_norm"] = (
                statement["item_name"]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.replace(r"[^a-z0-9]+", " ", regex=True)
                .str.strip()
            )
            pivot_rows = []
            for output_name, names in STATEMENT_LINE_MAP.items():
                subset = statement[statement["item_name_norm"].isin(names)].copy()
                if subset.empty:
                    continue
                deduped = subset.sort_values("report_date").drop_duplicates(subset=["report_date"], keep="last")
                pivot_rows.append(deduped.set_index("report_date")["item_value"].rename(output_name))
            if pivot_rows:
                frames.append(pd.concat(pivot_rows, axis=1))

        if not trailing_eps.empty:
            trailing_eps = trailing_eps.copy()
            trailing_eps["report_date"] = pd.to_datetime(trailing_eps["report_date"], errors="coerce")
            eps_frame = trailing_eps.dropna(subset=["report_date"]).drop_duplicates("report_date", keep="last")
            frames.append(eps_frame.set_index("report_date")[["tailing_eps"]].rename(columns={"tailing_eps": "eps_ttm"}))

        if not shares.empty:
            shares = shares.copy()
            shares["report_date"] = pd.to_datetime(shares["report_date"], errors="coerce")
            shares_frame = shares.dropna(subset=["report_date"]).drop_duplicates("report_date", keep="last")
            frames.append(
                shares_frame.set_index("report_date")[["shares_outstanding"]].rename(
                    columns={"shares_outstanding": "shares_out"}
                )
            )

        if not frames:
            return pd.DataFrame(index=dates)

        frame = pd.concat(frames, axis=1).sort_index()
        expanded_index = frame.index.union(dates)
        return frame.reindex(expanded_index).sort_index().ffill().reindex(dates)
