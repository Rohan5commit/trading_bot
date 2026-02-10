from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf
import requests

from src.data.bars import align_to_market_calendar, filter_market_hours, resample_to_interval
from src.data.cache import CacheKey, load_cache, write_cache
from src.data.provenance import ProvenanceLogger, ProvenanceRecord, utc_now
from src.data.providers import BenchmarkProvider, CorporateActionsProvider, HistoricalBarsProvider, UniverseProvider, UniverseSnapshot
from src.data.quality import check_quality
from src.data.sample import generate_sample_bars


class YFinanceBarsProvider(HistoricalBarsProvider):
    def __init__(
        self,
        cache_dir: Path,
        provenance: ProvenanceLogger,
        timezone: str,
        open_time: str,
        close_time: str,
        source_interval: str,
        fallback_to_sample: bool,
    ):
        self.cache_dir = cache_dir
        self.provenance = provenance
        self.timezone = timezone
        self.open_time = open_time
        self.close_time = close_time
        self.source_interval = source_interval
        self.fallback_to_sample = fallback_to_sample

    def get_bars(self, symbols: Iterable[str], start: str, end: str, interval: str) -> dict[str, pd.DataFrame]:
        data: dict[str, pd.DataFrame] = {}
        sample_paths = _load_sample_paths(Path("logs/provenance.jsonl")) if not self.fallback_to_sample else set()
        for symbol in symbols:
            key = CacheKey(provider="yfinance_bars", symbol=symbol, start=start, end=end, interval=interval)
            cached = load_cache(self.cache_dir, key)
            if cached is not None:
                if self.fallback_to_sample:
                    data[symbol] = cached
                    continue
                cache_path = self.cache_dir / key.provider / key.filename()
                if _is_sample_cache(sample_paths, cache_path):
                    cached = None
                else:
                    data[symbol] = cached
                    continue

            try:
                ticker = yf.Ticker(symbol)
                raw = ticker.history(start=start, end=end, interval=self.source_interval, auto_adjust=False)
            except Exception:
                raw = pd.DataFrame()
            if raw.empty:
                if self.fallback_to_sample:
                    sample = generate_sample_bars(symbol, start, end, interval)
                    if sample.empty:
                        continue
                    output_path = write_cache(self.cache_dir, key, sample)
                    self.provenance.log(
                        ProvenanceRecord(
                            dataset="bars",
                            source="sample",
                            retrieved_at=utc_now(),
                            last_updated=None,
                            parameters={"symbol": symbol, "start": start, "end": end, "interval": interval},
                            output_path=str(output_path),
                        )
                    )
                    data[symbol] = sample
                continue
            raw = raw.rename(columns=str.lower)
            raw = raw.tz_localize("UTC") if raw.index.tz is None else raw

            # Daily source interval: yfinance returns one bar per session (date-only index).
            # The intraday filtering/resampling pipeline is meant for <=1h bars and can drop daily bars.
            if str(self.source_interval).lower().endswith("d"):
                resampled = raw.copy()
                # Normalize to UTC timestamps and required columns.
                if isinstance(resampled.index, pd.DatetimeIndex):
                    resampled.index = resampled.index.tz_convert("UTC") if resampled.index.tz is not None else resampled.index.tz_localize("UTC")
                cols = [c for c in ["open", "high", "low", "close", "volume"] if c in resampled.columns]
                resampled = resampled.loc[:, cols]
            else:
                raw = filter_market_hours(raw, self.timezone, self.open_time, self.close_time)
                raw = align_to_market_calendar(raw, self.timezone)
                offset = _offset_from_time(self.open_time)
                resampled = resample_to_interval(raw, interval, offset=offset)
                resampled = resampled.tz_convert("UTC")
                resampled = resampled.loc[:, ["open", "high", "low", "close", "volume"]]

            quality = check_quality(resampled)
            output_path = write_cache(self.cache_dir, key, resampled)
            self.provenance.log(
                ProvenanceRecord(
                    dataset="bars",
                    source="yfinance",
                    retrieved_at=utc_now(),
                    last_updated=None,
                    parameters={"symbol": symbol, "start": start, "end": end, "interval": interval, "quality": quality.__dict__},
                    output_path=str(output_path),
                )
            )
            data[symbol] = resampled
        return data


class YFinanceActionsProvider(CorporateActionsProvider):
    def __init__(self, cache_dir: Path, provenance: ProvenanceLogger):
        self.cache_dir = cache_dir
        self.provenance = provenance

    def get_actions(self, symbols: Iterable[str], start: str, end: str) -> dict[str, pd.DataFrame]:
        data: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            key = CacheKey(provider="yfinance_actions", symbol=symbol, start=start, end=end, interval="1d")
            cached = load_cache(self.cache_dir, key)
            if cached is not None:
                data[symbol] = cached
                continue
            ticker = yf.Ticker(symbol)
            actions = ticker.actions
            if not isinstance(actions, pd.DataFrame) or actions.empty:
                data[symbol] = pd.DataFrame()
                continue
            actions = actions.rename(columns=str.lower)
            # Ensure tz-consistent slicing (pandas errors on tz-aware vs tz-naive comparisons).
            idx = pd.to_datetime(actions.index, errors="coerce")
            if hasattr(idx, "tz") and idx.tz is not None:
                idx = idx.tz_convert(None)
            else:
                try:
                    idx = idx.tz_localize(None)
                except Exception:
                    pass
            actions = actions.copy()
            actions.index = idx
            start_ts = pd.to_datetime(start)
            end_ts = pd.to_datetime(end)
            actions = actions.loc[(actions.index >= start_ts) & (actions.index <= end_ts)]
            output_path = write_cache(self.cache_dir, key, actions)
            self.provenance.log(
                ProvenanceRecord(
                    dataset="corporate_actions",
                    source="yfinance",
                    retrieved_at=utc_now(),
                    last_updated=None,
                    parameters={"symbol": symbol, "start": start, "end": end},
                    output_path=str(output_path),
                )
            )
            data[symbol] = actions
        return data


class YFinanceBenchmarkProvider(BenchmarkProvider):
    def __init__(self, cache_dir: Path, provenance: ProvenanceLogger, source_interval: str, fallback_to_sample: bool):
        self.cache_dir = cache_dir
        self.provenance = provenance
        self.source_interval = source_interval
        self.fallback_to_sample = fallback_to_sample

    def get_benchmark(self, start: str, end: str, interval: str) -> pd.DataFrame:
        key = CacheKey(provider="yfinance_benchmark", symbol="SPY", start=start, end=end, interval=interval)
        cached = load_cache(self.cache_dir, key)
        if cached is not None:
            cache_path = self.cache_dir / key.provider / key.filename()
            sample_paths = _load_sample_paths(Path("logs/provenance.jsonl")) if not self.fallback_to_sample else set()
            if not self.fallback_to_sample and _is_sample_cache(sample_paths, cache_path):
                cached = None
            else:
                return cached
        try:
            ticker = yf.Ticker("SPY")
            raw = ticker.history(start=start, end=end, interval=self.source_interval, auto_adjust=False)
        except Exception:
            raw = pd.DataFrame()
        if raw.empty and self.fallback_to_sample:
            sample = generate_sample_bars("SPY", start, end, interval)
            output_path = write_cache(self.cache_dir, key, sample)
            self.provenance.log(
                ProvenanceRecord(
                    dataset="benchmark",
                    source="sample",
                    retrieved_at=utc_now(),
                    last_updated=None,
                    parameters={"symbol": "SPY", "start": start, "end": end, "interval": interval},
                    output_path=str(output_path),
                )
            )
            return sample
        if raw.empty:
            return pd.DataFrame()
        raw = raw.rename(columns=str.lower)
        raw = raw.tz_localize("UTC") if raw.index.tz is None else raw
        if str(self.source_interval).lower().endswith("d"):
            resampled = raw.copy()
            cols = [c for c in ["open", "high", "low", "close", "volume"] if c in resampled.columns]
            resampled = resampled.loc[:, cols]
        else:
            offset = _offset_from_time("09:30")
            resampled = resample_to_interval(raw, interval, offset=offset)
            resampled = resampled.loc[:, ["open", "high", "low", "close", "volume"]]
        output_path = write_cache(self.cache_dir, key, resampled)
        self.provenance.log(
            ProvenanceRecord(
                dataset="benchmark",
                source="yfinance",
                retrieved_at=utc_now(),
                last_updated=None,
                parameters={"symbol": "SPY", "start": start, "end": end, "interval": interval},
                output_path=str(output_path),
            )
        )
        return resampled


class WikipediaUniverseProvider(UniverseProvider):
    def __init__(self, cache_dir: Path, provenance: ProvenanceLogger, source_url: str, as_of: str):
        self.cache_dir = cache_dir
        self.provenance = provenance
        self.source_url = source_url
        self.as_of = as_of

    def get_universe(self) -> UniverseSnapshot:
        key = CacheKey(provider="universe", symbol="NASDAQ100", start=self.as_of, end=self.as_of, interval="snapshot")
        cached = load_cache(self.cache_dir, key)
        if cached is not None:
            members = cached["symbol"].tolist()
            if _valid_ticker_set(members):
                as_of = pd.to_datetime(self.as_of).to_pydatetime()
                return UniverseSnapshot(members=members, as_of=as_of, source=self.source_url)

        response = requests.get(
            self.source_url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; QuantModel/1.0)"},
            timeout=20,
        )
        response.raise_for_status()
        tables = pd.read_html(response.text)
        table = next((t for t in tables if "Ticker" in t.columns), tables[0])
        symbol_col = "Ticker" if "Ticker" in table.columns else table.columns[1]
        members = table[symbol_col].astype(str).str.replace(".", "-", regex=False).tolist()
        members = [m for m in members if _is_valid_ticker(m)]
        snapshot = pd.DataFrame({"symbol": members})
        output_path = write_cache(self.cache_dir, key, snapshot)
        self.provenance.log(
            ProvenanceRecord(
                dataset="universe",
                source=self.source_url,
                retrieved_at=utc_now(),
                last_updated=None,
                parameters={"as_of": self.as_of},
                output_path=str(output_path),
            )
        )
        as_of = pd.to_datetime(self.as_of).to_pydatetime()
        return UniverseSnapshot(members=members, as_of=as_of, source=self.source_url)


def _offset_from_time(open_time: str) -> str:
    hours, minutes = open_time.split(":")
    return f"{int(hours)}h{int(minutes)}min"


def _is_valid_ticker(symbol: str) -> bool:
    if not symbol:
        return False
    cleaned = symbol.strip().upper()
    if any(char.isdigit() for char in cleaned):
        return False
    if len(cleaned) > 6:
        return False
    return all(char.isalnum() or char in {".", "-"} for char in cleaned)


def _valid_ticker_set(symbols: list[str]) -> bool:
    if not symbols:
        return False
    return all(_is_valid_ticker(symbol) for symbol in symbols)



def _load_sample_paths(log_path: Path) -> set[str]:
    if not log_path.exists():
        return set()
    sample_paths: set[str] = set()
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("source") == "sample" and record.get("output_path"):
            sample_paths.add(str(record["output_path"]))
    return sample_paths


def _is_sample_cache(sample_paths: set[str], cache_path: Path) -> bool:
    return str(cache_path) in sample_paths
