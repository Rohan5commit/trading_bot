from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from src.data.bars import align_to_market_calendar, filter_market_hours, resample_to_interval
from src.data.cache import CacheKey, load_cache, write_cache
from src.data.provenance import ProvenanceLogger, ProvenanceRecord, utc_now
from src.data.providers import BenchmarkProvider, HistoricalBarsProvider
from src.data.quality import check_quality
from src.data.sample import generate_sample_bars


class TwelveDataClient:
    def __init__(self, base_url: str, api_key: str | None, timeout_seconds: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()

    def fetch_time_series(self, symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
        if not self.api_key:
            return pd.DataFrame()
        params = {
            "symbol": symbol,
            "interval": interval,
            "start_date": start,
            "end_date": end,
            "timezone": "UTC",
            "format": "JSON",
            "apikey": self.api_key,
        }
        try:
            response = self.session.get(f"{self.base_url}/time_series", params=params, timeout=self.timeout_seconds)
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return pd.DataFrame()

        if not isinstance(payload, dict):
            return pd.DataFrame()
        if payload.get("status") == "error":
            return pd.DataFrame()
        values = payload.get("values")
        if not values:
            return pd.DataFrame()

        frame = pd.DataFrame(values)
        if "datetime" not in frame.columns:
            return pd.DataFrame()
        frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
        frame = frame.set_index("datetime").sort_index()

        for col in ["open", "high", "low", "close", "volume"]:
            if col not in frame.columns:
                frame[col] = 0.0
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

        frame = frame.loc[:, ["open", "high", "low", "close", "volume"]]
        return frame.dropna()


class TwelveDataBarsProvider(HistoricalBarsProvider):
    def __init__(
        self,
        cache_dir: Path,
        provenance: ProvenanceLogger,
        timezone: str,
        open_time: str,
        close_time: str,
        source_interval: str,
        fallback_to_sample: bool,
        api_key: str | None,
        base_url: str,
        timeout_seconds: int,
    ):
        self.cache_dir = cache_dir
        self.provenance = provenance
        self.timezone = timezone
        self.open_time = open_time
        self.close_time = close_time
        self.source_interval = source_interval
        self.fallback_to_sample = fallback_to_sample
        self.client = TwelveDataClient(base_url, api_key, timeout_seconds)

    def get_bars(self, symbols: Iterable[str], start: str, end: str, interval: str) -> dict[str, pd.DataFrame]:
        data: dict[str, pd.DataFrame] = {}
        sample_paths = _load_sample_paths(Path("logs/provenance.jsonl")) if not self.fallback_to_sample else set()
        for symbol in symbols:
            key = CacheKey(provider="twelvedata_bars", symbol=symbol, start=start, end=end, interval=interval)
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

            raw = self.client.fetch_time_series(symbol, start, end, self.source_interval)
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

            raw = raw.tz_localize("UTC") if raw.index.tz is None else raw

            # Daily source interval: TwelveData returns one bar per session (date-like timestamps).
            # The intraday filtering/resampling pipeline is intended for <=1h bars and can drop daily bars.
            if str(self.source_interval).lower().endswith("d"):
                resampled = raw.copy()
                resampled = resampled.loc[:, ["open", "high", "low", "close", "volume"]]
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
                    source="twelvedata",
                    retrieved_at=utc_now(),
                    last_updated=None,
                    parameters={
                        "symbol": symbol,
                        "start": start,
                        "end": end,
                        "interval": interval,
                        "quality": quality.__dict__,
                    },
                    output_path=str(output_path),
                )
            )
            data[symbol] = resampled
        return data


class TwelveDataBenchmarkProvider(BenchmarkProvider):
    def __init__(
        self,
        cache_dir: Path,
        provenance: ProvenanceLogger,
        source_interval: str,
        fallback_to_sample: bool,
        api_key: str | None,
        base_url: str,
        timeout_seconds: int,
    ):
        self.cache_dir = cache_dir
        self.provenance = provenance
        self.source_interval = source_interval
        self.fallback_to_sample = fallback_to_sample
        self.client = TwelveDataClient(base_url, api_key, timeout_seconds)

    def get_benchmark(self, start: str, end: str, interval: str) -> pd.DataFrame:
        key = CacheKey(provider="twelvedata_benchmark", symbol="SPY", start=start, end=end, interval=interval)
        cached = load_cache(self.cache_dir, key)
        if cached is not None:
            cache_path = self.cache_dir / key.provider / key.filename()
            sample_paths = _load_sample_paths(Path("logs/provenance.jsonl")) if not self.fallback_to_sample else set()
            if not self.fallback_to_sample and _is_sample_cache(sample_paths, cache_path):
                cached = None
            else:
                return cached

        raw = self.client.fetch_time_series("SPY", start, end, self.source_interval)
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

        raw = raw.tz_localize("UTC") if raw.index.tz is None else raw
        if str(self.source_interval).lower().endswith("d"):
            resampled = raw.copy()
            resampled = resampled.loc[:, ["open", "high", "low", "close", "volume"]]
        else:
            offset = _offset_from_time("09:30")
            resampled = resample_to_interval(raw, interval, offset=offset)
            resampled = resampled.loc[:, ["open", "high", "low", "close", "volume"]]
        output_path = write_cache(self.cache_dir, key, resampled)
        self.provenance.log(
            ProvenanceRecord(
                dataset="benchmark",
                source="twelvedata",
                retrieved_at=utc_now(),
                last_updated=None,
                parameters={"symbol": "SPY", "start": start, "end": end, "interval": interval},
                output_path=str(output_path),
            )
        )
        return resampled


def _offset_from_time(open_time: str) -> str:
    hours, minutes = open_time.split(":")
    return f"{int(hours)}h{int(minutes)}min"


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
