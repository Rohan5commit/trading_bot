from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from .adapters.calendar import trading_days
from .adapters.ohlcv import OhlcvAdapter
from .adapters.fundamentals import FundamentalsAdapter
from .adapters.news import NewsAdapter
from .adapters.macro import MacroAdapter
from .adapters.transcripts import TranscriptsAdapter
from .adapters.sentiment import SentimentAdapter
from .adapters.tradingview import TradingViewAdapter
from .adapters.sec_edgar import SecEdgarAdapter
from .adapters.sec_fsds import SecFsdsAdapter, SecFsdsConfig
from .indicators import compute_indicators
from .normalization import rolling_zscore
from .utils import ensure_dir, select_numeric_columns, update_progress


class FeatureStoreConfig(BaseModel):
    raw_path: str = "data/raw"
    processed_path: str = "data/processed"
    calendar: str = "NYSE"
    zscore_window: int = 252
    min_periods: int = 20
    forward_windows: List[int] = Field(default_factory=lambda: [1, 5, 20])
    indicator_mode: str = "expanded"
    ohlcv_batch: bool = True
    ohlcv_batch_size: int = 50
    hf_yahoo_enabled: bool = True
    hf_yahoo_cache_dir: str | None = None
    allow_source_fallbacks: bool = True
    sec_fsds: SecFsdsConfig = Field(default_factory=SecFsdsConfig)


class FeatureStore:
    def __init__(self, config: FeatureStoreConfig | None = None) -> None:
        self.config = config or FeatureStoreConfig()
        hf_cache_dir = self.config.hf_yahoo_cache_dir or str(Path(self.config.raw_path) / "hf_yahoo_finance")
        self.ohlcv = OhlcvAdapter(
            hf_enabled=self.config.hf_yahoo_enabled,
            hf_cache_dir=hf_cache_dir,
            allow_fallbacks=self.config.allow_source_fallbacks,
        )
        self.fundamentals = FundamentalsAdapter(
            hf_enabled=self.config.hf_yahoo_enabled,
            hf_cache_dir=hf_cache_dir,
            allow_fallbacks=self.config.allow_source_fallbacks,
        )
        self.news = NewsAdapter(
            hf_enabled=self.config.hf_yahoo_enabled,
            hf_cache_dir=hf_cache_dir,
            allow_fallbacks=self.config.allow_source_fallbacks,
        )
        self.macro = MacroAdapter()
        self.transcripts = TranscriptsAdapter(
            base_path=str(Path(self.config.raw_path) / "transcripts"),
            hf_enabled=self.config.hf_yahoo_enabled,
            hf_cache_dir=hf_cache_dir,
            allow_fallbacks=self.config.allow_source_fallbacks,
        )
        self.sentiment = SentimentAdapter()
        self.tradingview = TradingViewAdapter()
        self.sec_edgar = SecEdgarAdapter()
        self.sec_fsds = SecFsdsAdapter(self.config.sec_fsds)

    def build_feature_frame(
        self,
        tickers: Iterable[str],
        start: str,
        end: str,
        normalize: bool = True,
        refresh: bool = False,
    ) -> pd.DataFrame:
        processed_path = Path(self.config.processed_path)
        ensure_dir(processed_path)
        raw_path = processed_path / "features_raw.parquet"
        norm_path = processed_path / "features_norm.parquet"
        progress_path = processed_path / "progress.json"
        return self._build_feature_frame_internal(
            tickers,
            start,
            end,
            normalize,
            refresh,
            raw_path=raw_path,
            norm_path=norm_path,
            progress_path=progress_path,
            progress_key="features",
        )

    def build_feature_frame_chunk(
        self,
        tickers: Iterable[str],
        start: str,
        end: str,
        chunk_id: int,
        normalize: bool = True,
        refresh: bool = False,
        output_dir: str | None = None,
    ) -> pd.DataFrame:
        processed_path = Path(output_dir or self.config.processed_path)
        chunk_dir = processed_path / "chunks"
        ensure_dir(chunk_dir)
        raw_path = chunk_dir / f"features_raw_chunk_{chunk_id}.parquet"
        norm_path = chunk_dir / f"features_norm_chunk_{chunk_id}.parquet"
        progress_path = processed_path / "progress.json"
        return self._build_feature_frame_internal(
            tickers,
            start,
            end,
            normalize,
            refresh,
            raw_path=raw_path,
            norm_path=norm_path,
            progress_path=progress_path,
            progress_key=f"features_chunk_{chunk_id}",
        )

    def _build_feature_frame_internal(
        self,
        tickers: Iterable[str],
        start: str,
        end: str,
        normalize: bool,
        refresh: bool,
        raw_path: Path,
        norm_path: Path,
        progress_path: Path,
        progress_key: str,
    ) -> pd.DataFrame:
        if raw_path.exists() and norm_path.exists() and not refresh:
            return pd.read_parquet(norm_path if normalize else raw_path)

        if self.config.sec_fsds.enabled:
            self.sec_fsds.prepare()

        frames = []
        tickers_list = list(tickers)
        total_tickers = len(tickers_list)
        days = trading_days(start, end, calendar=self.config.calendar)
        macro_frame = self.macro.get_macro_frame(start, end)
        processed_tickers = 0
        update_progress(
            progress_path,
            {progress_key: {"status": "running", "total_tickers": total_tickers, "processed_tickers": 0}},
        )

        def _chunk(items: List[str], size: int) -> Iterable[List[str]]:
            for i in range(0, len(items), size):
                yield items[i : i + size]

        if self.config.ohlcv_batch and self.config.ohlcv_batch_size > 1:
            for batch in _chunk(tickers_list, self.config.ohlcv_batch_size):
                ohlcv_map = self.ohlcv.get_ohlcv_batch(batch, start, end, interval="1d")
                fundamentals_map = self.fundamentals.get_fundamentals_batch(batch, days)
                news_map = self.news.get_news_features_batch(batch, start, end)
                transcripts_map = self.transcripts.get_transcript_features_batch(batch)
                for ticker in batch:
                    frame = self._build_ticker_frame(
                        ticker,
                        start,
                        end,
                        days=days,
                        ohlcv=ohlcv_map.get(ticker, pd.DataFrame()),
                        fundamentals=fundamentals_map.get(ticker),
                        news=news_map.get(ticker),
                        transcripts=transcripts_map.get(ticker),
                        macro=macro_frame,
                    )
                    if frame.empty:
                        processed_tickers += 1
                        continue
                    frame["ticker"] = ticker
                    frames.append(frame.reset_index().rename(columns={"index": "date"}))
                    processed_tickers += 1
                update_progress(
                    progress_path,
                    {progress_key: {"processed_tickers": processed_tickers}},
                )
        else:
            for ticker in tickers_list:
                frame = self._build_ticker_frame(ticker, start, end, days=days, macro=macro_frame)
                processed_tickers += 1
                if frame.empty:
                    continue
                frame["ticker"] = ticker
                frames.append(frame.reset_index().rename(columns={"index": "date"}))
                if processed_tickers % 50 == 0:
                    update_progress(
                        progress_path,
                        {progress_key: {"processed_tickers": processed_tickers}},
                    )

        update_progress(
            progress_path,
            {progress_key: {"status": "done", "processed_tickers": processed_tickers}},
        )

        if not frames:
            return pd.DataFrame()

        raw = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])
        for col in raw.columns:
            if col in {"ticker", "date"}:
                continue
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        raw.replace([np.inf, -np.inf], np.nan, inplace=True)
        raw.to_parquet(raw_path, index=False)

        if normalize:
            normalized = self._normalize(raw)
            normalized.to_parquet(norm_path, index=False)
            return normalized
        return raw

    def get_features(self, ticker: str, date: str) -> Dict:
        processed_path = Path(self.config.processed_path)
        norm_path = processed_path / "features_norm.parquet"
        if not norm_path.exists():
            raise FileNotFoundError("features_norm.parquet not found. Build features first.")
        df = pd.read_parquet(norm_path)
        row = df[(df["ticker"] == ticker) & (df["date"] == pd.to_datetime(date))]
        if row.empty:
            raise ValueError(f"No features for {ticker} on {date}")
        return row.iloc[-1].to_dict()

    def _build_ticker_frame(
        self,
        ticker: str,
        start: str,
        end: str,
        days: pd.DatetimeIndex | None = None,
        ohlcv: pd.DataFrame | None = None,
        fundamentals: pd.DataFrame | None = None,
        fsds: pd.DataFrame | None = None,
        news: pd.DataFrame | None = None,
        transcripts: pd.DataFrame | None = None,
        sentiment: pd.DataFrame | None = None,
        macro: pd.DataFrame | None = None,
        tradingview: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if days is None:
            days = trading_days(start, end, calendar=self.config.calendar)
        if ohlcv is None:
            ohlcv = self.ohlcv.get_ohlcv(ticker, start, end, interval="1d")
        if ohlcv.empty:
            return pd.DataFrame()
        ohlcv = ohlcv.reindex(days).ffill()
        features = compute_indicators(ohlcv, mode=self.config.indicator_mode)

        if fundamentals is None:
            fundamentals = self.fundamentals.get_fundamentals(ticker, days)
        if fsds is None:
            fsds = self.sec_fsds.get_fsds_features(ticker, days)
        if news is None:
            news = self.news.get_news_features(ticker, start, end)
        if transcripts is None:
            transcripts = self.transcripts.get_transcript_features(ticker)
        if sentiment is None:
            sentiment = self.sentiment.get_sentiment_features(ticker, start, end)
        if macro is None:
            macro = self.macro.get_macro_frame(start, end)
        if tradingview is None:
            tradingview = self.tradingview.get_tradingview_features(ticker)
        aligned_frames: list[pd.DataFrame] = [features]
        for candidate in (fundamentals, fsds, news, transcripts, sentiment, macro):
            if candidate is not None and not candidate.empty:
                aligned_frames.append(candidate.reindex(days).ffill())
        if tradingview is not None and not tradingview.empty:
            aligned_frames.append(tradingview.reindex(days).ffill())

        frame = pd.concat(aligned_frames, axis=1).ffill()
        frame.index.name = "date"
        return frame

    def _normalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        raw = raw.copy()
        numeric_cols = select_numeric_columns(raw.drop(columns=["date", "ticker"]))
        normalized_frames = []
        for ticker, group in raw.groupby("ticker"):
            group = group.sort_values("date")
            z = rolling_zscore(
                group[numeric_cols], window=self.config.zscore_window, min_periods=self.config.min_periods
            )
            group[numeric_cols] = z
            normalized_frames.append(group)
        return pd.concat(normalized_frames, ignore_index=True)
