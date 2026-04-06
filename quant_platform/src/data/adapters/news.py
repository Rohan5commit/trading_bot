from __future__ import annotations

from typing import List

import os
import pandas as pd
import requests
from bs4 import BeautifulSoup

from .hf_yahoo_finance import HfYahooFinanceAdapter


def _simple_sentiment(text: str) -> float:
    text = text.lower()
    pos = sum(word in text for word in ["beat", "growth", "upgrade", "strong", "record"])
    neg = sum(word in text for word in ["miss", "downgrade", "weak", "lawsuit", "drop"])
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


class NewsAdapter:
    def __init__(
        self,
        hf_enabled: bool = True,
        hf_cache_dir: str = "data/raw/hf_yahoo_finance",
        allow_fallbacks: bool = True,
    ) -> None:
        self.hf = HfYahooFinanceAdapter(enabled=hf_enabled, cache_dir=hf_cache_dir)
        self.allow_fallbacks = allow_fallbacks

    def get_news_features(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        hf_frame = self.hf.get_news_features(ticker, start, end)
        if not hf_frame.empty:
            return hf_frame
        if not self.allow_fallbacks:
            return pd.DataFrame()
        api_key = os.getenv("NEWSAPI_KEY")
        if api_key:
            return self._from_newsapi(ticker, start, end, api_key)
        return self._from_yahoo_rss(ticker)

    def _from_newsapi(self, ticker: str, start: str, end: str, api_key: str) -> pd.DataFrame:
        url = "https://newsapi.org/v2/everything"
        params = {"q": ticker, "from": start, "to": end, "language": "en", "apiKey": api_key}
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            return pd.DataFrame()
        articles = resp.json().get("articles", [])
        rows = []
        for article in articles:
            published = article.get("publishedAt")
            if not published:
                continue
            rows.append({"date": published[:10], "sentiment": _simple_sentiment(article.get("title", ""))})
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        return df.groupby("date").agg(news_count=("sentiment", "size"), news_sentiment=("sentiment", "mean"))

    def get_news_features_batch(self, tickers: List[str], start: str, end: str) -> dict[str, pd.DataFrame]:
        hf_frames = self.hf.get_news_features_batch(tickers, start, end)
        if not self.allow_fallbacks:
            return hf_frames

        out: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            frame = hf_frames.get(ticker, pd.DataFrame())
            out[ticker] = frame if not frame.empty else self.get_news_features(ticker, start, end)
        return out

    def _from_yahoo_rss(self, ticker: str) -> pd.DataFrame:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        try:
            resp = requests.get(url, timeout=30)
        except Exception:
            return pd.DataFrame()
        if resp.status_code != 200:
            return pd.DataFrame()
        soup = BeautifulSoup(resp.text, "xml")
        rows = []
        for item in soup.find_all("item"):
            title = item.title.text if item.title else ""
            pub = item.pubDate.text if item.pubDate else ""
            if not pub:
                continue
            date = pd.to_datetime(pub).date()
            rows.append({"date": date, "sentiment": _simple_sentiment(title)})
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        return df.groupby("date").agg(news_count=("sentiment", "size"), news_sentiment=("sentiment", "mean"))
