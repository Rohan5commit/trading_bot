import json
import os
from datetime import datetime

import pandas as pd
import yaml
import yfinance as yf

from ingest_prices import PriceIngestor
from llm_trader import propose_trades_with_llm


REQUIRED_PRICE_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def load_config(path="config.yaml"):
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def _normalize_prices(symbol, prices_df):
    if prices_df is None:
        return None
    df = prices_df.copy()
    if df.empty:
        return None

    if "date" not in df.columns:
        df = df.reset_index()

    normalized_columns = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_")
        normalized_columns[col] = key
    df = df.rename(columns=normalized_columns)

    if "datetime" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"datetime": "date"})
    if "adj_close" in df.columns and "close" not in df.columns:
        df = df.rename(columns={"adj_close": "close"})

    if "date" not in df.columns:
        return None

    if "volume" not in df.columns:
        df["volume"] = 0
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            return None

    df["symbol"] = str(symbol or "").strip().upper()
    return df[["symbol", *REQUIRED_PRICE_COLUMNS]]


def _fetch_yfinance_daily(symbol):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="1y", interval="1d", auto_adjust=False)
    return _normalize_prices(symbol, df)


def fetch_candidate_prices(ingestor, symbol):
    methods = [
        ("twelvedata", lambda: ingestor.fetch_twelvedata_daily(symbol)),
        ("yfinance", lambda: _fetch_yfinance_daily(symbol)),
        ("stooq", lambda: ingestor.fetch_stooq_data(symbol)),
    ]
    errors = []
    for source_name, loader in methods:
        try:
            df = loader()
        except Exception as exc:
            errors.append(f"{source_name}:{exc}")
            continue
        df = _normalize_prices(symbol, df)
        if df is not None and not df.empty:
            return df, source_name, None
        errors.append(f"{source_name}:empty")
    return None, None, "; ".join(errors)


def compute_candidate(symbol, prices_df):
    df = prices_df.copy()
    if df.empty or len(df) < 60:
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close", "volume"])
    if len(df) < 60:
        return None

    df["return_1d"] = df["close"].pct_change(1)
    df["return_5d"] = df["close"].pct_change(5)
    df["return_10d"] = df["close"].pct_change(10)
    df["volatility_20d"] = df["return_1d"].rolling(20).std()
    df["ma_20"] = df["close"].rolling(20).mean()
    df["ma_50"] = df["close"].rolling(50).mean()
    df["dist_ma_20"] = (df["close"] - df["ma_20"]) / df["ma_20"]
    df["dist_ma_50"] = (df["close"] - df["ma_50"]) / df["ma_50"]

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, pd.NA)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df["volume_ma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma_20"]

    latest = df.iloc[-1]
    tail_n = min(10, len(df))
    closes_tail = [float(x) for x in df["close"].tail(tail_n).tolist()]

    def _val(name):
        value = latest.get(name)
        if pd.isna(value):
            return None
        return float(value)

    return {
        "symbol": symbol,
        "as_of_date": str(pd.to_datetime(latest["date"]).date()),
        "last_date": str(pd.to_datetime(latest["date"]).date()),
        "last_close": float(latest["close"]),
        "closes_tail": closes_tail,
        "volume_1d": float(latest["volume"]),
        "volume_20d_avg": _val("volume_ma_20"),
        "return_1d": _val("return_1d"),
        "return_5d": _val("return_5d"),
        "return_10d": _val("return_10d"),
        "volatility_20d": _val("volatility_20d"),
        "dist_ma_20": _val("dist_ma_20"),
        "dist_ma_50": _val("dist_ma_50"),
        "rsi_14": _val("rsi_14"),
        "volume_ratio": _val("volume_ratio"),
        "news_count_7d": 0,
        "news_sentiment_7d": 0.0,
    }


def build_candidates(config, tickers):
    ingestor = PriceIngestor()
    candidates = []
    failures = []
    for symbol in tickers:
        df, source_name, error = fetch_candidate_prices(ingestor, symbol)
        if df is None or df.empty:
            failures.append({"symbol": symbol, "error": error or "no_price_data"})
            continue
        candidate = compute_candidate(symbol, df)
        if candidate is None:
            failures.append({"symbol": symbol, "error": f"insufficient_history:{source_name}"})
            continue
        candidate["price_source"] = source_name
        candidates.append(candidate)
    return candidates, failures


def build_static_candidates():
    return [
        {
            "symbol": "AAPL",
            "as_of_date": "2026-04-04",
            "last_date": "2026-04-04",
            "last_close": 188.4,
            "closes_tail": [185.1, 186.2, 187.3, 188.4],
            "volume_1d": 55321000.0,
            "volume_20d_avg": 50234000.0,
            "return_1d": 0.6,
            "return_5d": 2.1,
            "return_10d": 3.0,
            "volatility_20d": 0.22,
            "dist_ma_20": 0.03,
            "dist_ma_50": 0.06,
            "rsi_14": 58.0,
            "volume_ratio": 1.1,
            "news_count_7d": 4,
            "news_sentiment_7d": 0.2,
            "price_source": "static_smoke",
        },
        {
            "symbol": "TSLA",
            "as_of_date": "2026-04-04",
            "last_date": "2026-04-04",
            "last_close": 166.2,
            "closes_tail": [172.3, 170.5, 168.1, 166.2],
            "volume_1d": 97412000.0,
            "volume_20d_avg": 81234000.0,
            "return_1d": -1.1,
            "return_5d": -3.4,
            "return_10d": -5.2,
            "volatility_20d": 0.46,
            "dist_ma_20": -0.07,
            "dist_ma_50": -0.11,
            "rsi_14": 38.0,
            "volume_ratio": 1.3,
            "news_count_7d": 7,
            "news_sentiment_7d": -0.25,
            "price_source": "static_smoke",
        },
        {
            "symbol": "MSFT",
            "as_of_date": "2026-04-04",
            "last_date": "2026-04-04",
            "last_close": 421.7,
            "closes_tail": [417.4, 419.3, 420.8, 421.7],
            "volume_1d": 23124000.0,
            "volume_20d_avg": 24456000.0,
            "return_1d": 0.2,
            "return_5d": 1.0,
            "return_10d": 1.8,
            "volatility_20d": 0.18,
            "dist_ma_20": 0.02,
            "dist_ma_50": 0.04,
            "rsi_14": 55.0,
            "volume_ratio": 0.95,
            "news_count_7d": 3,
            "news_sentiment_7d": 0.1,
            "price_source": "static_smoke",
        },
    ], []


def main():
    config = load_config()
    use_static = str(os.getenv("AI_SMOKE_USE_STATIC", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}
    tickers = [s.strip().upper() for s in os.getenv("AI_SMOKE_TICKERS", "AAPL,MSFT,NVDA,TSLA,SPY").split(",") if s.strip()]
    if use_static:
        candidates, failures = build_static_candidates()
        if tickers:
            tickers_set = set(tickers)
            candidates = [candidate for candidate in candidates if candidate["symbol"] in tickers_set]
    else:
        candidates, failures = build_candidates(config, tickers)

    ai_cfg = config.get("ai_trading", {}) if isinstance(config, dict) else {}
    trades, status = propose_trades_with_llm(
        config,
        candidates,
        max_positions=min(int(ai_cfg.get("max_positions", 10) or 10), max(1, len(candidates))),
        allow_shorts=bool(ai_cfg.get("allow_shorts", True)),
        max_shorts=int(ai_cfg.get("max_shorts", 5) or 5),
    )

    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "tickers": tickers,
        "candidates_built": len(candidates),
        "candidate_failures": failures,
        "status": status,
        "trades": trades,
        "price_sources": {c["symbol"]: c.get("price_source") for c in candidates},
    }

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"ai_smoke_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json")
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)

    print(json.dumps(payload, indent=2))
    if not status.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
