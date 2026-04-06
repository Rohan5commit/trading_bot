import json
import os
from datetime import datetime

import pandas as pd
import yaml

from ingest_prices import PriceIngestor
from llm_trader import propose_trades_with_llm


def load_config(path="config.yaml"):
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


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
        df = ingestor.fetch_stooq_data(symbol)
        if df is None or df.empty:
            failures.append({"symbol": symbol, "error": "no_price_data"})
            continue
        candidate = compute_candidate(symbol, df)
        if candidate is None:
            failures.append({"symbol": symbol, "error": "insufficient_history"})
            continue
        candidates.append(candidate)
    return candidates, failures


def main():
    config = load_config()
    tickers = [s.strip().upper() for s in os.getenv("AI_SMOKE_TICKERS", "AAPL,MSFT,NVDA,TSLA,SPY").split(",") if s.strip()]
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
