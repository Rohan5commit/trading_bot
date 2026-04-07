import json

import yaml

from trained_model_client import TrainedModelTradeClient


def load_config(path="config.yaml"):
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def main():
    config = load_config()
    ai_cfg = config.get("ai_trading", {}) if isinstance(config, dict) else {}
    client = TrainedModelTradeClient(ai_cfg)
    if not client.is_ready():
        raise SystemExit(client.last_error or "trained model client is not configured")
    candidate = {
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
    }
    prediction = client.predict_candidate(candidate)
    if not prediction:
        raise SystemExit(client.last_error or "trained model warmup failed")
    payload = {
        "ok": True,
        "model": client.last_model_used or client.model_identifier,
        "prediction": prediction,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
