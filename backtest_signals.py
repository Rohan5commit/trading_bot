import json
import logging
import os
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Signal:
    symbol: str
    timestamp: pd.Timestamp
    action: str
    entry_assumption: str
    confidence: float
    position_size: float
    rationale_tags: list[str]
    expected_holding_period: str
    entry_price: Optional[float] = None

    def to_json(self) -> dict:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        if payload.get("entry_price") is None:
            payload.pop("entry_price", None)
        return payload


@dataclass
class StrategyScore:
    name: str
    weight: float


@dataclass
class MetaState:
    weights: Dict[str, float]
    last_update: Optional[str]


def _load_config(config_path):
    import yaml
    with open(config_path, "r") as handle:
        return yaml.safe_load(handle)


def _resolve_path(base_dir, path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(base_dir, path_value)


def load_meta_state(state_path: Path) -> MetaState:
    if not state_path.exists():
        return MetaState(weights={"momentum": 0.5, "mean_reversion": 0.5}, last_update=None)
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    return MetaState(
        weights=payload.get("weights", {"momentum": 0.5, "mean_reversion": 0.5}),
        last_update=payload.get("last_update")
    )


def save_meta_state(state_path: Path, state: MetaState) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps({"weights": state.weights, "last_update": state.last_update}, indent=2),
        encoding="utf-8",
    )


def _sigmoid(value: float) -> float:
    return 1 / (1 + np.exp(-value))


def _momentum_score(prices: pd.Series, lookback: int) -> float:
    if len(prices) <= lookback:
        return 0.0
    ret = prices.pct_change(lookback).iloc[-1]
    return float(_sigmoid(ret * 10))


def _mean_reversion_score(prices: pd.Series, window: int) -> float:
    if len(prices) < window:
        return 0.0
    sma = prices.rolling(window).mean().iloc[-1]
    std = prices.rolling(window).std().iloc[-1]
    if std == 0 or np.isnan(std):
        return 0.0
    z = (prices.iloc[-1] - sma) / std
    return float(_sigmoid(-z))


def compute_daily_scores(
    bars: Dict[str, pd.DataFrame],
    lookback: int,
    mean_reversion_window: int,
) -> Dict[str, Dict[str, float]]:
    scores: Dict[str, Dict[str, float]] = {}
    for symbol, frame in bars.items():
        daily = frame.resample("1D").last().dropna()
        prices = daily["close"] if "close" in daily.columns else daily.iloc[:, 0]
        momentum = _momentum_score(prices, lookback)
        mean_reversion = _mean_reversion_score(prices, mean_reversion_window)
        scores[symbol] = {"momentum": momentum, "mean_reversion": mean_reversion}
    return scores


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, value) for value in weights.values())
    if total == 0:
        return {key: 1.0 / len(weights) for key in weights}
    return {key: max(0.0, value) / total for key, value in weights.items()}


def model_selector(recent_returns: Dict[str, List[float]]) -> List[StrategyScore]:
    scores: List[StrategyScore] = []
    for name, returns in recent_returns.items():
        avg = float(np.mean(returns)) if returns else 0.0
        weight = max(0.0, avg)
        scores.append(StrategyScore(name=name, weight=weight))
    total = sum(score.weight for score in scores)
    if total == 0:
        for score in scores:
            score.weight = 1.0 / len(scores) if scores else 0.0
    else:
        for score in scores:
            score.weight /= total
    return scores


def update_meta_learning(
    scores: Dict[str, Dict[str, float]],
    latest_prices: Dict[str, float],
    config: dict,
    state_path: Path,
    history_path: Path,
) -> MetaState:
    state = load_meta_state(state_path)
    mode = config.get("meta_learning", {}).get("mode", "online_update")
    learning_rate = float(config.get("meta_learning", {}).get("learning_rate", 0.1))
    today = pd.Timestamp.utcnow().date().isoformat()

    if state.last_update == today:
        return state

    if history_path.exists():
        lines = history_path.read_text(encoding="utf-8").strip().splitlines()
    else:
        lines = []

    momentum_perf = 0.0
    mean_reversion_perf = 0.0
    momentum_returns: List[float] = []
    mean_reversion_returns: List[float] = []

    for line in lines:
        if not line:
            continue
        record = json.loads(line)
        signal_date = record.get("signal_date")
        if signal_date == state.last_update:
            continue
        symbol = record.get("symbol")
        entry_price = record.get("entry_price")
        if not symbol or entry_price is None:
            continue
        current_price = latest_prices.get(symbol)
        if not current_price:
            continue
        realized_return = (current_price / entry_price) - 1
        momentum_score = float(record.get("scores", {}).get("momentum", 0.0))
        mean_reversion_score = float(record.get("scores", {}).get("mean_reversion", 0.0))
        momentum_perf += momentum_score * realized_return
        mean_reversion_perf += mean_reversion_score * realized_return
        momentum_returns.append(realized_return * momentum_score)
        mean_reversion_returns.append(realized_return * mean_reversion_score)

    if mode == "model_selector":
        strategy_scores = model_selector({"momentum": momentum_returns, "mean_reversion": mean_reversion_returns})
        state.weights = {score.name: score.weight for score in strategy_scores}
    else:
        weights = dict(state.weights)
        weights["momentum"] = weights.get("momentum", 0.5) + learning_rate * momentum_perf
        weights["mean_reversion"] = weights.get("mean_reversion", 0.5) + learning_rate * mean_reversion_perf
        state.weights = _normalize_weights(weights)

    state.last_update = today
    save_meta_state(state_path, state)
    return state


def generate_buy_signals(
    scores: Dict[str, Dict[str, float]],
    weights: Dict[str, float],
    config: dict,
    as_of: pd.Timestamp,
    latest_prices: Dict[str, float],
) -> List[Signal]:
    buy_threshold = float(config.get("signals", {}).get("buy_threshold", 0.55))
    signals: List[Signal] = []
    for symbol, per_symbol in scores.items():
        confidence = sum(weights.get(key, 0.0) * per_symbol.get(key, 0.0) for key in weights)
        signals.append(
            Signal(
                symbol=symbol,
                timestamp=as_of,
                action="BUY",
                entry_assumption="next_open",
                confidence=float(min(1.0, max(0.0, confidence))),
                position_size=0.0,
                rationale_tags=["daily_signal"],
                expected_holding_period=config.get("signals", {}).get("expected_holding_period", "1d"),
                entry_price=latest_prices.get(symbol),
            )
        )
    signals.sort(key=lambda s: s.confidence, reverse=True)
    top_n = int(config.get("signals", {}).get("top_n", 10))
    filtered = [s for s in signals if s.confidence >= buy_threshold]
    selected = filtered[:top_n] if filtered else signals[:top_n]
    if not filtered:
        selected = [
            Signal(
                **{**signal.__dict__, "rationale_tags": signal.rationale_tags + ["below_threshold"]}
            )
            for signal in selected
        ]
    return selected


def record_signal_history(
    signals: List[Signal],
    scores: Dict[str, Dict[str, float]],
    latest_prices: Dict[str, float],
    history_path: Path,
) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    signal_date = pd.Timestamp.utcnow().date().isoformat()
    with history_path.open("a", encoding="utf-8") as handle:
        for signal in signals:
            entry_price = latest_prices.get(signal.symbol)
            record = {
                "signal_date": signal_date,
                "symbol": signal.symbol,
                "entry_price": float(entry_price) if entry_price is not None else None,
                "confidence": signal.confidence,
                "scores": scores.get(signal.symbol, {}),
            }
            handle.write(json.dumps(record) + "\n")


def _load_bars_from_db(db_path: str, symbols: list[str], lookback_days: int):
    conn = sqlite3.connect(db_path)
    max_date = pd.read_sql("SELECT MAX(date) as max_date FROM prices", conn).iloc[0, 0]
    if not max_date:
        conn.close()
        return {}, None
    end_date = pd.to_datetime(max_date)
    start_date = (end_date - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    bars = {}
    for symbol in symbols:
        df = pd.read_sql(
            "SELECT date, close FROM prices WHERE symbol=? AND date>=? AND date<=? ORDER BY date",
            conn,
            params=(symbol, start_date, end_str)
        )
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df.attrs["symbol"] = symbol
        bars[symbol] = df

    conn.close()
    return bars, end_date


def build_signal_snapshot(config_path: str, symbols: list[str]) -> list[dict]:
    config = _load_config(config_path)
    base_dir = os.path.dirname(os.path.abspath(config_path))
    db_path = _resolve_path(base_dir, config["data"]["cache_path"])

    signals_cfg = config.get("signals", {})
    if not signals_cfg.get("enabled", False):
        return []

    lookback_days = int(signals_cfg.get("lookback_days", 120))
    bars, end_date = _load_bars_from_db(db_path, symbols, lookback_days)
    if not bars:
        return []

    scores = compute_daily_scores(
        bars,
        lookback=int(signals_cfg.get("momentum_lookback", 20)),
        mean_reversion_window=int(signals_cfg.get("mean_reversion_window", 20)),
    )
    latest_prices = {symbol: frame["close"].iloc[-1] for symbol, frame in bars.items() if not frame.empty}

    state_path = Path(base_dir) / "models" / "backtest_meta_state.json"
    history_path = Path(base_dir) / "models" / "backtest_signal_history.jsonl"
    meta_state = update_meta_learning(scores, latest_prices, config, state_path, history_path)

    as_of = pd.Timestamp(end_date)
    signals = generate_buy_signals(scores, meta_state.weights, config, as_of, latest_prices)
    record_signal_history(signals, scores, latest_prices, history_path)

    return [s.to_json() for s in signals]
