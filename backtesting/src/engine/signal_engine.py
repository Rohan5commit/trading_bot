from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.engine.meta_learning import model_selector
from src.engine.signals import Signal


@dataclass
class MetaState:
    weights: Dict[str, float]
    last_update: str | None


def load_meta_state(state_path: Path) -> MetaState:
    if not state_path.exists():
        return MetaState(weights={"momentum": 0.5, "mean_reversion": 0.5}, last_update=None)
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    return MetaState(weights=payload.get("weights", {"momentum": 0.5, "mean_reversion": 0.5}), last_update=payload.get("last_update"))


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
                timestamp=as_of.to_pydatetime(),
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
