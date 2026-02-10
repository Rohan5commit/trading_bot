from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import pandas as pd

from src.engine.pine_adapter import load_pine_script, pine_to_strategy_config, UnsupportedPineScriptError
from src.engine.signals import Signal, generate_simple_momentum_signals


@dataclass(frozen=True)
class ExitRules:
    time_based_days: int
    indicator_based: str
    take_profit_tiers: list[dict]


def build_exit_rules(config: Dict) -> ExitRules:
    tiers = config.get("take_profit", {}).get("tiers", []) if config.get("take_profit", {}).get("enabled") else []
    return ExitRules(
        time_based_days=int(config.get("exit_rules", {}).get("time_based_days", 10)),
        indicator_based=str(config.get("exit_rules", {}).get("indicator_based", "sma_cross")),
        take_profit_tiers=tiers,
    )


def generate_signals(
    bars: Dict[str, pd.DataFrame],
    strategy_config: Dict,
) -> list[Signal]:
    strategy_type = strategy_config.get("type", "native")
    if strategy_type == "pine":
        script_path = strategy_config.get("pine_script_path")
        if not script_path:
            raise ValueError("strategy.pine_script_path is required for Pine strategies")
        pine_script = load_pine_script(script_path)
        try:
            pine_config = pine_to_strategy_config(pine_script)
        except UnsupportedPineScriptError as exc:
            raise ValueError(str(exc)) from exc
        return _generate_pine_signals(bars, pine_config, strategy_config)

    lookback = int(strategy_config.get("lookback_days", 20))
    signals: list[Signal] = []
    for symbol, frame in bars.items():
        frame = frame.copy()
        frame.attrs["symbol"] = symbol
        if len(frame) <= lookback:
            continue
        signal = generate_simple_momentum_signals(frame, lookback)
        signals.append(signal)
    return signals


def _generate_pine_signals(
    bars: Dict[str, pd.DataFrame],
    pine_config: Dict,
    strategy_config: Dict,
) -> list[Signal]:
    signals: list[Signal] = []
    indicators = pine_config.get("indicators", {})
    entry = pine_config.get("entry")
    exit_cond = pine_config.get("exit")
    for symbol, frame in bars.items():
        if frame.empty:
            continue
        frame = frame.copy()
        frame.attrs["symbol"] = symbol
        entry_hit = _evaluate_condition(frame, indicators, entry) if entry else False
        exit_hit = _evaluate_condition(frame, indicators, exit_cond) if exit_cond else False

        action = "HOLD"
        if exit_hit:
            action = "SELL"
        elif entry_hit:
            action = "BUY"

        confidence = 1.0 if action in {"BUY", "SELL"} else 0.0
        signals.append(
            Signal(
                symbol=symbol,
                timestamp=frame.index[-1].to_pydatetime(),
                action=action,
                entry_assumption="next_open",
                confidence=confidence,
                position_size=0.0,
                rationale_tags=["pine", entry.get("op") if entry else "signal"],
                expected_holding_period=strategy_config.get("expected_holding_period", "1d"),
                entry_price=float(frame["close"].iloc[-1]) if "close" in frame.columns else None,
            )
        )
    return signals


def _evaluate_condition(frame: pd.DataFrame, indicators: Dict, cond: Dict | None) -> bool:
    if not cond:
        return False
    op = cond.get("op")
    left = _resolve_series(frame, indicators, cond.get("left"))
    right = _resolve_series(frame, indicators, cond.get("right"))
    if left is None or right is None:
        return False

    if op in {"crossover", "crossunder"}:
        if len(left) < 2 or len(right) < 2:
            return False
        prev = left.iloc[-2] - right.iloc[-2]
        curr = left.iloc[-1] - right.iloc[-1]
        return prev <= 0 < curr if op == "crossover" else prev >= 0 > curr

    if op in {">", "<", ">=", "<="}:
        lhs = left.iloc[-1] if isinstance(left, pd.Series) else left
        rhs = right.iloc[-1] if isinstance(right, pd.Series) else right
        if op == ">":
            return lhs > rhs
        if op == "<":
            return lhs < rhs
        if op == ">=":
            return lhs >= rhs
        return lhs <= rhs

    return False


def _resolve_series(frame: pd.DataFrame, indicators: Dict, key):
    if key is None:
        return None
    if isinstance(key, (int, float)):
        return float(key)
    if key in frame.columns:
        return frame[key]
    if key not in indicators:
        return None
    indicator = indicators[key]
    source = indicator.get("source", "close")
    if source not in frame.columns:
        return None
    series = frame[source]
    length = int(indicator.get("length", 1))
    if indicator["type"] == "sma":
        return series.rolling(length).mean()
    if indicator["type"] == "ema":
        return series.ewm(span=length, adjust=False).mean()
    if indicator["type"] == "rsi":
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=length).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=length).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    return None
