from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class Signal:
    symbol: str
    timestamp: datetime
    action: str
    entry_assumption: str
    confidence: float
    position_size: float
    rationale_tags: list[str]
    expected_holding_period: str
    entry_price: float | None = None

    def to_json(self) -> dict:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        if payload.get("entry_price") is None:
            payload.pop("entry_price", None)
        return payload


def generate_simple_momentum_signals(prices: pd.DataFrame, lookback: int) -> Signal:
    returns = prices["close"].pct_change(lookback).iloc[-1]
    action = "BUY" if returns > 0 else "SELL" if returns < 0 else "HOLD"
    confidence = float(min(1.0, abs(returns) * 10))
    position_size = float(min(1.0, abs(returns) * 2))
    tags = ["momentum", f"lookback_{lookback}"]
    return Signal(
        symbol=str(prices.attrs.get("symbol", "UNKNOWN")),
        timestamp=prices.index[-1].to_pydatetime(),
        action=action,
        entry_assumption="next_open",
        confidence=confidence,
        position_size=position_size,
        rationale_tags=tags,
        expected_holding_period=f"{lookback} bars",
    )


def normalize_signals(signals: Iterable[Signal], max_positions: int) -> list[Signal]:
    signals = list(signals)
    buys = [s for s in signals if s.action == "BUY"]
    if not buys:
        return signals
    size = 1.0 / min(max_positions, len(buys))
    normalized = []
    for signal in signals:
        if signal.action == "BUY":
            normalized.append(
                Signal(
                    **{**signal.__dict__, "position_size": size}
                )
            )
        else:
            normalized.append(signal)
    return normalized
