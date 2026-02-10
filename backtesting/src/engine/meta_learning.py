from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class StrategyScore:
    name: str
    weight: float


def online_update(signal_strengths: Dict[str, float], decay: float = 0.9) -> Dict[str, float]:
    updated: Dict[str, float] = {}
    for symbol, value in signal_strengths.items():
        updated[symbol] = decay * value + (1 - decay) * value
    return updated


def model_selector(recent_returns: Dict[str, List[float]]) -> List[StrategyScore]:
    scores: List[StrategyScore] = []
    for name, returns in recent_returns.items():
        if returns:
            avg = float(np.mean(returns))
        else:
            avg = 0.0
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
