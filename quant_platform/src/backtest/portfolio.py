from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class PortfolioState:
    equity: float
    positions: Dict[str, float]


def allocate_weights(scores: pd.Series, top_k: int, bottom_k: int, max_position: float) -> pd.Series:
    scores = scores.dropna()
    longs = scores.nlargest(top_k) if top_k > 0 else pd.Series(dtype=float)
    shorts = scores.nsmallest(bottom_k) if bottom_k > 0 else pd.Series(dtype=float)

    weights = pd.Series(0.0, index=scores.index)
    if not longs.empty:
        long_weights = longs / longs.abs().sum()
        weights[longs.index] = long_weights.clip(upper=max_position)
    if not shorts.empty:
        short_weights = shorts / shorts.abs().sum()
        weights[shorts.index] = -short_weights.clip(upper=max_position)
    return weights
