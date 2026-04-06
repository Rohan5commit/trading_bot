from __future__ import annotations

from typing import Tuple


def combine_scores(tabular: float, llm: float, technical: float, weights=(0.45, 0.35, 0.20)) -> float:
    w1, w2, w3 = weights
    return w1 * tabular + w2 * llm + w3 * technical
