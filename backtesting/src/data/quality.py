from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QualityReport:
    missing_bars: int
    outliers: int
    stale_prices: int
    survivorship_bias_note: str


def check_quality(frame: pd.DataFrame) -> QualityReport:
    missing_bars = int(frame.isna().any(axis=1).sum())
    returns = frame["close"].pct_change().dropna()
    outliers = int((np.abs(returns) > 0.2).sum())
    stale_prices = int((frame["close"].diff() == 0).rolling(10).sum().fillna(0).gt(9).sum())
    survivorship_note = (
        "Universe sourced from current constituents; survivorship bias likely unless historical membership is used."
    )
    return QualityReport(
        missing_bars=missing_bars,
        outliers=outliers,
        stale_prices=stale_prices,
        survivorship_bias_note=survivorship_note,
    )
