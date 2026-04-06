from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


def compute_ic(db_path: str, window: int = 60) -> float:
    path = Path(db_path)
    if not path.exists():
        return 0.0
    with sqlite3.connect(path) as conn:
        df = pd.read_sql_query("SELECT * FROM predictions", conn)
    if df.empty or "score" not in df.columns or "realized_return" not in df.columns:
        return 0.0
    df = df.tail(window)
    return float(df["score"].corr(df["realized_return"]))


def drift_alert(db_path: str, threshold: float = 0.05, window: int = 60) -> bool:
    ic = compute_ic(db_path, window)
    return ic < threshold
