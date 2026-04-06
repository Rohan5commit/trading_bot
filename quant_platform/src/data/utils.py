from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_datetime_index(df: pd.DataFrame, column: str | None = None) -> pd.DataFrame:
    if column is not None and column in df.columns:
        df = df.copy()
        df[column] = pd.to_datetime(df[column])
        df = df.set_index(column)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def safe_divide(a, b, default=0.0):
    try:
        return a / b if b not in (0, None) else default
    except Exception:
        return default


def select_numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def rolling_apply(series: pd.Series, window: int, func, min_periods: int = 1):
    return series.rolling(window=window, min_periods=min_periods).apply(func, raw=False)


def _merge_dict(target: dict, update: dict) -> dict:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_dict(target[key], value)
        else:
            target[key] = value
    return target


def load_progress(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def update_progress(path: str | Path, payload: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    data = load_progress(path)
    _merge_dict(data, payload)
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(data, indent=2))