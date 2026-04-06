from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict

import pandas as pd


def log_prediction(db_path: str, payload: Dict) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        pd.DataFrame([payload]).to_sql("predictions", conn, if_exists="append", index=False)
