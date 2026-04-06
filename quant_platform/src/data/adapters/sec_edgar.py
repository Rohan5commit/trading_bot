from __future__ import annotations

from pathlib import Path
import pandas as pd


class SecEdgarAdapter:
    def __init__(self, base_path: str = "data/raw/sec_edgar") -> None:
        self.base_path = Path(base_path)

    def get_sec_features(self, ticker: str) -> pd.DataFrame:
        path_parquet = self.base_path / f"{ticker}.parquet"
        path_csv = self.base_path / f"{ticker}.csv"
        if path_parquet.exists():
            df = pd.read_parquet(path_parquet)
        elif path_csv.exists():
            df = pd.read_csv(path_csv)
        else:
            return pd.DataFrame()
        if df.empty or "date" not in df.columns:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        return df
