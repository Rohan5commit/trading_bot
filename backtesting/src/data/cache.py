from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class CacheKey:
    provider: str
    symbol: str
    start: str
    end: str
    interval: str

    def filename(self) -> str:
        safe_symbol = self.symbol.replace("/", "-")
        return f"{safe_symbol}_{self.start}_{self.end}_{self.interval}.parquet"


def load_cache(cache_dir: Path, key: CacheKey) -> Optional[pd.DataFrame]:
    path = cache_dir / key.provider / key.filename()
    if not path.exists():
        return None
    return pd.read_parquet(path)


def write_cache(cache_dir: Path, key: CacheKey, frame: pd.DataFrame) -> Path:
    provider_dir = cache_dir / key.provider
    provider_dir.mkdir(parents=True, exist_ok=True)
    path = provider_dir / key.filename()
    frame.to_parquet(path)
    return path
