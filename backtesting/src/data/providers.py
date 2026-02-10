from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Protocol

import pandas as pd


@dataclass(frozen=True)
class UniverseSnapshot:
    members: list[str]
    as_of: datetime
    source: str


class HistoricalBarsProvider(Protocol):
    def get_bars(self, symbols: Iterable[str], start: str, end: str, interval: str) -> dict[str, pd.DataFrame]:
        ...


class CorporateActionsProvider(Protocol):
    def get_actions(self, symbols: Iterable[str], start: str, end: str) -> dict[str, pd.DataFrame]:
        ...


class UniverseProvider(Protocol):
    def get_universe(self) -> UniverseSnapshot:
        ...


class BenchmarkProvider(Protocol):
    def get_benchmark(self, start: str, end: str, interval: str) -> pd.DataFrame:
        ...
