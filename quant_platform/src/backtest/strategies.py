from __future__ import annotations

import pandas as pd


def momentum_breakout(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    rolling_high = df.groupby("ticker")["close"].transform(lambda x: x.rolling(lookback).max())
    return (df["close"] > rolling_high).astype(float)


def mean_reversion_pullback(df: pd.DataFrame, rsi_col: str = "rsi_14", threshold: float = 30) -> pd.Series:
    rsi = df[rsi_col]
    return (rsi < threshold).astype(float)


def factor_ranking(df: pd.DataFrame, factor_cols: list[str]) -> pd.Series:
    scores = df[factor_cols].mean(axis=1)
    return scores
