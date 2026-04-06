from __future__ import annotations

import numpy as np
import pandas as pd


def cagr(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) if years > 0 else 0.0


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    if returns.std() == 0:
        return 0.0
    return float((returns.mean() - rf / 252) / returns.std() * np.sqrt(252))


def sortino(returns: pd.Series, rf: float = 0.0) -> float:
    downside = returns[returns < 0]
    if downside.std() == 0:
        return 0.0
    return float((returns.mean() - rf / 252) / downside.std() * np.sqrt(252))


def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    return float(drawdown.min())


def volatility(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(252))


def win_rate(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    return float((trades["pnl"] > 0).mean())


def profit_factor(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    gross_profit = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gross_loss = trades.loc[trades["pnl"] < 0, "pnl"].sum()
    if gross_loss == 0:
        return float("inf")
    return float(gross_profit / abs(gross_loss))


def turnover(weights: pd.DataFrame) -> float:
    if weights.empty:
        return 0.0
    return float(weights.diff().abs().sum(axis=1).mean())


def alpha_beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> tuple[float, float]:
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if aligned.empty:
        return 0.0, 0.0
    x = aligned.iloc[:, 1].values
    y = aligned.iloc[:, 0].values
    beta = np.cov(x, y)[0, 1] / np.var(x) if np.var(x) != 0 else 0.0
    alpha = (y.mean() - beta * x.mean()) * 252
    return float(alpha), float(beta)
