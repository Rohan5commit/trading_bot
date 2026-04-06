from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curve(equity: pd.Series, path: str) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(equity.index, equity.values)
    plt.title("Equity Curve")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_drawdown(equity: pd.Series, path: str) -> None:
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    plt.figure(figsize=(10, 4))
    plt.plot(drawdown.index, drawdown.values)
    plt.title("Drawdown")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
