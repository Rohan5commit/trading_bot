from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict

import pandas as pd


@dataclass
class Position:
    symbol: str
    shares: float
    avg_cost: float
    entry_time: datetime


@dataclass
class Trade:
    symbol: str
    timestamp: datetime
    action: str
    shares: float
    price: float
    pnl: float


class Portfolio:
    def __init__(self, cash: float) -> None:
        self.cash = cash
        self.positions: Dict[str, Position] = {}
        self.trades: list[Trade] = []

    def total_value(self, prices: Dict[str, float]) -> float:
        equity = self.cash
        for symbol, position in self.positions.items():
            equity += position.shares * prices.get(symbol, position.avg_cost)
        return equity

    def apply_dividend(self, symbol: str, dividend: float) -> None:
        if symbol in self.positions:
            self.cash += self.positions[symbol].shares * dividend

    def apply_split(self, symbol: str, split_ratio: float) -> None:
        if symbol in self.positions and split_ratio > 0:
            position = self.positions[symbol]
            position.shares *= split_ratio
            position.avg_cost /= split_ratio

    def execute_trade(self, symbol: str, timestamp: datetime, action: str, shares: float, price: float) -> None:
        if action == "BUY":
            if self.cash <= 0 or price <= 0 or shares <= 0:
                return
            cost = shares * price
            if cost > self.cash:
                shares = self.cash / price
                cost = shares * price
            if shares <= 0:
                return
            self.cash -= cost
            if symbol in self.positions:
                position = self.positions[symbol]
                total_shares = position.shares + shares
                position.avg_cost = (position.avg_cost * position.shares + price * shares) / total_shares
                position.shares = total_shares
            else:
                self.positions[symbol] = Position(symbol=symbol, shares=shares, avg_cost=price, entry_time=timestamp)
            self.trades.append(Trade(symbol, timestamp, action, shares, price, 0.0))
        elif action == "SELL":
            if symbol not in self.positions:
                return
            position = self.positions[symbol]
            sell_shares = min(shares, position.shares)
            proceeds = sell_shares * price
            self.cash += proceeds
            pnl = (price - position.avg_cost) * sell_shares
            position.shares -= sell_shares
            self.trades.append(Trade(symbol, timestamp, action, sell_shares, price, pnl))
            if position.shares <= 0:
                del self.positions[symbol]
        else:
            raise ValueError(f"Unsupported action: {action}")

    def mark_to_market(self, timestamp: datetime, prices: Dict[str, float]) -> dict:
        return {
            "timestamp": timestamp,
            "equity": self.total_value(prices),
            "cash": self.cash,
            **{f"pos_{symbol}": position.shares for symbol, position in self.positions.items()},
        }
