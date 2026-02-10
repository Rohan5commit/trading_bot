import datetime as dt

from src.engine.portfolio import Portfolio


def test_pnl_math_buy_sell():
    portfolio = Portfolio(1000)
    portfolio.execute_trade("AAPL", dt.datetime(2024, 1, 1), "BUY", 10, 10)
    portfolio.execute_trade("AAPL", dt.datetime(2024, 1, 2), "SELL", 10, 12)
    assert portfolio.cash == 1020
    assert portfolio.trades[-1].pnl == 20


def test_dividend_application():
    portfolio = Portfolio(1000)
    portfolio.execute_trade("AAPL", dt.datetime(2024, 1, 1), "BUY", 10, 10)
    portfolio.apply_dividend("AAPL", 1.0)
    assert portfolio.cash == 1000 - 100 + 10


def test_split_application():
    portfolio = Portfolio(1000)
    portfolio.execute_trade("AAPL", dt.datetime(2024, 1, 1), "BUY", 10, 10)
    portfolio.apply_split("AAPL", 2)
    position = portfolio.positions["AAPL"]
    assert position.shares == 20
    assert position.avg_cost == 5
