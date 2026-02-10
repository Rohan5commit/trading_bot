# System Specification: Advanced Daily Trading Bot

## 1. Goal
Build a trading bot that integrates multiple data sources (Technical, GDELT Sentiment) into a Machine Learning pipeline for daily swing trading decisions, with a focus on "Daily Backtest" reporting.

## 2. Universe Definition
- **Primary Source**: `universe/tech.csv`.
- **Criteria**: US Tech stocks, filtered by market cap and sector.
- **Timeframe Policy**: "As needed" updates (daily check for new listings/delistings).

## 3. Data Ingestion
| Data Type | Source | Frequency | Cache |
|-----------|--------|-----------|-------|
| Prices (OHLCV) | Stooq / Free | Daily | SQLite |
| News/Sentiment | GDELT | Daily | SQLite |

## 4. Feature Engineering
- **Technical**: 10-day returns, RSI, Bollinger Bands, Volume/AvgVolume.
- **Sentiment**: 7-day rolling GDELT mention count and sentiment score (FinBERT/Lexicon).

## 5. ML Objective
- **Target**: Next-day return (Classification or Regression).
- **Model**: Gradient Boosted Trees (XGBoost/LightGBM).
- **Retrain**: Weekly walk-forward validation.

## 6. Trade Lifecycle
1. **Signal Generation (Post-Close Day T)**:
   - Ingest data for Day T.
   - Generate predictions for Day T+1.
2. **Daily Backtest (Post-Close Day T)**:
   - Evaluate signals generated at T-1 against actual Day T data.
   - Output performance report.
3. **Execution (Open Day T+1)**:
   - Place Limit orders at Open or VWAP proxy.
   - Monitor throughout the day for Risk/TP exits.
4. **Exit**: Market/Limit at Close or TP hit.

## 7. Risk Management
- **Hard Rails**: Max 5% position size, Max 80% total exposure, Kill-switch on 2% daily equity drop.
- **Dynamic**: Position sizing based on volatility (ATR) and model confidence.
