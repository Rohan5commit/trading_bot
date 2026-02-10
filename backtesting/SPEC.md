# Quant Model Repo Specification

This repository provides a fully local, reproducible quant research and backtest stack with free data sources, deterministic caching, and CI-style local commands. Configuration is editable without code changes via `config.yaml`.

## Core requirements mapping

- **Configuration**: `config.yaml` (universe, bars, portfolio, objective, risk rails, data sources, strategy, meta-learning, reporting).
- **Local commands**: `make lint`, `make test`, `make backtest`, `make report`, `make run_eod`.
- **Data layer (free)**: providers in `src/data/providers.py` using free sources (YFinance and Twelve Data); caching and provenance in `src/data/cache.py` and `src/data/provenance.py`.
- **Signal output schema**: `src/schemas/signal_output_schema.json` and `src/engine/signals.py`.
- **Backtesting engine**: `src/engine/backtest.py` with walk-forward, portfolio accounting, corporate actions, and benchmark.
- **Meta-learning loop**: `src/engine/meta_learning.py` with `online_update` and `model_selector` modes.
- **Tests**: `tests/` cover required unit/regression checks.

## Outputs per run

- `reports/latest/trades.csv`
- `reports/latest/daily_equity.csv`
- `reports/latest/metrics.json`
- `reports/latest/report.html`

## PineScript support (limited)

PineScript strategies are supported through a small translator in `src/engine/pine_adapter.py`. If a script uses unsupported syntax or indicators, you must either:
- Extend the translator in `src/engine/pine_adapter.py`, or
- Convert the logic into a Python strategy and run with `strategy.type: native` (or add a new Python strategy mode).

## Security

Never commit secrets. Place API keys (SMTP, Twelve Data) only in `.env` (see `.env.example`).
