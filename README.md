# Trading Bot

This repo now contains both parts of the system in one place:

1. The daily trading bot at the repo root
2. The train-once quant model platform under `quant_platform/`

## Repo Layout

- Root: live trading bot, daily orchestration, SQLite state, email/reporting
- `quant_platform/`: corpus building, one-time GPU training, frozen LoRA adapter workflow, backtesting/research platform

## Current Architecture

```
trading_bot/
├── main.py                      # Daily core bot + AI bot orchestration
├── llm_trader.py                # AI trading branch using the trained model
├── trained_model_client.py      # Remote HTTP client for trained-model inference
├── trained_model_service_runtime.py
├── lightning_trained_model_app.py
├── deploy_lightning_inference.py
├── backtesting/                 # Existing research stack in the bot repo
└── quant_platform/              # Merged train-once quant platform repo
```

## Core vs AI

- Core bot remains unchanged in principle: price ingestion, feature generation, OLS ranking, meta-learner, portfolio logic
- AI trading bot is separate and now uses the trained quant model over HTTP
- The AI path is batched and designed to call a remote CPU inference endpoint, not a local model

## Secrets

### Still used
- `NVIDIA_API_KEY`: news sentiment path
- `TRAINED_MODEL_INFERENCE_URL`: deployed inference URL for the AI trading bot
- `TRAINED_MODEL_API_KEY`: optional auth for the trained-model endpoint
- `TWELVEDATA_API_KEYS`, `ALPHAVANTAGE_API_KEYS`: optional price providers

### No longer used by the AI trading bot
- `NVIDIA_REASONING_API_KEY`
- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`

## Main Workflows

- `.github/workflows/daily_trading_bot.yml`
  - Daily root bot workflow
  - Core + AI orchestration
- `.github/workflows/ai_trading_smoke.yml`
  - AI-only smoke test against the remote trained-model endpoint
  - Does not run the core strategy
- `.github/workflows/deploy_lightning_inference.yml`
  - Deploys the trained-model inference service to Lightning AI
  - Leaves the core bot untouched

## AI-Only Smoke Test

Manual:
```bash
python run_ai_trading_smoke.py
```

GitHub Actions:
- Actions -> **AI Trading Smoke**
- This tests only the AI trading branch and the trained-model endpoint

## Quant Platform

The full train-once quant platform has been merged into:

- [quant_platform/](./quant_platform)

That subtree contains:
- corpus builders
- training scripts
- backtest engine
- inference/API scaffolding
- configs, docs, and tests from the original train-once repo

Start there if you want to inspect the model/training system rather than the daily bot.

## Local Setup

```bash
pip install -r requirements.txt
python main.py daily_job
```

For AI-only testing:
```bash
python run_ai_trading_smoke.py
```

## Notes

- The AI bot is remote-only and expects the trained model to be served externally.
- The current deployment target is Lightning AI CPU.
- The Lightning inference app can either mount a ready adapter directory or download a `tar.gz` / `.zip` archive via `TRAINED_MODEL_ADAPTER_ARCHIVE_URL`.
- The core bot and AI bot remain logically separate even though they now live in one combined repo.

## Lightning Account

- The funded Lightning deployment account for this repo is `Rohan San`
- Username: `rohansanthoshkumar1`
- Email: `rohan.santhoshkumar1@gmail.com`
- The repo should use the GitHub secrets `LIGHTNING_USERNAME`, `LIGHTNING_API_KEY`, and `LIGHTNING_USER_ID` for that account
- Do not switch deployment to the separate `rohansanthoshkumar` account unless explicitly told to do so
