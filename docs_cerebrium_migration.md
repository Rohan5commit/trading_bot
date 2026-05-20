# Cerebrium AI inference cutover

This repo now treats Cerebrium as the primary full-model AI inference runtime while keeping the existing trading SQLite database as the only source of truth for positions, AI memory, and trade decisions.

## What must not move

- Do not copy or reset `data/trading_bot.db` into Cerebrium.
- Do not reset `positions_ai`, `ai_manager_runs`, or `ai_manager_decisions` during deployment.
- Do not change Core bot settings when changing Cerebrium deployment settings.
- Do not retrain or replace the model adapter for this migration.

## Cerebrium app

The Cerebrium app is staged from `cerebrium_app/` by `deploy_cerebrium_inference.py` and reuses `trained_model_service_runtime.py`. The deployed app keeps the same HTTP contract:

- `GET /health`
- `POST /warmup`
- `POST /predict_trade_candidates`

Expected app base URL:

```text
https://api.aws.us-east-1.cerebrium.ai/v4/p-58018090/trading-bot-ai
```

Set the GitHub secret `CEREBRIUM_INFERENCE_URL` to that base URL or to its `/predict_trade_candidates` URL. The client will derive `/health` automatically.

## Required GitHub secrets

Do not commit access keys. Add these as GitHub repository/environment secrets:

- `CEREBRIUM_SERVICE_ACCOUNT_TOKEN` — deploy workflow authentication.
- `CEREBRIUM_PROJECT_ID` — optional; defaults to `p-58018090`.
- `CEREBRIUM_INFERENCE_URL` — daily workflow primary endpoint.
- `CEREBRIUM_API_KEY` — request auth for the app; can match `TRAINED_MODEL_API_KEY` while testing.
- `TRAINED_MODEL_API_KEY` — existing service auth fallback.
- `TRAINED_MODEL_ADAPTER_ARCHIVE_URL` — exact existing adapter archive.
- `TRAINED_MODEL_ADAPTER_ARCHIVE_TOKEN` — adapter archive token if needed.

## Required Cerebrium dashboard secrets

Add these in the Cerebrium dashboard so they are available as runtime environment variables inside the app:

- `TRAINED_MODEL_BASE_MODEL=Qwen/Qwen2.5-7B-Instruct`
- `TRAINED_MODEL_NAME=quant-trained-trading-model`
- `TRAINED_MODEL_ADAPTER_ARCHIVE_URL=<same archive used by Lightning>`
- `TRAINED_MODEL_ADAPTER_ARCHIVE_TOKEN=<same token used by Lightning, if needed>`
- `TRAINED_MODEL_API_KEY=<same auth value used by the daily bot>`
- `TRAINED_MODEL_CLASS_TOKEN_INFERENCE=1`

The app is configured with `min_replicas = 0`, so it is not always on. The daily workflow wakes it by running `wait_for_trained_model.py` and `warm_trained_model.py` before the AI-only Cerebrium daily run.

## Fallback behavior

1. Daily GitHub Actions restores the existing `data/`, `models/`, and registry cache.
2. Core bot runs independently with `DISABLE_AI_TRADING=1`.
3. Direct runtime path is `python run_ai_daily_cerebrium.py`, which runs AI-only execution on Cerebrium first and writes `results/ai_runtime_plan.json`.
4. If the Cerebrium daily job fails and `LIGHTNING_INFERENCE_URL` is configured, the direct runner retries once through Lightning HTTP as a fallback.
5. GitHub scheduled AI execution is disabled; manual workflow dispatch remains available for diagnostics.

AI manager context stays in SQLite and is sent to the remote model as compact `manager_context` in prediction payloads. Remote services do not write to production SQLite.
