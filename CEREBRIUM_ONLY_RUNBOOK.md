# Cerebrium-Only AI Daily Run (No GitHub Actions)

This runbook removes GitHub Actions from the AI execution path and runs AI through Cerebrium directly.

## 1) Deploy/refresh Cerebrium app

```bash
export CEREBRIUM_SERVICE_ACCOUNT_TOKEN="<token>"
python deploy_cerebrium_inference.py
```

## 2) Set required runtime secrets in Cerebrium app

- `TRAINED_MODEL_API_KEY`
- `TRAINED_MODEL_ADAPTER_ARCHIVE_URL`
- `TRAINED_MODEL_ADAPTER_ARCHIVE_TOKEN` (if required)
- `TWELVEDATA_API_KEYS`
- SMTP secrets if email must be sent from runtime

## 3) Execute AI daily run directly (outside GitHub Actions)

Run from your runtime host/orchestrator:

```bash
export CEREBRIUM_TRAINED_MODEL_URL="<https://api.../predict_trade_candidates>"
python run_ai_daily_cerebrium.py
```

Optional Lightning fallback:

```bash
export LIGHTNING_INFERENCE_URL="<http(s)://.../predict_trade_candidates>"
python run_ai_daily_cerebrium.py
```

## 4) Verify result artifact

The runner writes:

- `results/ai_runtime_plan.json`

Check:
- `selected_backend`
- `resolved_inference_url`
- `ok`
- `fallback_used`

## 5) GitHub Actions status

The daily workflow now marks AI as not expected (`AI_EXPECTED=false`) so scheduled CI no longer controls AI success/failure status.
