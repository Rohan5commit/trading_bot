# Operational Runbook: Advanced Daily Trading Bot

## 1. Prerequisites
- **Python 3.9+**
- **Dependencies**: `pandas`, `requests`, `pyyaml`, `yfinance`, `python-dotenv`
- **Email Configuration**: Gmail App Password required in `.env`.
- **News Sentiment (Optional)**: Local fallback scoring is built in (no external LLM key required).
- **AI Trading Bot**:
  - preferred path: remote trained-model inference endpoint in `.env` or GitHub secrets
  - fallback path: local distilled manager if the full runtime is unavailable

## 2. Setup
1. Move the `trading_bot` folder to your desired location (e.g., home folder).
2. `pip install -r requirements.txt`.
3. Fill in your email credentials in `.env`.
4. Create `.env` from the template and paste your keys:
   ```bash
   cp .env.example .env
   ```
5. AI trading endpoint:
   - `TRAINED_MODEL_INFERENCE_URL` points the AI trading bot at the hosted trained-model service.
   - `TRAINED_MODEL_API_KEY` optionally protects that endpoint.
   - `TWELVEDATA_API_KEYS` should come from env or GitHub secrets only.
   - Do not commit `.env` (it is gitignored).

## 3. Daily Workflow
The bot is fully automated:
1. **Start Scheduler**: `python3 scheduler.py`
2. **Automated Cycle**:
   - Checks for Wi-Fi and 8 AM start time.
   - Runs `main.py` (Price/News Ingest -> Features -> Train -> Backtest).
   - Sends Daily Email Report (combined pipeline + backtest details).
   - Updates Meta-Learner state.

## 4. Long-Range Backtesting (Optional)
Use the backtesting engine for historical runs and PineScript-ready strategy configuration.
- Runner: `python3 backtesting_runner.py backtest`
- Config: `backtesting/config.yaml`
- Dependencies: `backtesting/requirements.txt`

Note: PineScript translation is a placeholder; set `strategy.type: pine` with a script path when ready.

## 4. Safety & Persistence
- **State**: Positions are tracked in `data/trading_bot.db`.
- **Learning**: Adaptations are saved in `meta_learner_state.json`.
- **Logs**: View progress in `scheduler.log`.

## 5. Troubleshooting
- **No Email**: Verify `SENDER_EMAIL` and `SENDER_PASSWORD` in `.env`.
- **No Data**: Ensure internet connection is active (Wi-Fi check).
- **AI Strategy Not Trading**:
  - If the trained-model endpoint call fails, the workflow should retry via the distilled fallback path.
  - Check `AI Backend Selected` and `AI Router Reason` in the AI email/report.
  - Check that `TRAINED_MODEL_INFERENCE_URL` is set and the hosted service is healthy when the full runtime is selected.
