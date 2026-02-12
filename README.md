# Daily Trading Bot

This is a Python-based trading bot designed to run daily, ingest market data, generate signals, and execute trades (simulated or real).

## Setup

### Prerequisites
- Python 3.10+
- `pip`

### Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```bash
   cp .env.example .env
   ```
4. Open `.env` and paste your keys (do not commit `.env`).

## Usage

### Local Run
To run the daily job manually:
```bash
python main.py daily_job
```

### Full Pipeline
To run the full pipeline (ingest, train, backtest):
```bash
python main.py full
```

## API Keys (Step-by-Step)

This repo is set up so secrets are **not** stored in git. A fresh `git clone` will **not** include your API keys.

### A) Run Locally (Recommended)
1. Clone:
   ```bash
   git clone https://github.com/Rohan5commit/trading_bot.git
   cd trading_bot
   ```
2. Create `.env`:
   ```bash
   cp .env.example .env
   ```
3. Open `.env` and paste your keys:
   - In VS Code: `code .` then click `.env` and paste values.
   - In Terminal: `nano .env` then paste values and save.
4. Required env vars for the current setup:
   - `NVIDIA_API_KEY` (news/LLM sentiment)
   - `NVIDIA_REASONING_API_KEY` (AI strategy trade selection)
   - Email: `SMTP_SERVER`, `SMTP_PORT`, `SENDER_EMAIL`, `SENDER_PASSWORD`, `RECIPIENT_EMAIL`
   - Optional (faster S&P500 ingestion): `TWELVEDATA_API_KEYS` (comma-separated)
5. Run:
   ```bash
   python3 main.py daily_job
   ```

### B) Run In GitHub Actions (Cloud)
1. Go to your repo on GitHub.
2. `Settings` -> `Secrets and variables` -> `Actions`.
3. Click `New repository secret`.
4. Add secrets (names must match exactly):
   - `NVIDIA_API_KEY`
   - `NVIDIA_API_KEY_ID` (optional)
   - `NVIDIA_REASONING_API_KEY`
   - `NVIDIA_REASONING_API_KEY_ID` (optional)
   - `TWELVEDATA_API_KEYS` (optional, comma-separated)
   - Email secrets as used by your deployment (see your workflow / `send_email_report.py` configuration).

## State Persistence (Important)
Positions and account state live in SQLite at `data/trading_bot.db`.
- GitHub Actions keeps this via cache for cloud runs.
- A new local clone starts "fresh" unless you restore a saved copy of `data/trading_bot.db`.

## GitHub Actions Deployment

This repository includes a GitHub Actions workflow (`.github/workflows/daily_trading_bot.yml`) that runs the bot daily at 9:30 AM EST (Mon-Fri).

### Configuration
To execute successfully on GitHub Actions, you must configure the following **Repository Secrets** in your GitHub repository (Settings -> Secrets and variables -> Actions):

| Secret Name | Description |
|---|---|
| `TWELVEDATA_API_KEYS` | Comma-separated API keys for TwelveData (if used). |
| `ALPHAVANTAGE_API_KEYS` | Comma-separated API keys for AlphaVantage (if used). |
| `MAILGUN_API_KEY` | API Key for Mailgun (for email reports). |
| `MAILGUN_DOMAIN` | Mailgun domain (e.g., `mg.yourdomain.com`). |
| `EMAIL_RECIPIENTS` | Comma-separated list of email addresses to receive reports. |
| `EMAIL_SENDER` | Email address to send from (e.g., `bot@yourdomain.com`). |
| `OPENAI_API_KEY` | OpenAI API Key (for LLM analysis). |

### Workflows
- **Daily Trading Bot**: Triggers on schedule (Mon-Fri) and can be manually triggered via the "Run workflow" button in the Actions tab.
