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
3. Create a `.env` file with your API keys (see `env.example` if available, or Configuration section below).

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
