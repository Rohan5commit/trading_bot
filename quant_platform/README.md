# Train-Once Quant Trading Platform

One-time GPU training pipeline that produces a frozen LoRA adapter and deploys that same artifact everywhere: backtests, paper trading, live bot, and the research API. No retraining unless explicitly triggered.

## Architecture (Text Diagram)

```
          +-----------------------------+
          |  Data Ingestion + FeatureStore|
          |  (OHLCV, fundamentals, news,  |
          |   transcripts, macro, social) |
          +------------------+------------+
                             |
                             v
                 +------------------------+
                 | Training Corpus Builder|
                 | - Tabular (parquet)    |
                 | - Text (jsonl prompts) |
                 +-----------+------------+
                             |
             +---------------+---------------+
             |                               |
             v                               v
    +----------------------+        +----------------------+
    | Tabular Model (CPU) |        | LoRA LLM (GPU)       |
    | XGBoost + Optuna     |        | L40S or A100          |
    +-----------+----------+        +-----------+----------+
                |                               |
                +---------------+---------------+
                                v
                     +----------------------+
                     | Ensemble Scoring     |
                     +----------+-----------+
                                |
          +---------------------+---------------------+
          |                     |                     |
          v                     v                     v
    Backtester (CPU)      Paper Bot (CPU)      FastAPI (CPU)
```

## Key Principles

- Train ONCE on GPU, save a LoRA adapter artifact.
- All downstream uses load the same frozen adapter and tabular model.
- No GPU usage after training. Inference uses CPU or NVIDIA NIM.
- Backtests are the source of truth. No alpha claims without metrics.

## Data Sources (Free/Low-Cost First)

- OHLCV: `yfinance` (daily + limited intraday), optional Polygon/Alpha Vantage.
- Fundamentals: `yfinance` metadata + SEC Financial Statement Data Sets (FSDS).
- FSDS: bulk quarterly 10-Q/10-K numeric data from SEC (10GB+ when spanning 2010-2024).
- News: NewsAPI (free tier), Yahoo RSS.
- Transcripts: local JSON or SEC 8-K ingestion (stubbed).
- Macro: FRED API + `yfinance` (VIX, DXY, gold, crude).
- Sentiment: placeholders for Reddit/Twitter/short interest.

## GPU Cost Estimate

- Default GPU: **NVIDIA L40S** (lower cost, strong throughput).
- Optional: **A100-40GB** if you want extra headroom.
- Training budget target: 4 hours.

## Repo Structure

```
src/
  data/
  training/
  model/
  backtest/
  bot/
  api/
  monitoring/
configs/
artifacts/
data/
reports/
.github/workflows/
```

## Setup (GitHub Actions)

1. Create repo secrets:
   - `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`
   - `NVIDIA_NIM_API_KEY`
   - `SEC_USER_AGENT` (required for SEC downloads)
   - Optional: `NEWSAPI_KEY`, `FRED_API_KEY`, `WANDB_API_KEY`
2. Trigger training workflow:
   - GitHub Actions → **Train LoRA (Modal GPU)**.
2.5. The training workflow uploads `artifacts.tar.gz`. Extract it into `artifacts/` before CPU runs.
3. Run CPU backtest:
   - GitHub Actions → **Backtest CPU**.

## One Worked Example (AAPL, MSFT, NVDA, TSLA, SPY)

The default configs already target these tickers for 2023-2024 test period.

```
python -m src.cli build-data --config configs/data.yaml
python -m src.cli build-corpus --config configs/data.yaml
python -m src.training.train_lora --config configs/training.yaml
python -m src.backtest.engine --config configs/backtest.yaml
```

In GitHub Actions, run **Backtest CPU** after training completes.

## Known Limitations

- Free APIs have rate limits and partial history for intraday data.
- Full multimodal corpus at scale needs substantial CPU memory for preprocessing.
- Transcript/news ingestion is stubbed unless you provide sources.
- NIM inference requires a valid API key and model access.

## What Runs Where

- GPU: `src/training/train_lora.py` via Modal (L40S or A100).
- CPU: feature engineering, corpus build, backtests, bot, API.

## Artifacts

- `artifacts/tabular_model.ubj`
- `artifacts/lora_adapter/`
- `artifacts/training_metadata.json`
- `reports/backtest/*`

## TradingView Data

TradingView proprietary data is not publicly accessible. The pipeline includes TradingView-equivalent indicators and can ingest TradingView CSV exports if you provide `TRADINGVIEW_CSV_PATH` or `TRADINGVIEW_CSV_URL` as a secret/environment variable.

## Build Large Corpus

Use the `Build Large Corpus (Modal CPU)` workflow to generate a large dataset in the Modal volume. It outputs a small `corpus_summary.json` artifact with row counts and sizes.

## Lightning Auto-Resume Runs

If you want Lightning.ai to survive interruptions without trying to chain free interactive sessions forever, use the included Lightning run workflow:

- Configure [lightning_run.yaml](/Users/rohan/train-once-quant-platform/configs/lightning_run.yaml)
- Add GitHub secrets:
  - `LIGHTNING_USERNAME`
  - `LIGHTNING_API_KEY`
- Launch **Launch Lightning Auto-Resume Run**
- Let **Lightning Progress Snapshot** archive status and checkpoint manifests every 4 hours

Details: [lightning_autoresume.md](/Users/rohan/train-once-quant-platform/docs/lightning_autoresume.md)

## GitHub CPU Chunking (No Modal CPU)

If you want to avoid Modal for CPU, use the **Build Corpus Chunk (GitHub CPU)** workflow. It writes each chunk to external S3-compatible storage (OCI Object Storage works) and is limited by GitHub’s 6-hour runner cap, so keep `chunk_size` small.

## OCI CPU (Time-Boxed VM)

If you prefer OCI CPU, use the **Launch OCI CPU VM (Time-Boxed)** workflow. It launches a VM with a strict auto-shutdown window and provides an instance id artifact. Terminate any VM with **Terminate OCI VM**.

Required GitHub secrets:
- `OCI_TENANCY_OCID`
- `OCI_USER_OCID`
- `OCI_FINGERPRINT`
- `OCI_REGION`
- `OCI_PRIVATE_KEY`
- `OCI_AD`
- `OCI_COMPARTMENT_OCID`
- `OCI_SUBNET_OCID`
- `OCI_IMAGE_OCID`

## External Checkpointing (Switch Modal Accounts Safely)

If you might switch Modal accounts mid-run, enable external checkpointing to S3-compatible storage. Each chunk uploads to a bucket so a new Modal account can continue and the merge step can download chunks.

Add these GitHub Secrets (optional):
- `CHECKPOINT_S3_BUCKET`
- `CHECKPOINT_S3_ACCESS_KEY`
- `CHECKPOINT_S3_SECRET_KEY`
- `CHECKPOINT_S3_REGION` (default `us-east-1` if omitted)
- `CHECKPOINT_S3_ENDPOINT` (for R2/MinIO)
- `CHECKPOINT_S3_PREFIX` (default `train-once`)
- `CHECKPOINT_S3_USE_PATH_STYLE` (`true` for MinIO)

## Open-Source Pine Script References

See `docs/pine_sources.md` for open-source Pine Script indicator references and licenses.
