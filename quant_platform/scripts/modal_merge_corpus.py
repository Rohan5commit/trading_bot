from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal
import pandas as pd
import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
for candidate in ("/root/src", "/root", str(ROOT_DIR), str(ROOT_DIR / "src")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)


def _ignore_local_path(path: Path) -> bool:
    ignored_parts = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    ignored_names = {".DS_Store", "merge_summary.json", "artifacts.tar.gz"}
    return any(part in ignored_parts for part in path.parts) or path.name in ignored_names


IMAGE = (
    modal.Image.debian_slim()
    .pip_install(
        "pandas",
        "numpy",
        "pyarrow",
        "duckdb",
        "yfinance",
        "requests",
        "beautifulsoup4",
        "ta",
        "fredapi",
        "praw",
        "pydantic",
        "pyyaml",
        "dask",
        "polars",
        "pandas-market-calendars",
        "boto3",
    )
    .add_local_dir("src", "/root/src", ignore=_ignore_local_path)
    .add_local_dir("scripts", "/root/scripts", ignore=_ignore_local_path)
    .add_local_dir("configs", "/root/configs")
)

VOLUME_NAME = "train-once-data"
CHECKPOINT_KEYS = [
    "CHECKPOINT_S3_BUCKET",
    "CHECKPOINT_S3_ACCESS_KEY",
    "CHECKPOINT_S3_SECRET_KEY",
    "CHECKPOINT_S3_REGION",
    "CHECKPOINT_S3_ENDPOINT",
    "CHECKPOINT_S3_PREFIX",
    "CHECKPOINT_S3_USE_PATH_STYLE",
]
DEFAULT_CPU = int(os.getenv("MODAL_CPU", "16"))
DEFAULT_MEMORY_MB = int(os.getenv("MODAL_MEMORY_MB", "65536"))


def _feature_config_from(cfg: dict):
    from data.feature_store import FeatureStoreConfig

    paths = cfg.get("paths", {})
    return FeatureStoreConfig(
        raw_path=paths.get("raw", "data/raw"),
        processed_path=paths.get("processed", "data/processed"),
        calendar=cfg.get("calendar", "NYSE"),
        indicator_mode=cfg.get("indicator_mode", "expanded"),
        ohlcv_batch=cfg.get("ohlcv_batch", True),
        ohlcv_batch_size=cfg.get("ohlcv_batch_size", 50),
        hf_yahoo_enabled=cfg.get("hf_yahoo_enabled", True),
        hf_yahoo_cache_dir=cfg.get("hf_yahoo_cache_dir"),
        allow_source_fallbacks=cfg.get("allow_source_fallbacks", True),
        sec_fsds=cfg.get("sec_fsds", {}),
    )


def _sample_rows(frame: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    labeled = frame.dropna(subset=["label"]).copy()
    if not max_rows or len(labeled) <= max_rows:
        return labeled
    return labeled.sample(n=int(max_rows), random_state=seed).sort_values(["date", "ticker"])


def _write_text_rows(builder, frame: pd.DataFrame, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for _, row in frame.iterrows():
            record = {"text": builder._format_prompt(row), "label": row["label"]}
            handle.write(json.dumps(record) + "\n")
    return len(frame)


def _checkpoint_env() -> dict:
    values = {key: os.environ.get(key, "") for key in CHECKPOINT_KEYS}
    return {key: value for key, value in values.items() if value}


def _modal_secrets() -> list[modal.Secret]:
    env = _checkpoint_env()
    if env:
        return [modal.Secret.from_dict(env)]
    return []


def _checkpoint_client():
    import boto3
    from botocore.config import Config

    endpoint = os.environ.get("CHECKPOINT_S3_ENDPOINT")
    region = os.environ.get("CHECKPOINT_S3_REGION") or "us-east-1"
    access_key = os.environ.get("CHECKPOINT_S3_ACCESS_KEY")
    secret_key = os.environ.get("CHECKPOINT_S3_SECRET_KEY")
    use_path_style = os.environ.get("CHECKPOINT_S3_USE_PATH_STYLE", "").lower() in {"1", "true", "yes"}
    config = Config(s3={"addressing_style": "path"} if use_path_style else {})
    return boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
        endpoint_url=endpoint,
        config=config,
    )


def _checkpoint_prefix() -> str:
    prefix = os.environ.get("CHECKPOINT_S3_PREFIX", "train-once")
    return prefix.strip("/")


def _checkpoint_download_chunks(chunk_dir: Path) -> None:
    bucket = os.environ.get("CHECKPOINT_S3_BUCKET")
    if not bucket:
        return
    client = _checkpoint_client()
    prefix = f"{_checkpoint_prefix()}/corpus/chunks/"
    continuation = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if continuation:
            kwargs["ContinuationToken"] = continuation
        response = client.list_objects_v2(**kwargs)
        for obj in response.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".parquet"):
                continue
            local_path = chunk_dir / Path(key).name
            if local_path.exists():
                continue
            client.download_file(bucket, key, str(local_path))
        if response.get("IsTruncated"):
            continuation = response.get("NextContinuationToken")
            continue
        break


app = modal.App("train-once-corpus-merge")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    cpu=DEFAULT_CPU,
    memory=DEFAULT_MEMORY_MB,
    timeout=60 * 60 * 4,
    image=IMAGE,
    volumes={"/data": volume},
    secrets=_modal_secrets(),
)
def merge(config_path: str) -> dict:
    from data.corpus_builder import CorpusBuilder
    from data.utils import update_progress

    os.chdir("/root")
    os.environ["PYTHONPATH"] = "/root/src:/root"

    cfg = yaml.safe_load(open(config_path))
    corpus_base = cfg.get("paths", {}).get("corpus", "data/corpus")
    chunk_dir = Path(corpus_base) / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    if _checkpoint_env():
        _checkpoint_download_chunks(chunk_dir)

    chunks = sorted(chunk_dir.glob("tabular_chunk_*.parquet"))
    if not chunks:
        return {"status": "no_chunks"}

    frames = [pd.read_parquet(path) for path in chunks]
    df = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])

    builder = CorpusBuilder(_feature_config_from(cfg))
    df = builder._add_labels(df, window=cfg.get("label_window", 5))

    tab_path = Path(corpus_base) / "tabular.parquet"
    builder._write_tabular(df, str(tab_path))

    text_cfg = cfg.get("text_corpus", {})
    train_start = pd.Timestamp(text_cfg.get("train_start", "2010-01-01"))
    train_end = pd.Timestamp(text_cfg.get("train_end", "2021-12-31"))
    val_start = pd.Timestamp(text_cfg.get("val_start", "2022-01-01"))
    val_end = pd.Timestamp(text_cfg.get("val_end", "2022-12-31"))
    sample_seed = int(text_cfg.get("sample_seed", 42))

    train_rows = _sample_rows(
        df[(df["date"] >= train_start) & (df["date"] <= train_end)],
        text_cfg.get("max_train_rows"),
        sample_seed,
    )
    val_rows = _sample_rows(
        df[(df["date"] >= val_start) & (df["date"] <= val_end)],
        text_cfg.get("max_val_rows"),
        sample_seed + 1,
    )

    train_text_path = Path(text_cfg.get("train_output", Path(corpus_base) / "train_text.jsonl"))
    val_text_path = Path(text_cfg.get("val_output", Path(corpus_base) / "val_text.jsonl"))
    text_path = Path(cfg.get("corpus", {}).get("text", Path(corpus_base) / "text_corpus.jsonl"))
    all_text_rows = df.dropna(subset=["label"]).sort_values(["date", "ticker"])

    train_count = _write_text_rows(builder, train_rows, train_text_path)
    val_count = _write_text_rows(builder, val_rows, val_text_path)
    full_count = _write_text_rows(builder, all_text_rows, text_path)

    progress_path = Path(builder.feature_store.config.processed_path) / "progress.json"
    update_progress(
        progress_path,
        {
            "corpus_merge": {
                "status": "done",
                "rows": len(df),
                "chunks": len(chunks),
                "text_rows": full_count,
                "train_text_rows": train_count,
                "val_text_rows": val_count,
            }
        },
    )
    volume.commit()
    return {
        "status": "ok",
        "rows": len(df),
        "chunks": len(chunks),
        "text_rows": full_count,
        "train_text_rows": train_count,
        "val_text_rows": val_count,
    }


@app.local_entrypoint()
def main(
    config: str = "configs/data_large.yaml",
    summary: str = "merge_summary.json",
) -> None:
    result = merge.remote(config)
    Path(summary).write_text(json.dumps(result, indent=2))
