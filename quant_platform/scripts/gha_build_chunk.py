from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import boto3
from botocore.config import Config
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.feature_store import FeatureStore, FeatureStoreConfig
from data.corpus_builder import CorpusBuilder, CorpusConfig
from data.universe import get_us_equity_universe
from data.utils import ensure_dir, update_progress


CHECKPOINT_KEYS = [
    "CHECKPOINT_S3_BUCKET",
    "CHECKPOINT_S3_ACCESS_KEY",
    "CHECKPOINT_S3_SECRET_KEY",
    "CHECKPOINT_S3_REGION",
    "CHECKPOINT_S3_ENDPOINT",
    "CHECKPOINT_S3_PREFIX",
    "CHECKPOINT_S3_USE_PATH_STYLE",
]


def _checkpoint_env() -> dict:
    values = {key: os.environ.get(key, "") for key in CHECKPOINT_KEYS}
    return {key: value for key, value in values.items() if value}


def _checkpoint_client():
    endpoint = os.environ.get("CHECKPOINT_S3_ENDPOINT")
    region = os.environ.get("CHECKPOINT_S3_REGION") or "us-east-1"
    access_key = os.environ.get("CHECKPOINT_S3_ACCESS_KEY")
    secret_key = os.environ.get("CHECKPOINT_S3_SECRET_KEY")
    use_path_style = os.environ.get("CHECKPOINT_S3_USE_PATH_STYLE", "").lower() in {"1", "true", "yes"}
    s3_config = {"payload_signing_enabled": False}
    if use_path_style:
        s3_config["addressing_style"] = "path"
    config = Config(s3=s3_config)
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


def _checkpoint_upload(bucket: str, local_path: Path, object_key: str) -> None:
    client = _checkpoint_client()
    size = local_path.stat().st_size
    with local_path.open("rb") as handle:
        client.put_object(Bucket=bucket, Key=object_key, Body=handle, ContentLength=size)


def _resolve_universe(cfg: dict) -> list[str]:
    if cfg.get("universe"):
        return cfg["universe"]
    source = cfg.get("universe_source", "nasdaq_all")
    hf_cache_dir = cfg.get("hf_yahoo_cache_dir")
    if hf_cache_dir and hf_cache_dir.startswith("/data"):
        hf_cache_dir = str((ROOT_DIR / "data" / hf_cache_dir.lstrip("/")).resolve())
    if source in {"nasdaq_all", "sec_tickers", "hf_liquid"}:
        return get_us_equity_universe(
            max_tickers=cfg.get("max_tickers"),
            source=source,
            hf_cache_dir=hf_cache_dir,
        )
    return []


def _localize_sec_fsds(cfg: dict, data_root: Path) -> dict:
    sec_fsds = dict(cfg.get("sec_fsds", {}) or {})
    raw_dir = sec_fsds.get("raw_dir")
    processed_dir = sec_fsds.get("processed_dir")
    if raw_dir and raw_dir.startswith("/data"):
        sec_fsds["raw_dir"] = str(data_root / raw_dir.lstrip("/"))
    if processed_dir and processed_dir.startswith("/data"):
        sec_fsds["processed_dir"] = str(data_root / processed_dir.lstrip("/"))
    return sec_fsds


def _localize_path(path: str | None, data_root: Path) -> str | None:
    if not path:
        return path
    if path.startswith("/data"):
        return str(data_root / path.lstrip("/"))
    return path


def _feature_config_from(cfg: dict, data_root: Path) -> FeatureStoreConfig:
    return FeatureStoreConfig(
        raw_path=str(data_root / "raw"),
        processed_path=str(data_root / "processed"),
        calendar=cfg.get("calendar", "NYSE"),
        indicator_mode=cfg.get("indicator_mode", "expanded"),
        ohlcv_batch=cfg.get("ohlcv_batch", True),
        ohlcv_batch_size=cfg.get("ohlcv_batch_size", 50),
        hf_yahoo_enabled=cfg.get("hf_yahoo_enabled", True),
        hf_yahoo_cache_dir=_localize_path(cfg.get("hf_yahoo_cache_dir"), data_root),
        allow_source_fallbacks=cfg.get("allow_source_fallbacks", True),
        sec_fsds=_localize_sec_fsds(cfg, data_root),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_large.yaml")
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--start-idx", type=int)
    parser.add_argument("--end-idx", type=int)
    parser.add_argument("--summary", default="chunk_summary.json")
    args = parser.parse_args()

    checkpoint_disabled = os.environ.get("CHECKPOINT_DISABLE", "").lower() in {"1", "true", "yes"}
    checkpoint = {} if checkpoint_disabled else _checkpoint_env()
    if not checkpoint_disabled and (not checkpoint or not checkpoint.get("CHECKPOINT_S3_BUCKET")):
        raise RuntimeError("CHECKPOINT_S3_BUCKET is required for GitHub CPU runs.")

    cfg = yaml.safe_load(Path(args.config).read_text())
    universe = _resolve_universe(cfg)
    start_idx = args.start_idx if args.start_idx is not None else args.chunk_id * args.chunk_size
    end_idx = args.end_idx if args.end_idx is not None else start_idx + args.chunk_size
    chunk = universe[start_idx:end_idx]
    if not chunk:
        Path(args.summary).write_text(json.dumps({"status": "empty", "chunk_id": args.chunk_id}, indent=2))
        return

    data_root = ROOT_DIR / "data"
    feature_cfg = _feature_config_from(cfg, data_root)
    fs = FeatureStore(feature_cfg)

    progress_path = Path(feature_cfg.processed_path) / "progress.json"
    update_progress(
        progress_path,
        {
            "corpus_chunks": {
                str(args.chunk_id): {
                    "status": "running",
                    "chunk_size": len(chunk),
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                }
            }
        },
    )

    features = fs.build_feature_frame_chunk(
        chunk,
        cfg["start"],
        cfg["end"],
        chunk_id=args.chunk_id,
        normalize=True,
        refresh=cfg.get("feature_refresh", False),
        output_dir=feature_cfg.processed_path,
    )
    if features.empty:
        update_progress(
            progress_path,
            {"corpus_chunks": {str(args.chunk_id): {"status": "no_features"}}},
        )
        Path(args.summary).write_text(json.dumps({"status": "no_features", "chunk_id": args.chunk_id}, indent=2))
        return

    corpus_root = data_root / "corpus"
    chunk_dir = corpus_root / "chunks"
    ensure_dir(chunk_dir)
    tabular_path = chunk_dir / f"tabular_chunk_{args.chunk_id}.parquet"

    builder = CorpusBuilder(feature_cfg)
    builder.build_from_frame(
        features,
        CorpusConfig(
            start=cfg["start"],
            end=cfg["end"],
            output_tabular=str(tabular_path),
            output_text=str(chunk_dir / f"text_chunk_{args.chunk_id}.jsonl"),
            generate_labels=False,
            generate_text=False,
        ),
    )

    update_progress(
        progress_path,
        {"corpus_chunks": {str(args.chunk_id): {"status": "done", "rows": len(features)}}},
    )

    uploads: list[str] = []
    if not checkpoint_disabled:
        bucket = checkpoint["CHECKPOINT_S3_BUCKET"]
        prefix = _checkpoint_prefix()
        processed_path = Path(feature_cfg.processed_path)
        raw_path = processed_path / "chunks" / f"features_raw_chunk_{args.chunk_id}.parquet"
        norm_path = processed_path / "chunks" / f"features_norm_chunk_{args.chunk_id}.parquet"

        uploads = [
            (raw_path, f"{prefix}/features/raw_chunk_{args.chunk_id}.parquet"),
            (norm_path, f"{prefix}/features/norm_chunk_{args.chunk_id}.parquet"),
            (tabular_path, f"{prefix}/corpus/chunks/tabular_chunk_{args.chunk_id}.parquet"),
            (progress_path, f"{prefix}/progress/progress.json"),
        ]
        for local_path, key in uploads:
            if local_path.exists():
                _checkpoint_upload(bucket, local_path, key)

    summary = {
        "status": "ok",
        "chunk_id": args.chunk_id,
        "tickers": len(chunk),
        "rows": len(features),
        "uploads": [key for _, key in uploads] if uploads else [],
    }
    summary_path = Path(args.summary)
    summary_path.write_text(json.dumps(summary, indent=2))
    if not checkpoint_disabled:
        _checkpoint_upload(bucket, summary_path, f"{prefix}/reports/chunk_summary_{args.chunk_id}.json")


if __name__ == "__main__":
    main()
