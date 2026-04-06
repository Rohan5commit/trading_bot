from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

import modal
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
for candidate in ("/app/src", "/app", str(ROOT_DIR), str(ROOT_DIR / "src")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from data.feature_store import FeatureStore, FeatureStoreConfig
from data.corpus_builder import CorpusBuilder, CorpusConfig
from data.universe import get_us_equity_universe
from data.utils import ensure_dir, update_progress


def _ignore_local_path(path: Path) -> bool:
    ignored_parts = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    ignored_names = {".DS_Store"}
    return any(part in ignored_parts for part in path.parts) or path.name in ignored_names


IMAGE = (
    modal.Image.debian_slim()
    .pip_install(
        "pandas", "numpy", "pyarrow", "yfinance", "requests", "beautifulsoup4",
        "ta", "fredapi", "praw", "pydantic", "pyyaml", "dask", "polars", "duckdb",
        "pandas-market-calendars", "boto3"
    )
    .add_local_dir("src", "/app/src", ignore=_ignore_local_path)
    .add_local_dir("scripts", "/app/scripts", ignore=_ignore_local_path)
    .add_local_dir("configs", "/app/configs")
)

VOLUME_NAME = "train-once-data"
DEFAULT_CPU = int(os.getenv("MODAL_CPU", "16"))
DEFAULT_MEMORY_MB = int(os.getenv("MODAL_MEMORY_MB", "65536"))

CHECKPOINT_KEYS = [
    "CHECKPOINT_S3_BUCKET",
    "CHECKPOINT_S3_ACCESS_KEY",
    "CHECKPOINT_S3_SECRET_KEY",
    "CHECKPOINT_S3_REGION",
    "CHECKPOINT_S3_ENDPOINT",
    "CHECKPOINT_S3_PREFIX",
    "CHECKPOINT_S3_USE_PATH_STYLE",
]


def _feature_config_from(cfg: dict) -> FeatureStoreConfig:
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


def _resolve_universe(cfg: dict) -> list[str]:
    if cfg.get("universe"):
        return cfg["universe"]
    source = cfg.get("universe_source", "nasdaq_all")
    if source in {"nasdaq_all", "sec_tickers", "hf_liquid"}:
        return get_us_equity_universe(
            max_tickers=cfg.get("max_tickers"),
            source=source,
            hf_cache_dir=cfg.get("hf_yahoo_cache_dir"),
        )
    return []


def _checkpoint_env() -> dict:
    values = {key: os.environ.get(key, "") for key in CHECKPOINT_KEYS}
    return {key: value for key, value in values.items() if value}


def _modal_secrets() -> list[modal.Secret]:
    secrets = []
    env = _checkpoint_env()
    if os.environ.get("SEC_USER_AGENT"):
        env["SEC_USER_AGENT"] = os.environ["SEC_USER_AGENT"]
    if env:
        secrets.append(modal.Secret.from_dict(env))
    return secrets


app = modal.App("train-once-corpus-chunk")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pipeline_manifest_path(cfg: dict) -> Path:
    processed = cfg.get("paths", {}).get("processed", "data/processed")
    return Path(processed) / "pipeline_manifest.json"


def _load_pipeline_manifest(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {"status": "initialized", "started_at": _now(), "updated_at": _now(), "chunks": {}}


def _save_pipeline_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = _now()
    path.write_text(json.dumps(manifest, indent=2))
    volume.commit()


def _load_config(config_path: str) -> dict:
    candidates = [
        Path(config_path),
        Path("/app") / config_path,
        Path("/app/configs") / Path(config_path).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            with candidate.open() as handle:
                return yaml.safe_load(handle)
    raise FileNotFoundError(config_path)


def _build_chunk_impl(cfg: dict, chunk_id: int, chunk_size: int) -> dict:
    os.chdir("/app")
    os.environ["PYTHONPATH"] = "/app/src:/app"

    universe = _resolve_universe(cfg)
    start_idx = chunk_id * chunk_size
    end_idx = start_idx + chunk_size
    chunk = universe[start_idx:end_idx]
    if not chunk:
        return {"status": "empty", "chunk_id": chunk_id}

    feature_cfg = _feature_config_from(cfg)
    fs = FeatureStore(feature_cfg)
    progress_path = Path(feature_cfg.processed_path) / "progress.json"
    update_progress(
        progress_path,
        {
            "corpus_chunks": {
                str(chunk_id): {
                    "status": "running",
                    "chunk_size": len(chunk),
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                }
            }
        },
    )
    volume.commit()

    features = fs.build_feature_frame_chunk(
        chunk,
        cfg["start"],
        cfg["end"],
        chunk_id=chunk_id,
        normalize=True,
        refresh=cfg.get("feature_refresh", False),
    )
    if features.empty:
        update_progress(
            progress_path,
            {"corpus_chunks": {str(chunk_id): {"status": "no_features"}}},
        )
        volume.commit()
        return {"status": "no_features", "chunk_id": chunk_id}

    volume.commit()

    corpus_base = cfg.get("paths", {}).get("corpus", "data/corpus")
    chunk_dir = Path(corpus_base) / "chunks"
    ensure_dir(chunk_dir)
    tabular_path = chunk_dir / f"tabular_chunk_{chunk_id}.parquet"
    text_path = chunk_dir / f"text_chunk_{chunk_id}.jsonl"

    if tabular_path.exists() and not cfg.get("corpus_refresh", False):
        update_progress(
            progress_path,
            {"corpus_chunks": {str(chunk_id): {"status": "exists"}}},
        )
        volume.commit()
        return {"status": "exists", "chunk_id": chunk_id, "tickers": len(chunk)}

    builder = CorpusBuilder(feature_cfg)
    builder.build_from_frame(
        features,
        CorpusConfig(
            start=cfg["start"],
            end=cfg["end"],
            output_tabular=str(tabular_path),
            output_text=str(text_path),
            generate_labels=False,
            generate_text=False,
        ),
    )

    checkpoint = _checkpoint_env()
    if checkpoint:
        processed_path = Path(feature_cfg.processed_path)
        raw_path = processed_path / "chunks" / f"features_raw_chunk_{chunk_id}.parquet"
        norm_path = processed_path / "chunks" / f"features_norm_chunk_{chunk_id}.parquet"
        _checkpoint_uploads(
            chunk_id=chunk_id,
            tabular_path=tabular_path,
            raw_path=raw_path,
            norm_path=norm_path,
            progress_path=progress_path,
        )

    update_progress(
        progress_path,
        {"corpus_chunks": {str(chunk_id): {"status": "done", "rows": len(features)}}},
    )
    volume.commit()
    return {"status": "ok", "chunk_id": chunk_id, "tickers": len(chunk), "rows": int(len(features))}


@app.function(
    cpu=DEFAULT_CPU,
    memory=DEFAULT_MEMORY_MB,
    timeout=60 * 60 * 4,
    image=IMAGE,
    volumes={"/data": volume},
    secrets=_modal_secrets(),
)
def build_chunk(config_path: str, chunk_id: int, chunk_size: int) -> dict:
    cfg = _load_config(config_path)
    return _build_chunk_impl(cfg, chunk_id, chunk_size)


@app.function(
    cpu=DEFAULT_CPU,
    memory=DEFAULT_MEMORY_MB,
    timeout=60 * 60 * 24,
    image=IMAGE,
    volumes={"/data": volume},
    secrets=_modal_secrets(),
)
def run_pipeline(config_path: str, chunk_size: int = 100, start_chunk: int = 0, end_chunk: int = -1) -> dict:
    os.chdir("/app")
    os.environ["PYTHONPATH"] = "/app/src:/app"

    cfg = _load_config(config_path)
    universe = _resolve_universe(cfg)
    total_chunks = (len(universe) + chunk_size - 1) // chunk_size if universe else 0
    final_chunk = total_chunks - 1 if end_chunk < 0 else min(end_chunk, total_chunks - 1)

    manifest_path = _pipeline_manifest_path(cfg)
    manifest = _load_pipeline_manifest(manifest_path)
    manifest["status"] = "running"
    manifest["config_path"] = config_path
    manifest["chunk_size"] = chunk_size
    manifest["total_tickers"] = len(universe)
    manifest["total_chunks"] = total_chunks
    _save_pipeline_manifest(manifest_path, manifest)

    for chunk_id in range(start_chunk, final_chunk + 1):
        existing = manifest.get("chunks", {}).get(str(chunk_id), {})
        if existing.get("status") in {"ok", "exists", "no_features", "empty"}:
            continue
        manifest.setdefault("chunks", {})[str(chunk_id)] = {
            "status": "running",
            "started_at": _now(),
        }
        _save_pipeline_manifest(manifest_path, manifest)
        try:
            result = _build_chunk_impl(cfg, chunk_id, chunk_size)
        except Exception as exc:
            manifest["chunks"][str(chunk_id)] = {
                "status": "failed",
                "chunk_id": chunk_id,
                "failed_at": _now(),
                "error": repr(exc),
            }
            manifest["status"] = "failed"
            _save_pipeline_manifest(manifest_path, manifest)
            raise
        result["completed_at"] = _now()
        manifest["chunks"][str(chunk_id)] = result
        _save_pipeline_manifest(manifest_path, manifest)

    manifest["status"] = "done"
    _save_pipeline_manifest(manifest_path, manifest)
    return manifest


def _checkpoint_client():
    import boto3
    from botocore.config import Config

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


def _checkpoint_uploads(
    chunk_id: int,
    tabular_path: Path,
    raw_path: Path,
    norm_path: Path,
    progress_path: Path,
) -> None:
    bucket = os.environ.get("CHECKPOINT_S3_BUCKET")
    if not bucket:
        return
    client = _checkpoint_client()
    prefix = _checkpoint_prefix()
    uploads = [
        (raw_path, f"{prefix}/features/raw_chunk_{chunk_id}.parquet"),
        (norm_path, f"{prefix}/features/norm_chunk_{chunk_id}.parquet"),
        (tabular_path, f"{prefix}/corpus/chunks/tabular_chunk_{chunk_id}.parquet"),
        (progress_path, f"{prefix}/progress/progress.json"),
    ]
    for local_path, object_key in uploads:
        if local_path.exists():
            size = local_path.stat().st_size
            with local_path.open("rb") as handle:
                client.put_object(Bucket=bucket, Key=object_key, Body=handle, ContentLength=size)


@app.local_entrypoint()
def main(
    config: str = "configs/data_large.yaml",
    chunk_id: int = 0,
    chunk_size: int = 200,
    summary: str = "chunk_summary.json",
) -> None:
    result = build_chunk.remote(config, chunk_id, chunk_size)
    Path(summary).write_text(json.dumps(result, indent=2))
