from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import modal


def _ignore_local_path(path: Path) -> bool:
    ignored_roots = {".git", ".venv", "artifacts", "data", "reports"}
    ignored_parts = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    ignored_names = {
        ".DS_Store",
        "merge_summary.json",
        "artifacts.tar.gz",
    }
    if path.parts and path.parts[0] in ignored_roots:
        return True
    return any(part in ignored_parts for part in path.parts) or path.name in ignored_names

IMAGE = (
    modal.Image.debian_slim()
    .pip_install(
        "pandas", "numpy", "pyarrow", "yfinance", "requests", "beautifulsoup4",
        "ta", "fredapi", "praw", "pydantic", "pyyaml", "dask", "polars", "duckdb",
        "pandas-market-calendars"
    )
    .add_local_dir(".", "/app", ignore=_ignore_local_path)
)

VOLUME_NAME = "train-once-data"


def _modal_secrets() -> list[modal.Secret]:
    secrets = []
    sec_user_agent = os.environ.get("SEC_USER_AGENT")
    if sec_user_agent:
        secrets.append(modal.Secret.from_dict({"SEC_USER_AGENT": sec_user_agent}))
    tv_url = os.environ.get("TRADINGVIEW_CSV_URL")
    tv_path = os.environ.get("TRADINGVIEW_CSV_PATH")
    if tv_url or tv_path:
        secrets.append(
            modal.Secret.from_dict(
                {
                    "TRADINGVIEW_CSV_URL": tv_url or "",
                    "TRADINGVIEW_CSV_PATH": tv_path or "",
                }
            )
        )
    return secrets


def build_app(cpu: int, memory: int, gpu: str | None) -> modal.App:
    app = modal.App("train-once-corpus")
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    secrets = _modal_secrets()

    fn_kwargs = dict(
        cpu=cpu,
        memory=memory,
        timeout=60 * 60 * 6,
        image=IMAGE,
        volumes={"/data": volume},
        secrets=secrets,
        serialized=True,
    )
    if gpu:
        fn_kwargs["gpu"] = gpu

    @app.function(**fn_kwargs)
    def build(config_path: str):
        os.chdir("/app")
        os.environ["PYTHONPATH"] = "/app/src:/app"

        def _run(cmd: list[str]) -> None:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Command {cmd} failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
                )

        _run(["python", "-m", "src.cli", "build-data", "--config", config_path])
        _run(["python", "-m", "src.cli", "build-corpus", "--config", config_path])
        if not os.path.exists("/data/corpus/tabular.parquet"):
            raise RuntimeError("tabular.parquet not found after build")
        volume.commit()
        return True

    @app.function(image=IMAGE, volumes={"/data": volume}, serialized=True)
    def progress() -> dict:
        progress_path = Path("/data/processed/progress.json")
        if progress_path.exists():
            try:
                return json.loads(progress_path.read_text())
            except Exception:
                return {}
        return {}

    @app.function(image=IMAGE, volumes={"/data": volume}, serialized=True)
    def stats() -> dict:
        summary = {"tabular": None, "text": None}
        tab_path = "/data/corpus/tabular.parquet"
        text_path = "/data/corpus/text_corpus.jsonl"
        if os.path.exists(tab_path):
            import pyarrow.parquet as pq

            meta = pq.read_metadata(tab_path)
            summary["tabular"] = {
                "rows": meta.num_rows,
                "size_bytes": os.path.getsize(tab_path),
            }
        if os.path.exists(text_path):
            summary["text"] = {
                "size_bytes": os.path.getsize(text_path),
            }
        return summary

    app.build = build
    app.progress = progress
    app.stats = stats
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_large.yaml")
    parser.add_argument("--cpu", type=int, default=8)
    parser.add_argument("--memory", type=int, default=32768)
    parser.add_argument("--gpu", default="")
    parser.add_argument("--summary", default="corpus_summary.json")
    parser.add_argument("--progress", default="progress.json")
    args = parser.parse_args()

    app = build_app(args.cpu, args.memory, args.gpu)
    with app.run():
        app.build.remote(args.config)
        summary = app.stats.remote()
        progress = app.progress.remote()

    Path(args.summary).write_text(json.dumps(summary, indent=2))
    Path(args.progress).write_text(json.dumps(progress, indent=2))


if __name__ == "__main__":
    main()
