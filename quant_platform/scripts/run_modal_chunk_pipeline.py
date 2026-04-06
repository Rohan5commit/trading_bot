from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
for candidate in (str(ROOT_DIR), str(ROOT_DIR / "src")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from data.universe import get_us_equity_universe


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_universe(cfg: dict) -> list[str]:
    if cfg.get("universe"):
        return list(cfg["universe"])
    source = cfg.get("universe_source", "nasdaq_all")
    if source in {"nasdaq_all", "sec_tickers"}:
        return get_us_equity_universe(max_tickers=cfg.get("max_tickers"), source=source)
    return []


def _load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {
        "status": "initialized",
        "started_at": _now(),
        "updated_at": _now(),
        "chunks": {},
    }


def _save_manifest(path: Path, manifest: dict) -> None:
    manifest["updated_at"] = _now()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))


def _run(cmd: list[str], env: dict[str, str]) -> None:
    subprocess.run(cmd, check=True, cwd=ROOT_DIR, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_hf_max.yaml")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--cpu", type=int, default=8)
    parser.add_argument("--memory-mb", type=int, default=32768)
    parser.add_argument("--start-chunk", type=int, default=0)
    parser.add_argument("--end-chunk", type=int)
    parser.add_argument("--reports-dir", default="reports/hf_max_rebuild")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT_DIR / args.config).read_text())
    universe = _resolve_universe(cfg)
    total_chunks = math.ceil(len(universe) / args.chunk_size) if universe else 0
    end_chunk = args.end_chunk if args.end_chunk is not None else max(total_chunks - 1, -1)

    reports_dir = ROOT_DIR / args.reports_dir
    manifest_path = reports_dir / "manifest.json"
    merge_summary_path = reports_dir / "merge_summary.json"
    manifest = _load_manifest(manifest_path)
    manifest["config"] = args.config
    manifest["chunk_size"] = args.chunk_size
    manifest["total_tickers"] = len(universe)
    manifest["total_chunks"] = total_chunks
    manifest["status"] = "running"
    _save_manifest(manifest_path, manifest)

    env = os.environ.copy()
    env["MODAL_CPU"] = str(args.cpu)
    env["MODAL_MEMORY_MB"] = str(args.memory_mb)

    for chunk_id in range(args.start_chunk, end_chunk + 1):
        chunk_summary = reports_dir / f"chunk_{chunk_id:03d}.json"
        existing = manifest["chunks"].get(str(chunk_id), {})
        if existing.get("status") in {"ok", "exists", "no_features", "empty"} and chunk_summary.exists():
            continue

        manifest["chunks"][str(chunk_id)] = {
            "status": "running",
            "started_at": _now(),
            "summary_path": str(chunk_summary.relative_to(ROOT_DIR)),
        }
        _save_manifest(manifest_path, manifest)

        cmd = [
            sys.executable,
            "-m",
            "modal",
            "run",
            "scripts/modal_build_chunk.py",
            "--config",
            args.config,
            "--chunk-id",
            str(chunk_id),
            "--chunk-size",
            str(args.chunk_size),
            "--summary",
            str(chunk_summary),
        ]
        attempt = 0
        while True:
            attempt += 1
            manifest["chunks"][str(chunk_id)]["attempt"] = attempt
            _save_manifest(manifest_path, manifest)
            try:
                _run(cmd, env)
                break
            except subprocess.CalledProcessError as exc:
                manifest["chunks"][str(chunk_id)] = {
                    "status": "failed",
                    "chunk_id": chunk_id,
                    "attempt": attempt,
                    "exit_code": exc.returncode,
                    "summary_path": str(chunk_summary.relative_to(ROOT_DIR)),
                    "failed_at": _now(),
                }
                _save_manifest(manifest_path, manifest)
                if attempt > args.retries:
                    raise

        summary = json.loads(chunk_summary.read_text())
        summary["completed_at"] = _now()
        chunk_summary.write_text(json.dumps(summary, indent=2))
        manifest["chunks"][str(chunk_id)] = summary
        _save_manifest(manifest_path, manifest)

    if not args.skip_merge:
        manifest["merge"] = {"status": "running", "started_at": _now()}
        _save_manifest(manifest_path, manifest)
        cmd = [
            sys.executable,
            "-m",
            "modal",
            "run",
            "scripts/modal_merge_corpus.py",
            "--config",
            args.config,
            "--summary",
            str(merge_summary_path),
        ]
        _run(cmd, env)
        merge_summary = json.loads(merge_summary_path.read_text())
        merge_summary["completed_at"] = _now()
        merge_summary_path.write_text(json.dumps(merge_summary, indent=2))
        manifest["merge"] = merge_summary

    manifest["status"] = "done"
    _save_manifest(manifest_path, manifest)


if __name__ == "__main__":
    main()
