from __future__ import annotations

import argparse
import json
import math
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


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=ROOT_DIR)


def _chunk_bounds(total_items: int, target_chunks: int | None, chunk_size: int) -> list[tuple[int, int]]:
    if total_items <= 0:
        return []
    if target_chunks and target_chunks > 0:
        chunk_count = min(target_chunks, total_items)
        base, remainder = divmod(total_items, chunk_count)
        bounds = []
        start = 0
        for idx in range(chunk_count):
            size = base + (1 if idx < remainder else 0)
            end = start + size
            bounds.append((start, end))
            start = end
        return bounds

    bounds = []
    start = 0
    while start < total_items:
        end = min(total_items, start + chunk_size)
        bounds.append((start, end))
        start = end
    return bounds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_hf_max.yaml")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--target-chunks", type=int)
    parser.add_argument("--start-chunk", type=int, default=0)
    parser.add_argument("--end-chunk", type=int)
    parser.add_argument("--reports-dir", default="reports/oci_hf_liquid")
    parser.add_argument("--retries", type=int, default=1)
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT_DIR / args.config).read_text())
    universe = _resolve_universe(cfg)
    chunk_size = args.chunk_size
    if args.target_chunks:
        chunk_size = max(1, math.ceil(len(universe) / args.target_chunks)) if universe else 1
    bounds = _chunk_bounds(len(universe), args.target_chunks, chunk_size)
    total_chunks = len(bounds)
    end_chunk = args.end_chunk if args.end_chunk is not None else max(total_chunks - 1, -1)

    reports_dir = ROOT_DIR / args.reports_dir
    manifest_path = reports_dir / "manifest.json"
    manifest = _load_manifest(manifest_path)
    manifest["config"] = args.config
    manifest["chunk_size"] = chunk_size
    manifest["target_chunks"] = args.target_chunks
    manifest["total_tickers"] = len(universe)
    manifest["total_chunks"] = total_chunks
    manifest["status"] = "running"
    _save_manifest(manifest_path, manifest)

    for chunk_id in range(args.start_chunk, end_chunk + 1):
        start_idx, end_idx = bounds[chunk_id]
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
            "scripts/gha_build_chunk.py",
            "--config",
            args.config,
            "--chunk-id",
            str(chunk_id),
            "--chunk-size",
            str(chunk_size),
            "--start-idx",
            str(start_idx),
            "--end-idx",
            str(end_idx),
            "--summary",
            str(chunk_summary),
        ]

        attempt = 0
        while True:
            attempt += 1
            manifest["chunks"][str(chunk_id)]["attempt"] = attempt
            _save_manifest(manifest_path, manifest)
            try:
                _run(cmd)
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

    manifest["status"] = "done"
    _save_manifest(manifest_path, manifest)


if __name__ == "__main__":
    main()
