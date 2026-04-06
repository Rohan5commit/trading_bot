from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_snapshot_download():
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - installed in runtime
        raise RuntimeError("huggingface_hub is required for mirroring public corpora.") from exc
    return snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/aux_corpora_public.yaml")
    parser.add_argument("--reports-dir", default="reports/oci_aux_public")
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT_DIR / args.config).read_text())
    output_dir = Path(str(cfg.get("output_dir", "/data/raw_aux_public")).replace("/data", str(ROOT_DIR / "data"), 1))
    output_dir.mkdir(parents=True, exist_ok=True)

    reports_dir = ROOT_DIR / args.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = reports_dir / "manifest.json"
    manifest = {
        "status": "running",
        "started_at": _now(),
        "updated_at": _now(),
        "output_dir": str(output_dir),
        "sources": {},
    }

    snapshot_download = _load_snapshot_download()

    for source in cfg.get("sources", []):
        repo_id = source["repo_id"]
        repo_type = source.get("repo_type", "dataset")
        local_dir = output_dir / repo_id.replace("/", "__")
        state = manifest["sources"].setdefault(
            repo_id,
            {
                "status": "running",
                "repo_type": repo_type,
                "local_dir": str(local_dir),
                "started_at": _now(),
            },
        )
        manifest["updated_at"] = _now()
        manifest_path.write_text(json.dumps(manifest, indent=2))
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        state.update(
            {
                "status": "done",
                "completed_at": _now(),
                "snapshot_path": snapshot_path,
            }
        )
        manifest["updated_at"] = _now()
        manifest_path.write_text(json.dumps(manifest, indent=2))

    manifest["status"] = "done"
    manifest["updated_at"] = _now()
    manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
