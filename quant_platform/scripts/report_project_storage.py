from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT_DIR = Path(__file__).resolve().parents[2]
REPO = "Rohan5commit/trading_bot"


def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, cwd=ROOT_DIR).strip()


def _du_bytes(path: Path) -> int:
    try:
        out = _run(["du", "-sk", str(path)])
        kb = int(out.split()[0])
        return kb * 1024
    except Exception:
        return 0


def _gh_json(args: list[str]):
    return json.loads(_run(["gh", *args]))


def _workflow_env_value(path: Path, key: str) -> str | None:
    text = path.read_text(encoding="utf-8")
    match = re.search(rf"^\s*{re.escape(key)}:\s*\"?([^\"\n]+)\"?\s*$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def _yaml_scalar(path: Path, key: str) -> str | None:
    text = path.read_text(encoding="utf-8")
    match = re.search(rf"^\s*{re.escape(key)}:\s*([^\n#]+)", text, re.MULTILINE)
    return match.group(1).strip().strip('"').strip("'") if match else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out")
    args = parser.parse_args()

    repo_info = _gh_json(["api", f"repos/{REPO}"])
    artifacts = _gh_json(["api", f"repos/{REPO}/actions/artifacts", "--paginate"])
    releases = _gh_json(["api", f"repos/{REPO}/releases"])

    artifact_rows = artifacts.get("artifacts", []) if isinstance(artifacts, dict) else []
    nonexpired_artifact_bytes = sum(int(item.get("size_in_bytes", 0) or 0) for item in artifact_rows if not item.get("expired"))
    total_artifact_bytes = sum(int(item.get("size_in_bytes", 0) or 0) for item in artifact_rows)
    release_asset_bytes = 0
    release_count = 0
    for release in list(releases or []):
        release_count += 1
        for asset in list(release.get("assets") or []):
            release_asset_bytes += int(asset.get("size", 0) or 0)

    daily_workflow = ROOT_DIR / ".github" / "workflows" / "daily_trading_bot.yml"
    deploy_workflow = ROOT_DIR / ".github" / "workflows" / "deploy_lightning_inference.yml"
    studio_cfg = ROOT_DIR / "quant_platform" / "configs" / "lightning_inference_studio.yaml"

    payload = {
        "local_device": {
            "repo_bytes": _du_bytes(ROOT_DIR),
            "git_bytes": _du_bytes(ROOT_DIR / ".git"),
            "backtesting_bytes": _du_bytes(ROOT_DIR / "backtesting"),
            "quant_platform_bytes": _du_bytes(ROOT_DIR / "quant_platform"),
            "analysis_bytes": _du_bytes(ROOT_DIR / "analysis"),
        },
        "github": {
            "repo_bytes": int(repo_info.get("size", 0) or 0) * 1024,
            "repo_size_kb": int(repo_info.get("size", 0) or 0),
            "nonexpired_actions_artifact_bytes": nonexpired_artifact_bytes,
            "total_actions_artifact_bytes": total_artifact_bytes,
            "actions_artifact_count": len(artifact_rows),
            "release_asset_bytes": release_asset_bytes,
            "release_count": release_count,
        },
        "lightning": {
            "configured_daily_studio_disk_gb": int(float(_workflow_env_value(daily_workflow, "LIGHTNING_INFERENCE_DISK_GB") or 0)),
            "configured_deploy_disk_gb": int(float(_workflow_env_value(deploy_workflow, "LIGHTNING_INFERENCE_DISK_GB") or 0)),
            "configured_studio_yaml_disk_gb": int(float(_yaml_scalar(studio_cfg, "studio_disk_size_gb") or 0)),
            "configured_daily_compute": _workflow_env_value(daily_workflow, "LIGHTNING_INFERENCE_COMPUTE_NAME") or "",
            "configured_deploy_compute": _workflow_env_value(deploy_workflow, "LIGHTNING_INFERENCE_COMPUTE_NAME") or "",
            "notes": "Lightning file usage is not directly exposed here; this reports the configured disk allocation per run path.",
        },
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
