from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT_ID = "p-58018090"
APP_NAME = "trading-bot-cerebrium-inference"
REGION = "aws.us-east-1"
REPO_ROOT = Path(__file__).resolve().parent
APP_DIR = REPO_ROOT / "cerebrium_app"
BUILD_DIR = REPO_ROOT / ".cerebrium_build"


def _stage_app() -> Path:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    for filename in (
        "main.py",
        "requirements.txt",
        "cerebrium.toml",
    ):
        (BUILD_DIR / filename).write_text((APP_DIR / filename).read_text(encoding="utf-8"), encoding="utf-8")
    for filename in ("trained_model_service_runtime.py", "llm_sentiment.py"):
        (BUILD_DIR / filename).write_text((REPO_ROOT / filename).read_text(encoding="utf-8"), encoding="utf-8")
    return BUILD_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy the trained model runtime to Cerebrium.")
    parser.add_argument("--project-id", default=os.getenv("CEREBRIUM_PROJECT_ID", PROJECT_ID))
    parser.add_argument("--app-name", default=APP_NAME)
    parser.add_argument("--json-out", default="results/cerebrium_deploy.json")
    parser.add_argument("--skip-deploy", action="store_true")
    args = parser.parse_args()

    token = os.getenv("CEREBRIUM_SERVICE_ACCOUNT_TOKEN") or os.getenv("CEREBRIUM_API_KEY")
    if not token:
        raise SystemExit("Set CEREBRIUM_SERVICE_ACCOUNT_TOKEN as a GitHub secret or environment variable.")

    env = os.environ.copy()
    env["CEREBRIUM_SERVICE_ACCOUNT_TOKEN"] = token
    env.setdefault("CEREBRIUM_PROJECT_ID", args.project_id)

    build_dir = _stage_app()
    subprocess.check_call([sys.executable, "-m", "pip", "install", "cerebrium>=1.38.0"], env=env)
    subprocess.check_call(["cerebrium", "project", "set", args.project_id], cwd=str(build_dir), env=env)
    if not args.skip_deploy:
        subprocess.check_call(["cerebrium", "deploy", "-y"], cwd=str(build_dir), env=env)

    endpoint = f"https://api.{REGION}.cerebrium.ai/v4/{args.project_id}/{args.app_name}"
    payload = {
        "ok": True,
        "project_id": args.project_id,
        "app_name": args.app_name,
        "base_url": endpoint,
        "health_url": f"{endpoint}/health",
        "predict_url": f"{endpoint}/predict_trade_candidates",
        "warmup_url": f"{endpoint}/warmup",
    }
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
