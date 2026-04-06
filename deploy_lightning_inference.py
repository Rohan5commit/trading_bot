from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parent
QP_SRC_DIR = ROOT_DIR / "quant_platform" / "src"
if str(QP_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(QP_SRC_DIR))

from lightning_cloud_utils import (  # noqa: E402
    ensure_auth_env,
    find_app_by_name,
    get_client_and_project,
    json_safe,
    phase_name,
    set_process_env,
)

from lightning_app.runners.runtime import dispatch  # noqa: E402
from lightning_app.runners.runtime_type import RuntimeType  # noqa: E402


ENV_KEYS = (
    "TRAINED_MODEL_BASE_MODEL",
    "TRAINED_MODEL_NAME",
    "TRAINED_MODEL_CPU_THREADS",
    "TRAINED_MODEL_CPU",
    "TRAINED_MODEL_API_KEY",
    "TRAINED_MODEL_ADAPTER_PATH",
    "TRAINED_MODEL_ADAPTER_ARCHIVE_URL",
    "TRAINED_MODEL_ADAPTER_ARCHIVE_TOKEN",
    "TRAINED_MODEL_CACHE_DIR",
    "LIGHTNING_INFERENCE_COMPUTE_NAME",
    "LIGHTNING_INFERENCE_DISK_GB",
    "LIGHTNING_INFERENCE_PORT",
    "TRAINED_MODEL_LOG_LEVEL",
)


def _collect_env() -> dict[str, str]:
    env_vars: dict[str, str] = {}
    for key in ENV_KEYS:
        value = os.getenv(key)
        if value:
            env_vars[key] = value
    return env_vars


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-name", default="trading-bot-lightning-inference")
    parser.add_argument("--blocking", action="store_true")
    parser.add_argument("--open-ui", action="store_true")
    args = parser.parse_args()

    auth_env = ensure_auth_env()
    set_process_env(auth_env)
    client, project = get_client_and_project()

    entrypoint = ROOT_DIR / "lightning_trained_model_app.py"
    env_vars = _collect_env()

    dispatch(
        entrypoint,
        RuntimeType.CLOUD,
        start_server=False,
        no_cache=False,
        blocking=args.blocking,
        open_ui=args.open_ui,
        name=args.app_name,
        env_vars=env_vars,
        secrets={},
    )

    latest = find_app_by_name(client, project.project_id, args.app_name)
    payload = {
        "project_id": project.project_id,
        "project_name": project.name,
        "app_name": args.app_name,
        "app_id": getattr(latest, "id", None) if latest else None,
        "phase": phase_name(latest) if latest else None,
        "note": "Copy the Lightning service URL from the app layout once the inference work is running.",
    }
    print(json.dumps(json_safe(payload), indent=2))


if __name__ == "__main__":
    main()
