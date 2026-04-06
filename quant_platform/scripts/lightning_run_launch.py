from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.lightning_cloud_utils import (
    RESTARTABLE_PHASES,
    delete_matching_apps,
    ensure_auth_env,
    find_app_by_name,
    get_client_and_project,
    is_restartable_phase,
    json_safe,
    launch_app,
    load_run_config,
    next_app_name,
    phase_name,
    set_process_env,
    wait_for_app_removal,
    wait_for_app,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lightning_run.yaml")
    parser.add_argument("--project-id")
    parser.add_argument("--wait-seconds", type=int, default=300)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (ROOT_DIR / config_path).resolve()

    config = load_run_config(config_path)
    auth_env = ensure_auth_env(cluster_id=config.cloud_cluster_id)
    set_process_env(auth_env)
    client, project = get_client_and_project(project_id=args.project_id)

    action = "noop"
    app = find_app_by_name(client, project.project_id, config.app_name)
    if app is None:
        action = "launch"
        launch_app(config, config_path=config_path, env=dict(os.environ))
        app = wait_for_app(client, project.project_id, config.app_name, timeout_seconds=args.wait_seconds)

    if app is not None and is_restartable_phase(phase_name(app)) and config.restart_on_terminal:
        action = "relaunch" if action == "noop" else action
        delete_matching_apps(client, project.project_id, config.app_name, phases=RESTARTABLE_PHASES)
        wait_for_app_removal(client, project.project_id, config.app_name, timeout_seconds=min(args.wait_seconds, 120))
        launch_app(replace(config, app_name=next_app_name(config.app_name)), config_path=config_path, env=dict(os.environ))
        app = wait_for_app(client, project.project_id, config.app_name, timeout_seconds=args.wait_seconds)

    payload = {
        "action": action,
        "project_id": project.project_id,
        "app_name": config.app_name,
        "app": json_safe(app) if app is not None else None,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
