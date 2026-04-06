from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.lightning_cloud_utils import (
    RESTARTABLE_PHASES,
    collect_logs_text,
    delete_matching_apps,
    download_selected_artifacts,
    ensure_auth_env,
    find_app_by_name,
    get_client_and_project,
    is_restartable_phase,
    json_safe,
    launch_app,
    list_app_artifacts,
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
    parser.add_argument("--out", default="lightning-run-status.json")
    parser.add_argument("--logs-out", default="lightning-app-logs.txt")
    parser.add_argument("--download-artifacts-dir")
    parser.add_argument("--artifact-pattern", action="append", default=[])
    parser.add_argument("--wait-seconds", type=int, default=300)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--relaunch-if-missing", action="store_true")
    parser.add_argument("--restart-on-terminal", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (ROOT_DIR / config_path).resolve()

    config = load_run_config(config_path)
    auth_env = ensure_auth_env(cluster_id=config.cloud_cluster_id)
    set_process_env(auth_env)
    client, project = get_client_and_project(project_id=args.project_id)

    allow_relaunch = args.relaunch_if_missing or config.relaunch_if_missing
    allow_restart = args.restart_on_terminal or config.restart_on_terminal
    generated_at = datetime.now(UTC).isoformat()

    action = "noop"
    app = find_app_by_name(client, project.project_id, config.app_name)
    if app is None and allow_relaunch:
        action = "launch"
        launch_app(config, config_path=config_path, env=dict(os.environ))
        app = wait_for_app(client, project.project_id, config.app_name, timeout_seconds=args.wait_seconds)

    if app is not None:
        phase = phase_name(app)
        if is_restartable_phase(phase) and allow_restart:
            action = "relaunch" if action == "noop" else action
            delete_matching_apps(client, project.project_id, config.app_name, phases=RESTARTABLE_PHASES)
            wait_for_app_removal(client, project.project_id, config.app_name, timeout_seconds=min(args.wait_seconds, 120))
            launch_app(replace(config, app_name=next_app_name(config.app_name)), config_path=config_path, env=dict(os.environ))
            app = wait_for_app(client, project.project_id, config.app_name, timeout_seconds=args.wait_seconds)

    payload = {
        "generated_at_utc": generated_at,
        "project_id": project.project_id,
        "app_name": config.app_name,
        "action": action,
        "present": app is not None,
    }

    if app is None:
        payload["status"] = "missing"
        Path(args.out).write_text(json.dumps(payload, indent=2))
        if args.allow_missing:
            return
        raise SystemExit(f"Lightning app '{config.app_name}' was not found.")

    payload["app"] = json_safe(app)

    log_error = None
    logs_text = ""
    try:
        logs_text = collect_logs_text(client, project.project_id, app.id)
    except Exception as exc:  # pragma: no cover - depends on cloud log availability
        log_error = str(exc)

    if args.logs_out:
        logs_path = Path(args.logs_out)
        if logs_text:
            logs_path.write_text(logs_text)
        elif log_error:
            logs_path.write_text(log_error)

    artifact_error = None
    artifacts = []
    downloaded = []
    try:
        artifacts = list_app_artifacts(client, project.project_id, app.id)
        if args.download_artifacts_dir and artifacts:
            downloaded = [
                str(path)
                for path in download_selected_artifacts(
                    artifacts,
                    destination_dir=Path(args.download_artifacts_dir),
                    include_patterns=args.artifact_pattern,
                )
            ]
    except Exception as exc:  # pragma: no cover - depends on artifact support in cloud
        artifact_error = str(exc)

    payload["artifacts"] = json_safe(artifacts)
    payload["downloaded_artifacts"] = downloaded
    if log_error:
        payload["log_error"] = log_error
    if artifact_error:
        payload["artifact_error"] = artifact_error

    Path(args.out).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
