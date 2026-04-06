from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lightning_studio_utils import (
    build_bootstrap_command,
    build_repo_sync_command,
    build_session_command,
    ensure_studio_auth_env,
    ensure_studio_exists,
    ensure_studio_running,
    execute_studio_command,
    get_client_and_project,
    get_session_status,
    load_studio_config,
    resolve_studio,
    resolve_studio_instance,
    studio_status_report,
    wait_for_session_status,
)
from lightning_cloud_utils import json_safe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lightning_run.yaml")
    parser.add_argument("--out", default="lightning-studio-status.json")
    parser.add_argument("--restart-if-needed", action="store_true")
    parser.add_argument("--allow-create", action="store_true")
    args = parser.parse_args()

    config = load_studio_config(args.config)
    ensure_studio_auth_env()
    client, project = get_client_and_project()

    studio = resolve_studio(client, project.project_id, config)
    bootstrap_payload = None
    if studio is None and args.restart_if_needed:
        studio = ensure_studio_exists(client, project.project_id, config, allow_create=args.allow_create)
    if studio is None:
        payload = json.dumps({"project_id": project.project_id, "studio": None, "instance": None, "session": None}, indent=2)
        Path(args.out).write_text(payload + "\n")
        print(payload)
        return

    studio_id = str(getattr(studio, "id", "") or "")
    instance = resolve_studio_instance(client, project.project_id, studio_id)
    session_status = get_session_status(client, project.project_id, studio_id, config.studio_session_name)

    should_restart = args.restart_if_needed and (
        instance is None
        or str(getattr(instance, "phase", "") or "") != "CLOUD_SPACE_INSTANCE_STATE_RUNNING"
        or session_status is None
        or session_status["state"] != "running"
    )
    if should_restart:
        operation_suffix = str(int(time.time()))
        instance = ensure_studio_running(client, project.project_id, studio, config)
        repo_sync = execute_studio_command(
            client,
            project.project_id,
            studio_id,
            command=build_repo_sync_command(config),
            session_name=f"{config.studio_session_name}-repo-sync-{operation_suffix}",
            detached=False,
        )
        report_repo_sync = json_safe(repo_sync.to_dict() if hasattr(repo_sync, "to_dict") else repo_sync)
        bootstrap = execute_studio_command(
            client,
            project.project_id,
            studio_id,
            command=build_bootstrap_command(config),
            session_name=f"{config.studio_session_name}-bootstrap-{operation_suffix}",
            detached=False,
        )
        bootstrap_payload = json_safe(bootstrap.to_dict() if hasattr(bootstrap, "to_dict") else bootstrap)
        launch = execute_studio_command(
            client,
            project.project_id,
            studio_id,
            command=build_session_command(config),
            session_name=config.studio_session_name,
            detached=True,
        )
        session_status = wait_for_session_status(client, project.project_id, studio_id, config.studio_session_name) or {
            "state": "unknown",
        }
        session_status["launch"] = json_safe(launch.to_dict() if hasattr(launch, "to_dict") else launch)
    else:
        report_repo_sync = None

    report = studio_status_report(studio=studio, instance=instance, session_status=session_status, project=project)
    if report_repo_sync is not None:
        report["repo_sync"] = report_repo_sync
    if bootstrap_payload is not None:
        report["bootstrap"] = bootstrap_payload
    payload = json.dumps(json_safe(report), indent=2)
    Path(args.out).write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
