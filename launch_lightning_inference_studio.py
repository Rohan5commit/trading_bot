from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any

import requests
import re


ROOT_DIR = Path(__file__).resolve().parent
QP_SRC_DIR = ROOT_DIR / "quant_platform" / "src"
if str(QP_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(QP_SRC_DIR))

from lightning_cloud_utils import ensure_auth_env, json_safe, set_process_env  # noqa: E402
from lightning_studio_utils import (  # noqa: E402
    build_bootstrap_command,
    build_repo_sync_command,
    ensure_studio_auth_env,
    ensure_studio_exists,
    ensure_studio_running,
    execute_studio_command,
    get_client_and_project,
    get_session_status,
    load_studio_config,
    resolve_studio_instance,
    wait_for_session_status,
)


RUNNING_PHASE = "CLOUD_SPACE_INSTANCE_STATE_RUNNING"


def _resolved_project_id() -> str | None:
    for key in ("LIGHTNING_CLOUD_PROJECT_ID", "LIGHTNING_PROJECT_ID"):
        value = str(os.getenv(key) or "").strip()
        if value:
            return value
    return None


def _service_port(config) -> int:
    override = str(os.getenv("LIGHTNING_INFERENCE_PORT") or "").strip()
    if override:
        return int(override)
    match = re.search(r"--port\s+(\d+)", str(config.run.command or ""))
    if match:
        return int(match.group(1))
    return 8000


def _instance_payload(instance: Any) -> dict[str, Any]:
    if instance is None:
        return {}
    if hasattr(instance, "to_dict"):
        return instance.to_dict()
    return json_safe(instance)


def _wait_for_instance_phase(client, project_id: str, studio_id: str, *, phases: set[str], timeout_seconds: int = 300) -> dict[str, Any] | None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        instance = resolve_studio_instance(client, project_id, studio_id)
        payload = _instance_payload(instance)
        phase = str(getattr(instance, "phase", "") or payload.get("phase") or "").strip()
        if phase in phases:
            return payload
        time.sleep(5)
    return None


def _command_payload(result: Any) -> dict[str, Any]:
    if hasattr(result, "to_dict"):
        return json_safe(result.to_dict())
    return json_safe(result)


def _contains_setup_not_ready(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    haystack = json.dumps(payload, sort_keys=True).lower()
    return "still setting things up" in haystack or "progress bar at the top of the studio disappears" in haystack


def _execute_checked(
    client,
    project_id: str,
    studio_id: str,
    *,
    command: str,
    session_name: str,
    detached: bool,
    max_attempts: int = 20,
    retry_sleep_seconds: int = 30,
):
    last_payload: dict[str, Any] | None = None
    for attempt in range(1, max_attempts + 1):
        result = execute_studio_command(
            client,
            project_id,
            studio_id,
            command=command,
            session_name=session_name,
            detached=detached,
        )
        payload = _command_payload(result)
        last_payload = payload
        if detached:
            if _contains_setup_not_ready(payload):
                time.sleep(retry_sleep_seconds)
                continue
            return result
        exit_code = payload.get("exit_code")
        if _contains_setup_not_ready(payload):
            time.sleep(retry_sleep_seconds)
            continue
        if exit_code in (0, None):
            return result
        raise RuntimeError(json.dumps(payload, indent=2))
    raise RuntimeError(json.dumps({"error": "Studio command never became ready", "last_payload": last_payload}, indent=2))


def _launch_service_session(
    client,
    project_id: str,
    studio_id: str,
    *,
    command: str,
    session_name: str,
    max_attempts: int = 20,
    retry_sleep_seconds: int = 30,
):
    last_status: dict[str, Any] | None = None
    last_launch: dict[str, Any] | None = None
    for attempt in range(1, max_attempts + 1):
        launch = _execute_checked(
            client,
            project_id,
            studio_id,
            command=command,
            session_name=session_name,
            detached=True,
            max_attempts=1,
            retry_sleep_seconds=retry_sleep_seconds,
        )
        last_launch = _command_payload(launch)
        session_status = wait_for_session_status(
            client,
            project_id,
            studio_id,
            session_name,
            timeout_seconds=120,
            poll_seconds=10,
        ) or {"state": "unknown"}
        last_status = session_status
        if session_status.get("state") == "running":
            return launch, session_status
        if _contains_setup_not_ready(session_status):
            time.sleep(retry_sleep_seconds)
            continue
        if session_status.get("state") == "failed":
            raise RuntimeError(json.dumps({"launch": last_launch, "session": json_safe(session_status)}, indent=2))
        time.sleep(retry_sleep_seconds)
    raise RuntimeError(json.dumps({"error": "Studio service session never reached running", "launch": last_launch, "session": last_status}, indent=2))


def _candidate_urls(studio_id: str, instance_payload: dict[str, Any], *, port: int) -> list[str]:
    tokens: list[str] = []
    for value in (
        studio_id,
        instance_payload.get("cloud_space_id"),
        instance_payload.get("id"),
    ):
        text = str(value or "").strip()
        if text and text not in tokens:
            tokens.append(text)
    urls: list[str] = []
    app_url = str(instance_payload.get("app_url") or "").strip().rstrip("/")
    if app_url:
        urls.append(f"{app_url}/proxy/{port}")
    for token in tokens:
        candidate = f"https://{port}-{token}.cloudspaces.litng.ai"
        if candidate not in urls:
            urls.append(candidate)
    return urls


def _request_headers() -> dict[str, str]:
    headers = {"Accept": "application/json"}
    api_key = str(os.getenv("TRAINED_MODEL_API_KEY") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _wait_for_reachable_health(urls: list[str], *, timeout_seconds: int = 1200) -> tuple[str, dict[str, Any]]:
    deadline = time.time() + timeout_seconds
    last_error = "no candidate Lightning Studio URL responded"
    headers = _request_headers()
    while time.time() < deadline:
        for base_url in urls:
            health_url = base_url.rstrip("/") + "/health"
            try:
                response = requests.get(health_url, headers=headers, timeout=30)
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict) and payload.get("ok") is True:
                    return base_url, payload
                last_error = str(payload)
            except Exception as exc:  # noqa: BLE001
                last_error = f"{base_url}: {exc}"
        time.sleep(10)
    raise TimeoutError(f"Timed out waiting for Lightning inference health: {last_error}")


def _build_service_command(config) -> str:
    exports = [f"export {key}={shlex.quote(value)}" for key, value in config.run.app_env.items()]
    script_lines = [
        "set -euo pipefail",
        *exports,
        f"cd {shlex.quote(str((Path(config.studio_root_dir.rstrip('/')) / config.studio_repo_dir).as_posix()))}",
        "if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi",
        str(config.run.command or "python -m uvicorn trained_model_service_runtime:app --host 0.0.0.0 --port 8510"),
    ]
    return f"bash -lc {shlex.quote(chr(10).join(script_lines))}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="quant_platform/configs/lightning_inference_studio.yaml")
    parser.add_argument("--status-out", default="")
    parser.add_argument("--allow-create", action="store_true")
    parser.add_argument("--force-restart-instance", action="store_true")
    parser.add_argument("--skip-public-health-check", action="store_true")
    args = parser.parse_args()

    config = load_studio_config(args.config)
    auth_env = ensure_auth_env()
    os.environ.update(auth_env)
    set_process_env(auth_env)
    ensure_studio_auth_env()
    client, project = get_client_and_project(project_id=_resolved_project_id())

    studio = ensure_studio_exists(client, project.project_id, config, allow_create=args.allow_create)
    studio_id = str(getattr(studio, "id", "") or "").strip()
    if not studio_id:
        raise RuntimeError("Lightning Studio did not expose an id.")

    existing = resolve_studio_instance(client, project.project_id, studio_id)
    existing_payload = _instance_payload(existing)
    existing_phase = str(getattr(existing, "phase", "") or existing_payload.get("phase") or "").strip()
    if args.force_restart_instance and existing_phase == RUNNING_PHASE:
        client.cloud_space_service_stop_cloud_space_instance(project_id=project.project_id, id=studio_id)
        _wait_for_instance_phase(
            client,
            project.project_id,
            studio_id,
            phases={
                "",
                "CLOUD_SPACE_INSTANCE_STATE_STOPPED",
                "CLOUD_SPACE_INSTANCE_STATE_DELETED",
                "CLOUD_SPACE_INSTANCE_STATE_FAILED",
            },
            timeout_seconds=300,
        )

    instance = ensure_studio_running(client, project.project_id, studio, config, timeout_seconds=600)
    repo_sync = _execute_checked(
        client,
        project.project_id,
        studio_id,
        command=build_repo_sync_command(config),
        session_name=f"{config.studio_session_name}-repo-sync-{int(time.time())}",
        detached=False,
    )
    bootstrap = _execute_checked(
        client,
        project.project_id,
        studio_id,
        command=build_bootstrap_command(config),
        session_name=f"{config.studio_session_name}-bootstrap-{int(time.time())}",
        detached=False,
    )
    launch, session_status = _launch_service_session(
        client,
        project.project_id,
        studio_id,
        command=_build_service_command(config),
        session_name=config.studio_session_name,
    )
    instance = resolve_studio_instance(client, project.project_id, studio_id)
    instance_payload = _instance_payload(instance)
    service_port = _service_port(config)
    candidate_urls = _candidate_urls(studio_id, instance_payload, port=service_port)
    print(
        json.dumps(
            {
                "stage": "studio-launched",
                "studio_id": studio_id,
                "service_port": service_port,
                "session_status": json_safe(session_status),
                "candidate_urls": candidate_urls,
            },
            indent=2,
        ),
        flush=True,
    )
    if args.skip_public_health_check:
        active_url = candidate_urls[0] if candidate_urls else ""
        report = {
            "project_id": project.project_id,
            "project_name": project.name,
            "studio_id": studio_id,
            "studio_name": str(getattr(studio, "name", "") or ""),
            "instance": instance_payload,
            "session": json_safe(session_status),
            "repo_sync": json_safe(repo_sync.to_dict() if hasattr(repo_sync, "to_dict") else repo_sync),
            "bootstrap": json_safe(bootstrap.to_dict() if hasattr(bootstrap, "to_dict") else bootstrap),
            "launch": json_safe(launch.to_dict() if hasattr(launch, "to_dict") else launch),
            "candidate_urls": candidate_urls,
            "inference_url": active_url,
            "public_health_check_skipped": True,
        }
        payload = json.dumps(json_safe(report), indent=2)
        print(payload)
        if args.status_out:
            Path(args.status_out).write_text(payload + "\n")
        return
    try:
        active_url, health_payload = _wait_for_reachable_health(candidate_urls, timeout_seconds=1200)
    except Exception as exc:
        latest_session_status = get_session_status(client, project.project_id, studio_id, config.studio_session_name)
        failure_payload = {
            "error": str(exc),
            "studio_id": studio_id,
            "service_port": service_port,
            "candidate_urls": candidate_urls,
            "instance": instance_payload,
            "session": json_safe(latest_session_status),
        }
        raise RuntimeError(json.dumps(json_safe(failure_payload), indent=2)) from exc

    report = {
        "project_id": project.project_id,
        "project_name": project.name,
        "studio_id": studio_id,
        "studio_name": str(getattr(studio, "name", "") or ""),
        "instance": instance_payload,
        "session": json_safe(session_status),
        "repo_sync": json_safe(repo_sync.to_dict() if hasattr(repo_sync, "to_dict") else repo_sync),
        "bootstrap": json_safe(bootstrap.to_dict() if hasattr(bootstrap, "to_dict") else bootstrap),
        "launch": json_safe(launch.to_dict() if hasattr(launch, "to_dict") else launch),
        "candidate_urls": candidate_urls,
        "inference_url": active_url,
        "health": health_payload,
    }
    payload = json.dumps(json_safe(report), indent=2)
    print(payload)
    if args.status_out:
        Path(args.status_out).write_text(payload + "\n")


if __name__ == "__main__":
    main()
