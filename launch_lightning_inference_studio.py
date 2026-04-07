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


def _candidate_urls(studio_id: str, instance_payload: dict[str, Any]) -> list[str]:
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
    for token in tokens:
        candidate = f"https://8000-{token}.cloudspaces.litng.ai"
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
        "python -m uvicorn trained_model_service_runtime:app --host 0.0.0.0 --port 8000",
    ]
    return f"bash -lc {shlex.quote(chr(10).join(script_lines))}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="quant_platform/configs/lightning_inference_studio.yaml")
    parser.add_argument("--status-out", default="")
    parser.add_argument("--allow-create", action="store_true")
    parser.add_argument("--force-restart-instance", action="store_true")
    args = parser.parse_args()

    config = load_studio_config(args.config)
    auth_env = ensure_auth_env()
    os.environ.update(auth_env)
    set_process_env(auth_env)
    ensure_studio_auth_env()
    client, project = get_client_and_project()

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
    repo_sync = execute_studio_command(
        client,
        project.project_id,
        studio_id,
        command=build_repo_sync_command(config),
        session_name=f"{config.studio_session_name}-repo-sync-{int(time.time())}",
        detached=False,
    )
    bootstrap = execute_studio_command(
        client,
        project.project_id,
        studio_id,
        command=build_bootstrap_command(config),
        session_name=f"{config.studio_session_name}-bootstrap-{int(time.time())}",
        detached=False,
    )
    launch = execute_studio_command(
        client,
        project.project_id,
        studio_id,
        command=_build_service_command(config),
        session_name=config.studio_session_name,
        detached=True,
    )
    session_status = wait_for_session_status(
        client,
        project.project_id,
        studio_id,
        config.studio_session_name,
        timeout_seconds=90,
        poll_seconds=5,
    ) or {"state": "unknown"}
    instance = resolve_studio_instance(client, project.project_id, studio_id)
    instance_payload = _instance_payload(instance)
    candidate_urls = _candidate_urls(studio_id, instance_payload)
    active_url, health_payload = _wait_for_reachable_health(candidate_urls, timeout_seconds=1200)

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
