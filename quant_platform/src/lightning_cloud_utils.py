from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Iterable
import urllib.error
import urllib.request

import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_AUTH_URL = "https://api.lightning.ai"
DEFAULT_CLOUD_URL = "https://lightning.ai"
DEFAULT_CLOUD_CLUSTER_ID = "lightning-public-prod"
DEFAULT_APP_ENTRYPOINT = "lightning_cloud_app.py"
DEFAULT_RESTART_EXIT_CODES = (75, 137, 143)
TERMINAL_PHASES = {
    "LIGHTNINGAPP_INSTANCE_STATE_COMPLETED",
    "LIGHTNINGAPP_INSTANCE_STATE_DELETED",
    "LIGHTNINGAPP_INSTANCE_STATE_FAILED",
    "LIGHTNINGAPP_INSTANCE_STATE_STOPPED",
}
RESTARTABLE_PHASES = {
    "LIGHTNINGAPP_INSTANCE_STATE_FAILED",
    "LIGHTNINGAPP_INSTANCE_STATE_NOT_STARTED",
    "LIGHTNINGAPP_INSTANCE_STATE_STOPPED",
}
ACTIVE_PHASES = {
    "LIGHTNINGAPP_INSTANCE_STATE_IMAGE_BUILDING",
    "LIGHTNINGAPP_INSTANCE_STATE_PENDING",
    "LIGHTNINGAPP_INSTANCE_STATE_RUNNING",
}


@dataclass(frozen=True)
class LightningCloudConfig:
    app_name: str
    command: str
    tracked_paths: tuple[str, ...]
    checkpoint_dir: str = ".lightning-checkpoints"
    save_every_seconds: int = 4 * 60 * 60
    grace_period_seconds: int = 300
    cloud_cluster_id: str = DEFAULT_CLOUD_CLUSTER_ID
    cloud_compute_name: str = "cpu-2"
    force_run_on_launch: bool = True
    disk_size_gb: int = 30
    drive_id: str = ""
    work_requirements_file: str = "requirements.txt"
    app_entrypoint: str = DEFAULT_APP_ENTRYPOINT
    config_env_var: str = "LIGHTNING_RUN_CONFIG"
    max_restarts: int = 16
    restart_exit_codes: tuple[int, ...] = DEFAULT_RESTART_EXIT_CODES
    without_server: bool = True
    blocking: bool = False
    open_ui: bool = False
    relaunch_if_missing: bool = True
    restart_on_terminal: bool = True
    app_env: dict[str, str] = field(default_factory=dict)


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _request_json(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request_headers = dict(headers or {})
    data = None
    if payload is not None:
        request_headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, headers=request_headers, data=data, method=method)
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def fetch_user_id(username: str, api_key: str, *, auth_url: str = DEFAULT_AUTH_URL) -> str:
    login_payload = _request_json(
        f"{auth_url}/v1/auth/login",
        method="POST",
        payload={"username": username, "apiKey": api_key},
    )
    token = login_payload.get("token")
    if not token:
        raise RuntimeError("Lightning login succeeded without returning a token.")
    user_payload = _request_json(
        f"{auth_url}/v1/auth/user",
        headers={"Authorization": f"Bearer {token}"},
    )
    user_id = _clean(str(user_payload.get("id", "")))
    if not user_id:
        raise RuntimeError("Lightning user lookup did not return a user id.")
    return user_id


def ensure_auth_env(
    base_env: dict[str, str] | None = None,
    *,
    auth_url: str = DEFAULT_AUTH_URL,
    cloud_url: str = DEFAULT_CLOUD_URL,
    cluster_id: str | None = None,
) -> dict[str, str]:
    env = dict(base_env or os.environ)
    username = _clean(env.get("LIGHTNING_USERNAME"))
    api_key = _clean(env.get("LIGHTNING_API_KEY"))
    user_id = _clean(env.get("LIGHTNING_USER_ID"))
    resolved_cluster_id = (
        _clean(cluster_id) or _clean(env.get("LIGHTNING_CLUSTER_ID")) or _clean(env.get("GRID_CLUSTER_ID")) or DEFAULT_CLOUD_CLUSTER_ID
    )
    if not username or not api_key:
        raise RuntimeError("Lightning credentials are missing. Set LIGHTNING_USERNAME and LIGHTNING_API_KEY.")
    if not user_id:
        user_id = fetch_user_id(username=username, api_key=api_key, auth_url=auth_url)
    env["LIGHTNING_USERNAME"] = username
    env["LIGHTNING_API_KEY"] = api_key
    env["LIGHTNING_USER_ID"] = user_id
    env["LIGHTNING_CLUSTER_ID"] = resolved_cluster_id
    env["GRID_CLUSTER_ID"] = resolved_cluster_id
    env.setdefault("LIGHTNING_CLOUD_URL", cloud_url)
    env.setdefault("GRID_URL", cloud_url)
    return env


def _normalized_env_items(raw_env: Any) -> dict[str, str]:
    if raw_env is None:
        return {}
    if not isinstance(raw_env, dict):
        raise ValueError("The 'app_env' config field must be a mapping of environment variable names to values.")
    normalized: dict[str, str] = {}
    for key, value in raw_env.items():
        key_text = str(key).strip()
        if not key_text:
            continue
        normalized[key_text] = "" if value is None else str(value)
    return normalized


def sanitize_drive_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug or "lightning-autoresume"


def next_app_name(app_name: str) -> str:
    suffix = datetime.now(UTC).strftime("%Y%m%d%H%M%S").lower()
    return f"{app_name}-{suffix}"


def load_run_config(path: str | Path) -> LightningCloudConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text()) or {}
    app_name = _clean(str(payload.get("app_name") or payload.get("run_name") or ""))
    command = _clean(str(payload.get("command") or ""))
    tracked_paths = tuple(str(item) for item in (payload.get("tracked_paths") or []) if str(item).strip())
    if not app_name:
        raise ValueError("The Lightning config must define 'app_name' (or legacy 'run_name').")
    if not command:
        raise ValueError("The Lightning config must define 'command'.")
    if not tracked_paths:
        raise ValueError("The Lightning config must define at least one tracked path.")

    drive_id = _clean(str(payload.get("drive_id") or "")) or sanitize_drive_id(f"{app_name}-checkpoints")
    work_requirements_file = _clean(str(payload.get("work_requirements_file") or payload.get("dependency_file") or ""))
    cloud_cluster_id = _clean(str(payload.get("cloud_cluster_id") or "")) or DEFAULT_CLOUD_CLUSTER_ID
    cloud_compute_name = _clean(str(payload.get("cloud_compute_name") or "")) or "cpu-2"
    restart_exit_codes = payload.get("restart_exit_codes") or DEFAULT_RESTART_EXIT_CODES

    return LightningCloudConfig(
        app_name=app_name,
        command=command,
        tracked_paths=tracked_paths,
        checkpoint_dir=str(payload.get("checkpoint_dir") or ".lightning-checkpoints"),
        save_every_seconds=int(payload.get("save_every_seconds") or 4 * 60 * 60),
        grace_period_seconds=int(payload.get("grace_period_seconds") or 300),
        cloud_cluster_id=cloud_cluster_id,
        cloud_compute_name=cloud_compute_name,
        force_run_on_launch=bool(payload.get("force_run_on_launch", True)),
        disk_size_gb=int(payload.get("disk_size_gb") or 30),
        drive_id=drive_id,
        work_requirements_file=work_requirements_file or "requirements.txt",
        app_entrypoint=str(payload.get("app_entrypoint") or DEFAULT_APP_ENTRYPOINT),
        config_env_var=str(payload.get("config_env_var") or "LIGHTNING_RUN_CONFIG"),
        max_restarts=int(payload.get("max_restarts") or 16),
        restart_exit_codes=tuple(int(code) for code in restart_exit_codes),
        without_server=bool(payload.get("without_server", True)),
        blocking=bool(payload.get("blocking", False)),
        open_ui=bool(payload.get("open_ui", False)),
        relaunch_if_missing=bool(payload.get("relaunch_if_missing", True)),
        restart_on_terminal=bool(payload.get("restart_on_terminal", True)),
        app_env=_normalized_env_items(payload.get("app_env")),
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        return json_safe(value.to_dict())
    return value


def set_process_env(env: dict[str, str]) -> None:
    for key, value in env.items():
        os.environ[key] = value


def get_client_and_project(project_id: str | None = None):
    from lightning_app.utilities.cloud import _get_project
    from lightning_app.utilities.network import LightningClient

    client = LightningClient(retry=False)
    project = _get_project(client, project_id=project_id, verbose=False)
    os.environ["LIGHTNING_CLOUD_PROJECT_ID"] = project.project_id
    return client, project


def list_apps(client, project_id: str) -> list[Any]:
    response = client.lightningapp_instance_service_list_lightningapp_instances(project_id=project_id)
    return list(response.lightningapps or [])


def list_named_apps(client, project_id: str, app_name: str) -> list[Any]:
    apps = list_apps(client, project_id)
    matches = [
        app
        for app in apps
        if str(getattr(app, "name", "") or "") == app_name
        or str(getattr(app, "name", "") or "").startswith(f"{app_name}-")
    ]
    return sorted(matches, key=lambda app: getattr(app, "created_at", None) or datetime.min)


def find_app_by_name(client, project_id: str, app_name: str):
    matches = list_named_apps(client, project_id, app_name)
    return matches[-1] if matches else None


def phase_name(app: Any) -> str:
    return str(getattr(getattr(app, "status", None), "phase", "") or "")


def is_terminal_phase(phase: str) -> bool:
    return phase in TERMINAL_PHASES


def is_restartable_phase(phase: str) -> bool:
    return phase in RESTARTABLE_PHASES


def build_launch_command(config: LightningCloudConfig, *, config_path: str | Path) -> list[str]:
    command = [
        sys.executable,
        str((ROOT_DIR / "scripts" / "lightning_cloud_dispatch.py").resolve()),
        "--entrypoint",
        str(config.app_entrypoint),
        "--name",
        config.app_name,
        "--config-env-var",
        config.config_env_var,
        "--config-path",
        str(config_path),
        "--blocking",
        str(config.blocking).lower(),
        "--open-ui",
        str(config.open_ui).lower(),
    ]
    if config.without_server:
        command.append("--without-server")
    if config.force_run_on_launch:
        command.append("--force-running")

    env_items = {config.config_env_var: str(config_path), **config.app_env}
    for key, value in env_items.items():
        command.extend(["--env", f"{key}={value}"])
    return command


def launch_app(
    config: LightningCloudConfig,
    *,
    config_path: str | Path,
    env: dict[str, str] | None = None,
) -> None:
    subprocess.run(
        build_launch_command(config, config_path=config_path),
        cwd=ROOT_DIR,
        env=dict(env or os.environ),
        check=True,
    )


def delete_app(client, project_id: str, app_id: str):
    return client.lightningapp_instance_service_delete_lightningapp_instance(project_id=project_id, id=app_id)


def delete_matching_apps(
    client,
    project_id: str,
    app_name: str,
    *,
    phases: Iterable[str] | None = None,
) -> list[str]:
    allowed_phases = set(phases or ())
    deleted_ids: list[str] = []
    for app in list_named_apps(client, project_id, app_name):
        if allowed_phases and phase_name(app) not in allowed_phases:
            continue
        app_id = _clean(str(getattr(app, "id", "")))
        if not app_id:
            continue
        delete_app(client, project_id, app_id)
        deleted_ids.append(app_id)
    return deleted_ids


def wait_for_app_removal(
    client,
    project_id: str,
    app_name: str,
    *,
    timeout_seconds: int = 120,
    poll_seconds: int = 5,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() <= deadline:
        if not list_named_apps(client, project_id, app_name):
            return True
        time.sleep(poll_seconds)
    return False

def wait_for_app(
    client,
    project_id: str,
    app_name: str,
    *,
    timeout_seconds: int = 300,
    poll_seconds: int = 5,
):
    deadline = time.monotonic() + timeout_seconds
    latest = None
    while time.monotonic() <= deadline:
        latest = find_app_by_name(client, project_id, app_name)
        if latest is not None and phase_name(latest) in ACTIVE_PHASES | TERMINAL_PHASES:
            return latest
        time.sleep(poll_seconds)
    return latest


def list_app_artifacts(client, project_id: str, app_id: str) -> list[Any]:
    response = client.lightningapp_instance_service_list_lightningapp_instance_artifacts(project_id=project_id, id=app_id)
    return list(getattr(response, "artifacts", []) or [])


def _download_text(url: str) -> str:
    with urllib.request.urlopen(url) as response:
        return response.read().decode("utf-8", errors="replace")


def collect_logs_text(client, project_id: str, app_id: str, *, max_pages: int = 2) -> str:
    logs_response = client.lightningapp_instance_service_get_lightningapp_instance_logs(project_id=project_id, id=app_id)
    pages = list(getattr(logs_response, "pages", []) or [])
    if not pages:
        return ""
    chunks: list[str] = []
    for page in pages[-max_pages:]:
        page_url = getattr(page, "url", "")
        if not page_url:
            continue
        try:
            chunks.append(_download_text(page_url))
        except urllib.error.URLError as exc:
            chunks.append(f"[log-download-error] {exc}")
    return "\n".join(chunk.rstrip() for chunk in chunks if chunk).strip()


def download_selected_artifacts(
    artifacts: Iterable[Any],
    *,
    destination_dir: Path,
    include_patterns: Iterable[str] = (),
) -> list[Path]:
    compiled = [re.compile(pattern) for pattern in include_patterns if pattern]
    downloaded: list[Path] = []
    destination_dir.mkdir(parents=True, exist_ok=True)

    for artifact in artifacts:
        filename = str(getattr(artifact, "filename", "") or "")
        if compiled and not any(pattern.search(filename) for pattern in compiled):
            continue
        url = str(getattr(artifact, "url", "") or "")
        if not filename or not url:
            continue
        output_path = destination_dir / filename.lstrip("/")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as response:
            output_path.write_bytes(response.read())
        downloaded.append(output_path)
    return downloaded
