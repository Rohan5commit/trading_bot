from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import time
from typing import Any

from lightning_cloud.openapi import IdCodeconfigBody, IdExecuteBody, IdStartBody, ProjectIdCloudspacesBody, V1UserRequestedComputeConfig
from lightning_cloud.openapi.rest import ApiException

from lightning_cloud_utils import (
    LightningCloudConfig,
    _clean,
    ensure_auth_env,
    get_client_and_project,
    json_safe,
    load_run_config,
    set_process_env,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_STUDIO_CLUSTER_ID = "gcp-lightning-public-prod"
DEFAULT_STUDIO_COMPUTE_NAME = "cpu-4"
DEFAULT_STUDIO_DISK_SIZE_GB = 400
DEFAULT_STUDIO_IDE = "jupyterlab"
DEFAULT_STUDIO_REPO_URL = "https://github.com/Rohan5commit/train-once-quant-platform.git"
DEFAULT_STUDIO_REPO_REF = "main"
DEFAULT_STUDIO_REPO_DIR = "train-once-quant-platform-studio"
DEFAULT_STUDIO_ROOT = "/teamspace/studios/this_studio"
DEFAULT_STUDIO_SESSION_NAME = "train-once-build-data"
RUNNING_PHASE = "CLOUD_SPACE_INSTANCE_STATE_RUNNING"


@dataclass(frozen=True)
class LightningStudioConfig:
    run: LightningCloudConfig
    studio_name: str = ""
    studio_cluster_id: str = DEFAULT_STUDIO_CLUSTER_ID
    studio_compute_name: str = DEFAULT_STUDIO_COMPUTE_NAME
    studio_disk_size_gb: int = DEFAULT_STUDIO_DISK_SIZE_GB
    studio_ide: str = DEFAULT_STUDIO_IDE
    studio_repo_url: str = DEFAULT_STUDIO_REPO_URL
    studio_repo_ref: str = DEFAULT_STUDIO_REPO_REF
    studio_repo_dir: str = DEFAULT_STUDIO_REPO_DIR
    studio_root_dir: str = DEFAULT_STUDIO_ROOT
    studio_session_name: str = DEFAULT_STUDIO_SESSION_NAME
    studio_checkpoint_dir: str = ""
    studio_auto_discover_free: bool = True
    studio_auto_start_ports: tuple[str, ...] = ()


def _git_remote_origin() -> str:
    try:
        import subprocess

        return (
            subprocess.check_output(["git", "-C", str(ROOT_DIR), "remote", "get-url", "origin"], text=True)
            .strip()
            .replace(".git", "")
            + ".git"
        )
    except Exception:
        return DEFAULT_STUDIO_REPO_URL


def load_studio_config(path: str | Path) -> LightningStudioConfig:
    run = load_run_config(path)
    payload = json.loads(Path(path).read_text()) if str(path).endswith(".json") else None
    if payload is None:
        import yaml

        payload = yaml.safe_load(Path(path).read_text()) or {}

    studio_compute_name = (
        _clean(str(os.getenv("LIGHTNING_INFERENCE_COMPUTE_NAME") or ""))
        or _clean(str(payload.get("studio_compute_name") or ""))
        or DEFAULT_STUDIO_COMPUTE_NAME
    )
    studio_disk_size_gb = int(
        _clean(str(os.getenv("LIGHTNING_INFERENCE_DISK_GB") or ""))
        or payload.get("studio_disk_size_gb")
        or DEFAULT_STUDIO_DISK_SIZE_GB
    )

    return LightningStudioConfig(
        run=run,
        studio_name=_clean(str(payload.get("studio_name") or "")) or "",
        studio_cluster_id=_clean(str(payload.get("studio_cluster_id") or "")) or DEFAULT_STUDIO_CLUSTER_ID,
        studio_compute_name=studio_compute_name,
        studio_disk_size_gb=studio_disk_size_gb,
        studio_ide=_clean(str(payload.get("studio_ide") or "")) or DEFAULT_STUDIO_IDE,
        studio_repo_url=_clean(str(payload.get("studio_repo_url") or "")) or _git_remote_origin(),
        studio_repo_ref=_clean(str(payload.get("studio_repo_ref") or "")) or DEFAULT_STUDIO_REPO_REF,
        studio_repo_dir=_clean(str(payload.get("studio_repo_dir") or "")) or DEFAULT_STUDIO_REPO_DIR,
        studio_root_dir=_clean(str(payload.get("studio_root_dir") or "")) or DEFAULT_STUDIO_ROOT,
        studio_session_name=_clean(str(payload.get("studio_session_name") or "")) or DEFAULT_STUDIO_SESSION_NAME,
        studio_checkpoint_dir=_clean(str(payload.get("studio_checkpoint_dir") or "")) or "",
        studio_auto_discover_free=bool(payload.get("studio_auto_discover_free", True)),
        studio_auto_start_ports=tuple(
            str(item).strip()
            for item in (payload.get("studio_auto_start_ports") or payload.get("auto_start_ports") or [])
            if str(item).strip()
        ),
    )


def ensure_studio_auth_env() -> dict[str, str]:
    env = ensure_auth_env()
    set_process_env(env)
    return env


def studio_repo_path(config: LightningStudioConfig) -> str:
    return f"{config.studio_root_dir.rstrip('/')}/{config.studio_repo_dir}"


def studio_checkpoint_path(config: LightningStudioConfig) -> str:
    if config.studio_checkpoint_dir:
        return config.studio_checkpoint_dir
    return f"{config.studio_root_dir.rstrip('/')}/.lightning-checkpoints/{config.studio_session_name}"


def build_repo_sync_command(config: LightningStudioConfig) -> str:
    repo_path = studio_repo_path(config)
    repo_parent = str(Path(repo_path).parent)
    repo_url = config.studio_repo_url
    repo_ref = config.studio_repo_ref
    quoted_repo_path = shlex.quote(repo_path)
    quoted_repo_parent = shlex.quote(repo_parent)
    quoted_repo_ref = shlex.quote(repo_ref)

    if config.studio_repo_url.startswith("https://github.com/") and os.environ.get("GITHUB_TOKEN"):
        repo_url = config.studio_repo_url.replace("https://github.com/", "https://x-access-token:" + os.environ["GITHUB_TOKEN"] + "@github.com/", 1)

    quoted_repo_url = shlex.quote(repo_url)

    script = "\n".join(
        [
            "set -euo pipefail",
            "export GIT_TERMINAL_PROMPT=0",
            f"mkdir -p {quoted_repo_parent}",
            f"if [ -d {quoted_repo_path}/.git ]; then",
            f"  git -C {quoted_repo_path} remote set-url origin {quoted_repo_url}",
            f"  git -C {quoted_repo_path} fetch --depth 1 origin {quoted_repo_ref}",
            f"  git -C {quoted_repo_path} reset --hard FETCH_HEAD",
            "else",
            f"  git clone --depth 1 --branch {quoted_repo_ref} {quoted_repo_url} {quoted_repo_path}",
            "fi",
        ]
    )
    return f"bash -lc {shlex.quote(script)}"


def studio_compute_config(config: LightningStudioConfig) -> V1UserRequestedComputeConfig:
    return V1UserRequestedComputeConfig(
        name=config.studio_compute_name,
        disk_size=config.studio_disk_size_gb,
        spot=False,
        same_compute_on_resume=False,
    )


def list_studios(client, project_id: str) -> list[Any]:
    response = client.cloud_space_service_list_cloud_spaces(project_id=project_id)
    return list(getattr(response, "cloudspaces", []) or [])


def list_studio_instances(client, project_id: str) -> list[Any]:
    response = client.cloud_space_service_list_cloud_space_instances(project_id=project_id)
    return list(getattr(response, "cloudspace_instances", []) or [])


def find_studio_by_name(client, project_id: str, studio_name: str):
    studio_name = studio_name.strip()
    for studio in list_studios(client, project_id):
        name = str(getattr(studio, "name", "") or "")
        display_name = str(getattr(studio, "display_name", "") or "")
        if name == studio_name or display_name == studio_name:
            return studio
    return None


def find_running_free_studio_instance(client, project_id: str):
    instances = [
        instance
        for instance in list_studio_instances(client, project_id)
        if bool(getattr(instance, "free", False)) and str(getattr(instance, "phase", "") or "") == RUNNING_PHASE
    ]
    instances.sort(key=lambda item: getattr(item, "start_timestamp", None) or getattr(item, "creation_timestamp", None) or 0, reverse=True)
    return instances[0] if instances else None


def resolve_studio(client, project_id: str, config: LightningStudioConfig):
    if config.studio_name:
        return find_studio_by_name(client, project_id, config.studio_name)
    if config.studio_auto_discover_free:
        instance = find_running_free_studio_instance(client, project_id)
        if instance is None:
            return None
        studio_id = str(getattr(instance, "cloud_space_id", "") or "")
        return next((studio for studio in list_studios(client, project_id) if str(getattr(studio, "id", "") or "") == studio_id), None)
    return None


def resolve_studio_instance(client, project_id: str, studio_id: str):
    for instance in list_studio_instances(client, project_id):
        if str(getattr(instance, "cloud_space_id", "") or "") == studio_id:
            return instance
    return None


def create_studio(client, project_id: str, config: LightningStudioConfig):
    desired_name = config.studio_name or f"{config.run.app_name}-studio"
    body = ProjectIdCloudspacesBody(
        name=desired_name,
        display_name=desired_name,
        cluster_id=config.studio_cluster_id,
        compute_name=config.studio_compute_name,
        disk_size=str(config.studio_disk_size_gb),
        can_download_source_code=True,
        spot=False,
    )
    return client.cloud_space_service_create_cloud_space(project_id=project_id, body=body)


def ensure_studio_exists(client, project_id: str, config: LightningStudioConfig, *, allow_create: bool):
    studio = resolve_studio(client, project_id, config)
    if studio is not None:
        return studio
    if not allow_create:
        raise RuntimeError("No matching Lightning Studio was found.")
    return create_studio(client, project_id, config)


def ensure_studio_running(client, project_id: str, studio, config: LightningStudioConfig, *, timeout_seconds: int = 300):
    studio_id = str(getattr(studio, "id", "") or "")
    instance = resolve_studio_instance(client, project_id, studio_id)
    if instance is not None and str(getattr(instance, "phase", "") or "") == RUNNING_PHASE:
        return instance

    body = IdCodeconfigBody(
        auto_start_ports=list(config.studio_auto_start_ports) or None,
        compute_config=studio_compute_config(config),
        disable_auto_shutdown=False,
        idle_shutdown_seconds=0,
        ide=config.studio_ide,
    )
    client.cloud_space_service_update_cloud_space_instance_config(body=body, project_id=project_id, id=studio_id)
    client.cloud_space_service_start_cloud_space_instance(
        body=IdStartBody(compute_config=studio_compute_config(config)),
        project_id=project_id,
        id=studio_id,
    )

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        instance = resolve_studio_instance(client, project_id, studio_id)
        if instance is not None and str(getattr(instance, "phase", "") or "") == RUNNING_PHASE:
            return instance
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting for Studio {studio_id} to reach RUNNING.")


def build_bootstrap_command(config: LightningStudioConfig) -> str:
    repo_path = studio_repo_path(config)
    requirements_file = shlex.quote(config.run.work_requirements_file)

    script = "\n".join(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(repo_path)}",
            "rm -rf .venv",
            "python -m venv .venv",
            "source .venv/bin/activate",
            "python -m pip install --upgrade pip",
            f"pip install -r {requirements_file}",
            "pip uninstall -y torchvision torchaudio >/dev/null 2>&1 || true",
        ]
    )
    return f"bash -lc {shlex.quote(script)}"


def build_session_command(config: LightningStudioConfig) -> str:
    tracked_args = " ".join(f"--tracked-path {shlex.quote(path)}" for path in config.run.tracked_paths)
    exports = [f"export {key}={shlex.quote(value)}" for key, value in config.run.app_env.items()]
    checkpoint_dir = studio_checkpoint_path(config)
    script_lines = [
        "set -euo pipefail",
        *exports,
        f"cd {shlex.quote(studio_repo_path(config))}",
        f"mkdir -p {shlex.quote(checkpoint_dir)}",
        "if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi",
        "python scripts/lightning_checkpoint_runner.py "
        f"--shell-command {shlex.quote(config.run.command)} "
        "--workdir . "
        f"--checkpoint-dir {shlex.quote(checkpoint_dir)} "
        f"--save-every-seconds {config.run.save_every_seconds} "
        f"--grace-period-seconds {config.run.grace_period_seconds} "
        "--shutdown-exit-code 75 "
        f"{tracked_args}",
    ]
    return f"bash -lc {shlex.quote(chr(10).join(script_lines))}"


def execute_studio_command(client, project_id: str, studio_id: str, *, command: str, session_name: str, detached: bool):
    return client.cloud_space_service_execute_command_in_cloud_space(
        body=IdExecuteBody(command=command, detached=detached, session_name=session_name),
        project_id=project_id,
        id=studio_id,
    )


def get_session_status(client, project_id: str, studio_id: str, session_name: str) -> dict[str, Any] | None:
    try:
        response = client.cloud_space_service_get_long_running_command_in_cloud_space(
            project_id=project_id,
            id=studio_id,
            session=session_name,
        )
    except ApiException as exc:
        if exc.status == 404:
            return None
        body_text = exc.body.decode("utf-8", errors="ignore") if isinstance(exc.body, (bytes, bytearray)) else str(exc.body)
        if exc.status == 500 and "no running instances found" in body_text.lower():
            return None
        raise

    payload = response.to_dict() if hasattr(response, "to_dict") else {"exit_code": getattr(response, "exit_code", None), "output": getattr(response, "output", "")}
    exit_code = int(payload.get("exit_code", -1))
    if exit_code == -1:
        state = "running"
    elif exit_code == 0:
        state = "completed"
    else:
        state = "failed"
    payload["state"] = state
    return payload


def wait_for_session_status(
    client,
    project_id: str,
    studio_id: str,
    session_name: str,
    *,
    timeout_seconds: int = 30,
    poll_seconds: int = 5,
) -> dict[str, Any] | None:
    deadline = time.time() + timeout_seconds
    last_status = None
    first_poll = True
    while time.time() < deadline:
        if first_poll:
            time.sleep(min(poll_seconds, max(timeout_seconds, 1)))
            first_poll = False
        status = get_session_status(client, project_id, studio_id, session_name)
        if status is not None:
            last_status = status
            if status["state"] == "running":
                return status
        time.sleep(poll_seconds)
    return last_status


def studio_status_report(*, studio, instance, session_status: dict[str, Any] | None, project) -> dict[str, Any]:
    return {
        "project_id": project.project_id,
        "project_name": project.name,
        "studio": json_safe(studio.to_dict() if hasattr(studio, "to_dict") else studio),
        "instance": json_safe(instance.to_dict() if hasattr(instance, "to_dict") else instance),
        "session": json_safe(session_status),
    }
