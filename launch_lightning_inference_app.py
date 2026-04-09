from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any

import requests
from lightning_cloud.openapi import IdStartBody, V1UserRequestedComputeConfig
from lightning_cloud.openapi.models.appinstances_id_body import AppinstancesIdBody
from lightning_cloud.openapi.models.v1_lightningapp_instance_spec import V1LightningappInstanceSpec


ROOT_DIR = Path(__file__).resolve().parent
QP_SRC_DIR = ROOT_DIR / "quant_platform" / "src"
if str(QP_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(QP_SRC_DIR))

from lightning_cloud_utils import (  # noqa: E402
    ACTIVE_PHASES,
    TERMINAL_PHASES,
    collect_logs_text,
    delete_matching_apps,
    ensure_auth_env,
    find_app_by_name,
    get_client_and_project,
    json_safe,
    phase_name,
    set_process_env,
    wait_for_app,
    wait_for_app_removal,
)

from deploy_lightning_inference import _collect_env, _patch_lightning_dispatch_compat  # noqa: E402
try:  # noqa: E402
    from lightning.app.runners.runtime import dispatch
    from lightning.app.runners.runtime_type import RuntimeType
except ModuleNotFoundError:  # noqa: E402
    from lightning_app.runners.runtime import dispatch
    from lightning_app.runners.runtime_type import RuntimeType


DEFAULT_APP_NAME = "trading-bot-lightning-inference"
URL_RE = re.compile(r"LIGHTNING_INFERENCE_URL=(https?://\S+)")


def _resolved_project_id() -> str | None:
    for key in ("LIGHTNING_CLOUD_PROJECT_ID", "LIGHTNING_PROJECT_ID"):
        value = str(os.getenv(key) or "").strip()
        if value:
            return value
    return None


def _default_entrypoint() -> Path:
    bundle_entrypoint = ROOT_DIR / "lightning_inference_bundle" / "lightning_trained_model_app.py"
    if bundle_entrypoint.exists():
        return bundle_entrypoint
    return ROOT_DIR / "lightning_trained_model_app.py"


def _app_payload(app: Any) -> dict[str, Any]:
    if app is None:
        return {}
    if hasattr(app, "to_dict"):
        return json_safe(app.to_dict())
    return json_safe(app)


def _request_headers() -> dict[str, str]:
    headers = {"Accept": "application/json"}
    api_key = str(os.getenv("TRAINED_MODEL_API_KEY") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _poll_health(base_url: str, *, timeout_seconds: int = 30) -> dict[str, Any]:
    response = requests.get(
        base_url.rstrip("/") + "/health",
        headers=_request_headers(),
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise RuntimeError(f"Health check returned unexpected payload: {payload}")
    return payload


def _latest_app(client, project_id: str, app_name: str):
    latest = find_app_by_name(client, project_id, app_name)
    if latest is None:
        return None
    app_id = str(getattr(latest, "id", "") or "").strip()
    if not app_id:
        return latest
    return client.lightningapp_instance_service_get_lightningapp_instance(project_id=project_id, id=app_id)


def _work_candidate_urls(client, project_id: str, app_id: str) -> list[str]:
    urls: list[str] = []
    try:
        response = client.lightningwork_service_list_lightningwork(project_id=project_id, app_id=app_id)
    except Exception:  # noqa: BLE001
        return urls
    works = list(getattr(response, "lightningworks", []) or [])
    for work in works:
        spec = getattr(work, "spec", None)
        network_config = list(getattr(spec, "network_config", []) or [])
        for item in network_config:
            host = str(getattr(item, "host", "") or "").strip().rstrip("/")
            if not host:
                continue
            if not host.startswith("http://") and not host.startswith("https://"):
                host = f"https://{host}"
            if host not in urls:
                urls.append(host)
    return urls


def _candidate_urls(client, project_id: str, app, logs_text: str) -> list[str]:
    urls: list[str] = []
    for match in URL_RE.findall(logs_text or ""):
        match = str(match).strip().rstrip("/")
        if match and match not in urls:
            urls.append(match)
    app_id = str(getattr(app, "id", "") or "").strip()
    if app_id:
        for candidate in _work_candidate_urls(client, project_id, app_id):
            if candidate not in urls:
                urls.append(candidate)
    status_url = str(getattr(getattr(app, "status", None), "url", "") or "").strip().rstrip("/")
    if status_url and status_url not in urls:
        urls.append(status_url)
    return urls


def _start_stopped_app(client, project_id: str, app: Any) -> Any:
    cloud_space_id = str(getattr(getattr(app, "spec", None), "cloud_space_id", "") or "").strip()
    if not cloud_space_id:
        return app

    compute_name = str(os.getenv("LIGHTNING_INFERENCE_COMPUTE_NAME") or "cpu-4").strip() or "cpu-4"
    disk_size_gb = int(str(os.getenv("LIGHTNING_INFERENCE_DISK_GB") or "80").strip() or "80")

    client.cloud_space_service_start_cloud_space_instance(
        project_id=project_id,
        id=cloud_space_id,
        body=IdStartBody(
            compute_config=V1UserRequestedComputeConfig(
                name=compute_name,
                disk_size=disk_size_gb,
                spot=False,
                same_compute_on_resume=False,
            )
        ),
    )

    spec_dict = app.spec.to_dict()
    spec_dict["desired_state"] = "LIGHTNINGAPP_INSTANCE_STATE_RUNNING"
    update_body = AppinstancesIdBody(
        display_name=getattr(app, "display_name", None),
        name=getattr(app, "name", None),
        spec=V1LightningappInstanceSpec(**spec_dict),
    )
    client.lightningapp_instance_service_update_lightningapp_instance(
        body=update_body,
        project_id=project_id,
        id=str(getattr(app, "id", "")),
    )
    return client.lightningapp_instance_service_get_lightningapp_instance(project_id=project_id, id=str(getattr(app, "id", "")))


def _wait_for_inference_url(
    client,
    project_id: str,
    app_name: str,
    *,
    timeout_seconds: int,
    require_health: bool,
    dispatch_thread: threading.Thread | None = None,
    dispatch_state: dict[str, Any] | None = None,
) -> tuple[Any, str, dict[str, Any] | None, str]:
    deadline = time.time() + timeout_seconds
    last_error = "waiting for Lightning app URL"
    last_logs = ""
    attempted_stopped_recovery = False
    while time.time() < deadline:
        if dispatch_thread is not None and dispatch_state is not None and not dispatch_thread.is_alive():
            exc_text = str(dispatch_state.get("error") or "").strip()
            if exc_text:
                raise RuntimeError(
                    json.dumps(
                        {
                            "error": "Lightning dispatch thread failed before inference URL discovery",
                            "dispatch_error": exc_text,
                            "app_name": app_name,
                        },
                        indent=2,
                    )
                )
        app = _latest_app(client, project_id, app_name)
        if app is None:
            last_error = "Lightning app not found yet"
            time.sleep(10)
            continue
        base_candidates = _candidate_urls(client, project_id, app, "")
        for candidate in base_candidates:
            if not require_health:
                return app, candidate, None, last_logs
            try:
                health = _poll_health(candidate)
                return app, candidate, health, last_logs
            except Exception as exc:  # noqa: BLE001
                last_error = f"{candidate}: {exc}"
        logs_text = ""
        try:
            logs_text = collect_logs_text(client, project_id, str(getattr(app, "id", "")), max_pages=4)
        except Exception as exc:  # noqa: BLE001
            last_logs = f"[log-collection-error] {exc}"
        else:
            last_logs = logs_text
        for candidate in _candidate_urls(client, project_id, app, logs_text):
            if candidate in base_candidates:
                continue
            if not require_health:
                return app, candidate, None, last_logs
            try:
                health = _poll_health(candidate)
                return app, candidate, health, last_logs
            except Exception as exc:  # noqa: BLE001
                last_error = f"{candidate}: {exc}"
        phase = phase_name(app)
        if phase == "LIGHTNINGAPP_INSTANCE_STATE_STOPPED" and not attempted_stopped_recovery:
            attempted_stopped_recovery = True
            try:
                app = _start_stopped_app(client, project_id, app)
                time.sleep(15)
                continue
            except Exception as exc:  # noqa: BLE001
                last_error = f"failed to recover stopped app: {exc}"
        if phase in TERMINAL_PHASES:
            raise RuntimeError(
                json.dumps(
                    {
                        "error": "Lightning app entered terminal phase before becoming healthy",
                        "phase": phase,
                        "app": _app_payload(app),
                        "logs_tail": "\n".join((logs_text or "").splitlines()[-120:]),
                    },
                    indent=2,
                )
            )
        time.sleep(15)
    raise TimeoutError(
        json.dumps(
            {
                "error": f"Timed out waiting for Lightning inference app health: {last_error}",
                "app_name": app_name,
                "logs_tail": "\n".join((last_logs or "").splitlines()[-120:]),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-name", default=DEFAULT_APP_NAME)
    parser.add_argument("--entrypoint", default="")
    parser.add_argument("--status-out", default="")
    parser.add_argument("--replace-existing", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--require-health", action="store_true")
    args = parser.parse_args()

    auth_env = ensure_auth_env()
    set_process_env(auth_env)
    _patch_lightning_dispatch_compat()
    project_id = _resolved_project_id()
    client, project = get_client_and_project(project_id=project_id)

    if args.replace_existing:
        delete_matching_apps(client, project.project_id, args.app_name)
        wait_for_app_removal(client, project.project_id, args.app_name, timeout_seconds=300, poll_seconds=10)

    entrypoint = Path(args.entrypoint).expanduser().resolve() if args.entrypoint else _default_entrypoint()
    env_vars = _collect_env()
    dispatch_state: dict[str, Any] = {}

    def _dispatch_target() -> None:
        try:
            dispatch(
                entrypoint,
                RuntimeType.CLOUD,
                start_server=False,
                no_cache=False,
                blocking=False,
                open_ui=False,
                name=args.app_name,
                env_vars=env_vars,
                secrets={},
            )
        except Exception as exc:  # noqa: BLE001
            dispatch_state["error"] = repr(exc)

    dispatch_thread = threading.Thread(target=_dispatch_target, name="lightning-dispatch", daemon=True)
    dispatch_thread.start()

    app = wait_for_app(client, project.project_id, args.app_name, timeout_seconds=300, poll_seconds=10)
    phase = phase_name(app) if app else ""
    if app is None or phase not in ACTIVE_PHASES | TERMINAL_PHASES:
        raise RuntimeError(f"Lightning app {args.app_name} did not appear after dispatch.")

    app, inference_url, health, logs_text = _wait_for_inference_url(
        client,
        project.project_id,
        args.app_name,
        timeout_seconds=args.timeout_seconds,
        require_health=args.require_health,
        dispatch_thread=dispatch_thread,
        dispatch_state=dispatch_state,
    )
    payload = {
        "ok": True,
        "project_id": project.project_id,
        "project_name": project.name,
        "app_name": args.app_name,
        "app_id": getattr(app, "id", None),
        "phase": phase_name(app),
        "status_url": getattr(getattr(app, "status", None), "url", None),
        "inference_url": inference_url,
        "health": health,
        "logs_tail": "\n".join((logs_text or "").splitlines()[-40:]),
    }
    text = json.dumps(json_safe(payload), indent=2)
    if args.status_out:
        Path(args.status_out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
