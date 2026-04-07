from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Unsupported boolean value: {value}")


def _format_env_items(items: list[str]) -> dict[str, str]:
    env_vars: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Environment variables must use KEY=VALUE format: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Environment variable name cannot be empty: {item}")
        env_vars[key] = value
    return env_vars


def _list_instances_for_cloudspace(client: Any, *, project_id: str, cloudspace_id: str) -> list[Any]:
    method = client.lightningapp_instance_service_list_lightningapp_instances
    try:
        response = method(project_id=project_id, cloud_space_id=cloudspace_id)
    except TypeError:
        response = method(project_id=project_id, app_id=cloudspace_id)
    return list(getattr(response, "lightningapps", []) or [])


def _patch_cloud_runtime(*, force_running: bool) -> None:
    try:
        from lightning.app.runners.cloud import CloudRuntime
    except ModuleNotFoundError:
        from lightning_app.runners.cloud import CloudRuntime

    def _resolve_existing_run_instance(self, cluster_id, project_id, existing_cloudspaces):
        existing_cloudspace = None
        existing_run_instance = None

        if cluster_id is not None:
            for cloudspace in existing_cloudspaces:
                run_instances = _list_instances_for_cloudspace(
                    self.backend.client,
                    project_id=project_id,
                    cloudspace_id=cloudspace.id,
                )
                if run_instances and getattr(getattr(run_instances[0], "spec", None), "cluster_id", None) == cluster_id:
                    existing_cloudspace = cloudspace
                    existing_run_instance = run_instances[0]
                    break

        return existing_cloudspace, existing_run_instance

    CloudRuntime._resolve_existing_run_instance = _resolve_existing_run_instance
    if force_running:
        # The current client treats zero balance as "must start stopped", even for the public free CPU path.
        # Creating the app directly in RUNNING avoids a later restart mutation that currently fails for Drive-backed works.
        CloudRuntime._resolve_needs_credits = staticmethod(lambda project: False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint", default="lightning_cloud_app.py")
    parser.add_argument("--name", required=True)
    parser.add_argument("--config-env-var", default="LIGHTNING_RUN_CONFIG")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--without-server", action="store_true")
    parser.add_argument("--blocking", default="false")
    parser.add_argument("--open-ui", default="false")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--force-running", action="store_true")
    parser.add_argument("--env", action="append", default=[])
    args = parser.parse_args()

    env_vars = _format_env_items(args.env)
    env_vars[args.config_env_var] = args.config_path
    os.environ.update(env_vars)

    _patch_cloud_runtime(force_running=args.force_running)

    try:
        from lightning.app.runners.runtime import dispatch
        from lightning.app.runners.runtime_type import RuntimeType
    except ModuleNotFoundError:
        from lightning_app.runners.runtime import dispatch
        from lightning_app.runners.runtime_type import RuntimeType

    entrypoint = Path(args.entrypoint)
    if not entrypoint.is_absolute():
        entrypoint = (ROOT_DIR / entrypoint).resolve()

    dispatch(
        entrypoint,
        RuntimeType.CLOUD,
        start_server=not args.without_server,
        no_cache=args.no_cache,
        blocking=_parse_bool(args.blocking),
        open_ui=_parse_bool(args.open_ui),
        name=args.name,
        env_vars=env_vars,
        secrets={},
    )


if __name__ == "__main__":
    main()
