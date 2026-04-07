from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path


QP_SRC_DIR = Path(__file__).resolve().parent / "quant_platform" / "src"
if str(QP_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(QP_SRC_DIR))

from lightning_cloud_utils import ensure_auth_env, get_client_and_project, json_safe, set_process_env
from lightning_cloud.openapi import IdStartBody


RUNNING_PHASE = "CLOUD_SPACE_INSTANCE_STATE_RUNNING"
HOST_RE = re.compile(r"https?://([^/]+)")


def _target_tokens(url: str) -> list[str]:
    match = HOST_RE.search(url or "")
    if not match:
        return []
    host = match.group(1).strip().lower()
    subdomain = host.split(".")[0]
    tokens = {host, subdomain}
    if "-" in subdomain:
        tokens.add(subdomain.split("-", 1)[1])
    return [token for token in tokens if token]


def _instance_payload(instance) -> dict:
    if hasattr(instance, "to_dict"):
        return instance.to_dict()
    return json_safe(instance)


def _match_instance(instances, tokens: list[str]):
    lowered = [token.lower() for token in tokens if token]
    for instance in instances:
        payload = _instance_payload(instance)
        haystack = json.dumps(payload, sort_keys=True).lower()
        if any(token in haystack for token in lowered):
            return instance, payload
    return None, None


def main() -> None:
    inference_url = str(os.getenv("TRAINED_MODEL_INFERENCE_URL") or "").strip()
    if not inference_url:
        raise SystemExit("TRAINED_MODEL_INFERENCE_URL is not set.")

    env = ensure_auth_env(dict(os.environ))
    os.environ.update(env)
    set_process_env(env)
    client, project = get_client_and_project()

    instances_response = client.cloud_space_service_list_cloud_space_instances(project_id=project.project_id)
    instances = list(getattr(instances_response, "cloudspace_instances", []) or [])
    target_tokens = _target_tokens(inference_url)
    instance, payload = _match_instance(instances, target_tokens)

    if instance is None:
        raise RuntimeError(
            "Could not match TRAINED_MODEL_INFERENCE_URL to any Lightning CloudSpace instance. "
            f"tokens={target_tokens} available={json.dumps([_instance_payload(item) for item in instances], default=str)[:4000]}"
        )

    cloud_space_id = str(
        getattr(instance, "cloud_space_id", "")
        or payload.get("cloud_space_id")
        or payload.get("id")
        or ""
    ).strip()
    phase = str(getattr(instance, "phase", "") or payload.get("phase") or "").strip()

    report = {
        "project_id": project.project_id,
        "project_name": project.name,
        "target_tokens": target_tokens,
        "matched_cloud_space_id": cloud_space_id,
        "matched_phase": phase,
        "matched_instance": payload,
    }

    if phase == RUNNING_PHASE:
        print(json.dumps({"ok": True, "action": "already_running", **report}, indent=2, default=str))
        return

    if not cloud_space_id:
        raise RuntimeError(f"Matched CloudSpace instance did not expose a cloud_space_id: {json.dumps(report, default=str)[:4000]}")

    client.cloud_space_service_start_cloud_space_instance(
        body=IdStartBody(),
        project_id=project.project_id,
        id=cloud_space_id,
    )

    deadline = time.time() + 600
    while time.time() < deadline:
        instances_response = client.cloud_space_service_list_cloud_space_instances(project_id=project.project_id)
        instances = list(getattr(instances_response, "cloudspace_instances", []) or [])
        instance, payload = _match_instance(instances, target_tokens)
        if instance is None:
            time.sleep(5)
            continue
        phase = str(getattr(instance, "phase", "") or payload.get("phase") or "").strip()
        if phase == RUNNING_PHASE:
            print(json.dumps({"ok": True, "action": "started", **report, "matched_phase": phase, "matched_instance": payload}, indent=2, default=str))
            return
        time.sleep(5)

    raise TimeoutError(f"Timed out waiting for Lightning CloudSpace to reach RUNNING for tokens={target_tokens}")


if __name__ == "__main__":
    main()
