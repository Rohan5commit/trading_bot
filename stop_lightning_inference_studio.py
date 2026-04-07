from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


ROOT_DIR = Path(__file__).resolve().parent
QP_SRC_DIR = ROOT_DIR / "quant_platform" / "src"
if str(QP_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(QP_SRC_DIR))

from lightning_cloud_utils import ensure_auth_env, json_safe, set_process_env  # noqa: E402
from lightning_studio_utils import ensure_studio_auth_env, get_client_and_project, load_studio_config, resolve_studio, resolve_studio_instance  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="quant_platform/configs/lightning_inference_studio.yaml")
    args = parser.parse_args()

    auth_env = ensure_auth_env()
    set_process_env(auth_env)
    ensure_studio_auth_env()
    client, project = get_client_and_project()
    config = load_studio_config(args.config)
    studio = resolve_studio(client, project.project_id, config)
    if studio is None:
        print(json.dumps({"ok": True, "action": "no_studio_found"}, indent=2))
        return
    studio_id = str(getattr(studio, "id", "") or "").strip()
    instance = resolve_studio_instance(client, project.project_id, studio_id)
    if instance is None:
        print(json.dumps({"ok": True, "action": "no_instance_found", "studio_id": studio_id}, indent=2))
        return
    client.cloud_space_service_stop_cloud_space_instance(project_id=project.project_id, id=studio_id)
    deadline = time.time() + 300
    final_phase = ""
    while time.time() < deadline:
        instance = resolve_studio_instance(client, project.project_id, studio_id)
        payload = json_safe(instance.to_dict() if hasattr(instance, "to_dict") else instance)
        final_phase = str(getattr(instance, "phase", "") or (payload or {}).get("phase") or "").strip()
        if final_phase in {"", "CLOUD_SPACE_INSTANCE_STATE_STOPPED", "CLOUD_SPACE_INSTANCE_STATE_DELETED"}:
            print(json.dumps({"ok": True, "studio_id": studio_id, "phase": final_phase or "stopped"}, indent=2))
            return
        time.sleep(5)
    raise TimeoutError(json.dumps({"ok": False, "studio_id": studio_id, "phase": final_phase}, indent=2))


if __name__ == "__main__":
    main()
