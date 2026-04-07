from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parent
QP_SRC_DIR = ROOT_DIR / "quant_platform" / "src"
if str(QP_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(QP_SRC_DIR))

from lightning_cloud_utils import (  # noqa: E402
    ACTIVE_PHASES,
    delete_matching_apps,
    ensure_auth_env,
    get_client_and_project,
    set_process_env,
    wait_for_app_removal,
)


DEFAULT_APP_NAME = "trading-bot-lightning-inference"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-name", default=DEFAULT_APP_NAME)
    args = parser.parse_args()

    auth_env = ensure_auth_env()
    set_process_env(auth_env)
    client, project = get_client_and_project()
    deleted = delete_matching_apps(client, project.project_id, args.app_name, phases=ACTIVE_PHASES)
    wait_for_app_removal(client, project.project_id, args.app_name, timeout_seconds=300, poll_seconds=10)
    print(json.dumps({"ok": True, "deleted_app_ids": deleted}, indent=2))


if __name__ == "__main__":
    main()
