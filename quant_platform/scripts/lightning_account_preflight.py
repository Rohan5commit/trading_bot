from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lightning_cloud.rest_client import create_swagger_client
from lightning_cloud.openapi import BillingServiceApi

from lightning_cloud_utils import DEFAULT_AUTH_URL, _request_json, ensure_auth_env, get_client_and_project, json_safe, set_process_env


IMPORTANT_FEATURE_FLAGS = (
    "restartableJobs",
    "persistentDisk",
    "cloudspaceSchedules",
    "driveV2",
)


def _login_and_fetch_user(env: dict[str, str]) -> dict:
    login_payload = _request_json(
        f"{DEFAULT_AUTH_URL}/v1/auth/login",
        method="POST",
        payload={"username": env["LIGHTNING_USERNAME"], "apiKey": env["LIGHTNING_API_KEY"]},
    )
    token = str(login_payload.get("token") or "").strip()
    if not token:
        raise RuntimeError("Lightning auth did not return a token during preflight.")
    return _request_json(
        f"{DEFAULT_AUTH_URL}/v1/auth/user",
        headers={"Authorization": f"Bearer {token}"},
    )


def build_preflight_report() -> dict:
    env = ensure_auth_env()
    set_process_env(env)

    user_payload = _login_and_fetch_user(env)
    client, project = get_client_and_project()
    instances = client.cloud_space_service_list_cloud_space_instances(project_id=project.project_id)

    billing_api = BillingServiceApi(create_swagger_client())
    user_balance = billing_api.billing_service_get_user_balance()
    project_balance = billing_api.billing_service_get_project_balance(project_id=project.project_id)

    features = user_payload.get("features") or {}
    feature_flags = {name: bool(features.get(name, False)) for name in IMPORTANT_FEATURE_FLAGS}

    report = {
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "username": user_payload.get("username"),
        "user_id": user_payload.get("id"),
        "project_id": project.project_id,
        "project_name": project.name,
        "completed_signup": bool((user_payload.get("status") or {}).get("completedSignup")),
        "can_start_free_cloud_space": bool(getattr(instances, "can_start_free_cloud_space", False)),
        "cloudspace_instance_count": len(list(getattr(instances, "cloudspace_instances", []) or [])),
        "user_balance": float(getattr(user_balance, "balance", 0.0) or 0.0),
        "project_balance": float(getattr(project_balance, "balance", 0.0) or 0.0),
        "feature_flags": feature_flags,
        "recommended_runner": "lightning"
        if bool(getattr(instances, "can_start_free_cloud_space", False))
        else "github-actions",
        "blockers": [],
    }

    blockers: list[str] = report["blockers"]
    if not report["can_start_free_cloud_space"]:
        blockers.append("Lightning reports can_start_free_cloud_space=false for this account.")
    if report["user_balance"] <= 0.0 and report["project_balance"] <= 0.0:
        blockers.append("Both Lightning user and project balances are exhausted.")
    if not feature_flags["restartableJobs"]:
        blockers.append("Lightning account feature flag restartableJobs=false.")
    if not feature_flags["persistentDisk"]:
        blockers.append("Lightning account feature flag persistentDisk=false.")
    if not feature_flags["cloudspaceSchedules"]:
        blockers.append("Lightning account feature flag cloudspaceSchedules=false.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", help="Optional path to write the preflight JSON report.")
    parser.add_argument(
        "--require-free-cloud-space",
        action="store_true",
        help="Exit non-zero when Lightning cannot currently start a free cloud space.",
    )
    args = parser.parse_args()

    report = build_preflight_report()
    payload = json.dumps(json_safe(report), indent=2)
    print(payload)

    if args.json_out:
        Path(args.json_out).write_text(payload + "\n")

    if args.require_free_cloud_space and report["blockers"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
