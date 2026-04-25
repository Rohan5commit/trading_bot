from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
REPO_ROOT = ROOT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _load_router_config() -> dict:
    config_path = REPO_ROOT / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml

        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return dict(((payload.get("ai_trading") or {}).get("runtime_router") or {}))


def choose_runtime() -> dict:
    router_cfg = _load_router_config()
    min_balance = float(
        os.getenv("AI_FULL_MODEL_MIN_BALANCE")
        or os.getenv("LIGHTNING_LARGE_CPU_MIN_BALANCE")
        or router_cfg.get("min_project_balance_for_large_cpu")
        or "5.0"
    )
    preferred_compute = str(
        os.getenv("LIGHTNING_LARGE_COMPUTE_NAME")
        or router_cfg.get("preferred_large_compute_name")
        or "cpu-8"
    ).strip() or "cpu-8"
    preferred_disk_gb = int(
        float(
            os.getenv("LIGHTNING_LARGE_DISK_GB")
            or router_cfg.get("preferred_large_disk_gb")
            or "80"
        )
    )
    fallback_mode = str(os.getenv("AI_FALLBACK_RUNTIME_MODE") or "distilled_local").strip() or "distilled_local"

    try:
        from lightning_account_preflight import build_preflight_report

        report = build_preflight_report()
        user_balance = float(report.get("user_balance", 0.0) or 0.0)
        project_balance = float(report.get("project_balance", 0.0) or 0.0)
        available_balance = max(user_balance, project_balance)
        feature_flags = dict(report.get("feature_flags") or {})
        full_runtime_blockers = []
        if not bool(report.get("completed_signup", True)):
            full_runtime_blockers.append("lightning_signup_incomplete")
        if not bool(feature_flags.get("persistentDisk", True)):
            full_runtime_blockers.append("lightning_persistent_disk_unavailable")

        if available_balance >= min_balance and not full_runtime_blockers:
            return {
                "runtime_mode": "lightning_full",
                "selected_backend": "trained_model_http",
                "selected_compute_name": preferred_compute,
                "selected_disk_gb": preferred_disk_gb,
                "reason": f"lightning_balance_{available_balance:.2f}_meets_threshold_{min_balance:.2f}",
                "preflight": report,
            }
        reason = f"lightning_balance_{available_balance:.2f}_below_threshold_{min_balance:.2f}"
        if full_runtime_blockers:
            reason = "full_runtime_blocked:" + ",".join(full_runtime_blockers)
        return {
            "runtime_mode": fallback_mode,
            "selected_backend": "distilled_local",
            "selected_compute_name": "",
            "selected_disk_gb": 0,
            "reason": reason,
            "preflight": report,
        }
    except Exception as exc:
        return {
            "runtime_mode": fallback_mode,
            "selected_backend": "distilled_local",
            "selected_compute_name": "",
            "selected_disk_gb": 0,
            "reason": f"lightning_preflight_error:{exc}",
            "preflight": {"error": str(exc)},
        }


def write_github_output(path: str, payload: dict) -> None:
    if not path:
        return
    lines = {
        "runtime_mode": payload.get("runtime_mode", "distilled_local"),
        "selected_backend": payload.get("selected_backend", "distilled_local"),
        "selected_compute_name": payload.get("selected_compute_name", ""),
        "selected_disk_gb": str(payload.get("selected_disk_gb", 0)),
        "reason": payload.get("reason", ""),
    }
    with open(path, "a", encoding="utf-8") as handle:
        for key, value in lines.items():
            text = str(value or "")
            if "\n" in text or "\r" in text:
                handle.write(f"{key}<<__CODEx_EOF__\n{text}\n__CODEx_EOF__\n")
            else:
                handle.write(f"{key}={text}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", help="Optional path to write the runtime plan JSON.")
    parser.add_argument("--github-output", help="Optional GitHub Actions output file path.")
    args = parser.parse_args()

    payload = choose_runtime()
    text = json.dumps(payload, indent=2)
    print(text)

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(text + "\n", encoding="utf-8")
    if args.github_output:
        write_github_output(args.github_output, payload)


if __name__ == "__main__":
    main()
