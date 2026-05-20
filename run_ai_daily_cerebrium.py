from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _first_non_empty(*values: str | None) -> str:
    for value in values:
        if value and str(value).strip():
            return str(value).strip()
    return ""


def _run_daily_job(env: dict[str, str]) -> int:
    proc = subprocess.run([sys.executable, "main.py", "daily_job"], env=env)
    return int(proc.returncode)


def _run_ai_smoke(env: dict[str, str]) -> int:
    smoke_env = env.copy()
    smoke_env["AI_SMOKE_USE_STATIC"] = "1"
    proc = subprocess.run([sys.executable, "run_ai_trading_smoke.py"], env=smoke_env)
    return int(proc.returncode)


def main() -> None:
    base_env = os.environ.copy()
    base_env["DISABLE_CORE_TRADING"] = "1"
    base_env["AI_RUNTIME_MODE"] = "full"
    base_env["AI_PRIMARY_BACKEND"] = "cerebrium"
    base_env["AI_ROUTER_REASON"] = "cerebrium_primary_direct"

    resolved_url = _first_non_empty(
        base_env.get("CEREBRIUM_TRAINED_MODEL_URL"),
        base_env.get("CEREBRIUM_INFERENCE_URL"),
        base_env.get("TRAINED_MODEL_INFERENCE_URL"),
    )
    if not resolved_url:
        raise SystemExit("No inference URL configured in CEREBRIUM_TRAINED_MODEL_URL/CEREBRIUM_INFERENCE_URL/TRAINED_MODEL_INFERENCE_URL.")

    base_env["CEREBRIUM_INFERENCE_URL"] = resolved_url
    base_env["TRAINED_MODEL_INFERENCE_URL"] = resolved_url

    run_meta = {
        "selected_backend": "cerebrium",
        "reason": "cerebrium_primary_direct",
        "resolved_inference_url": resolved_url,
    }

    rc = _run_daily_job(base_env)
    if rc == 0:
        run_meta["ok"] = True
        run_meta["path"] = "daily_job"
        Path("results").mkdir(parents=True, exist_ok=True)
        Path("results/ai_runtime_plan.json").write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")
        return

    # Retry once on the same Cerebrium backend to absorb transient cold-start or readiness races.
    retry_rc = _run_daily_job(base_env)
    if retry_rc == 0:
        run_meta["ok"] = True
        run_meta["path"] = "daily_job_retry"
        run_meta["retry_used"] = True
        Path("results").mkdir(parents=True, exist_ok=True)
        Path("results/ai_runtime_plan.json").write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")
        return

    lightning_url = _first_non_empty(base_env.get("LIGHTNING_INFERENCE_URL"))
    if not lightning_url:
        # Optional continuity mode: keep AI runtime alive even if market-data ingestion breaks.
        allow_smoke_fallback = str(os.getenv("AI_DAILY_ALLOW_SMOKE_FALLBACK", "1")).strip().lower() in {"1", "true", "yes", "on"}
        if allow_smoke_fallback:
            smoke_rc = _run_ai_smoke(base_env)
            run_meta["ok"] = smoke_rc == 0
            run_meta["fallback_used"] = True
            run_meta["fallback_backend"] = "cerebrium_smoke_static"
            run_meta["fallback_inference_url"] = resolved_url
            run_meta["path"] = "ai_smoke_static"
            run_meta["daily_job_rc"] = rc
            run_meta["daily_job_retry_rc"] = retry_rc
            Path("results").mkdir(parents=True, exist_ok=True)
            Path("results/ai_runtime_plan.json").write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")
            raise SystemExit(smoke_rc)

        run_meta["ok"] = False
        run_meta["fallback_used"] = False
        run_meta["daily_job_rc"] = rc
        run_meta["daily_job_retry_rc"] = retry_rc
        Path("results").mkdir(parents=True, exist_ok=True)
        Path("results/ai_runtime_plan.json").write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(retry_rc)

    fallback_env = base_env.copy()
    fallback_env["AI_PRIMARY_BACKEND"] = "http"
    fallback_env["AI_ROUTER_REASON"] = "lightning_after_cerebrium_failure"
    fallback_env["CEREBRIUM_INFERENCE_URL"] = lightning_url
    fallback_env["TRAINED_MODEL_INFERENCE_URL"] = lightning_url

    fallback_rc = _run_daily_job(fallback_env)
    run_meta["fallback_used"] = True
    run_meta["fallback_backend"] = "lightning_http"
    run_meta["fallback_inference_url"] = lightning_url
    run_meta["ok"] = fallback_rc == 0
    run_meta["daily_job_rc"] = rc
    run_meta["daily_job_retry_rc"] = retry_rc
    run_meta["path"] = "lightning_fallback_daily_job"
    Path("results").mkdir(parents=True, exist_ok=True)
    Path("results/ai_runtime_plan.json").write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(fallback_rc)


if __name__ == "__main__":
    main()
