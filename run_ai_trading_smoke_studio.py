from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent
QP_SRC_DIR = ROOT_DIR / "quant_platform" / "src"
if str(QP_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(QP_SRC_DIR))

from lightning_cloud_utils import ensure_auth_env, json_safe, set_process_env  # noqa: E402
from lightning_studio_utils import (  # noqa: E402
    ensure_studio_auth_env,
    execute_studio_command,
    get_client_and_project,
    load_studio_config,
    resolve_studio,
    resolve_studio_instance,
)


RESULT_BEGIN = "__AI_SMOKE_RESULT_BEGIN__"
RESULT_END = "__AI_SMOKE_RESULT_END__"
RESULT_PATH_PREFIX = "__AI_SMOKE_RESULT_PATH__="


def _resolved_project_id() -> str | None:
    for key in ("LIGHTNING_CLOUD_PROJECT_ID", "LIGHTNING_PROJECT_ID"):
        value = str(os.getenv(key) or "").strip()
        if value:
            return value
    return None


def _command_payload(result: Any) -> dict[str, Any]:
    if hasattr(result, "to_dict"):
        return json_safe(result.to_dict())
    return json_safe(result)


def _checked_execute(client, project_id: str, studio_id: str, *, command: str, session_name: str) -> dict[str, Any]:
    result = execute_studio_command(
        client,
        project_id,
        studio_id,
        command=command,
        session_name=session_name,
        detached=False,
    )
    payload = _command_payload(result)
    exit_code = payload.get("exit_code")
    if exit_code not in (0, None):
        raise RuntimeError(json.dumps(payload, indent=2))
    return payload


def _service_port(config) -> int:
    override = str(os.getenv("LIGHTNING_INFERENCE_PORT") or "").strip()
    if override:
        return int(override)
    command = str(config.run.command or "")
    parts = command.split("--port", 1)
    if len(parts) == 2:
        return int(parts[1].strip().split()[0])
    return 8000


def _build_smoke_command(config, *, service_port: int) -> str:
    repo_dir = Path(config.studio_root_dir.rstrip("/")) / config.studio_repo_dir
    exports = {
        **dict(config.run.app_env),
        "TRAINED_MODEL_INFERENCE_URL": f"http://127.0.0.1:{service_port}",
        "AI_SMOKE_USE_STATIC": str(os.getenv("AI_SMOKE_USE_STATIC", "1") or "1"),
        "AI_SMOKE_TICKERS": str(os.getenv("AI_SMOKE_TICKERS", "AAPL,TSLA,MSFT") or "AAPL,TSLA,MSFT"),
        "TRAINED_MODEL_READY_TIMEOUT_SECONDS": str(os.getenv("TRAINED_MODEL_READY_TIMEOUT_SECONDS", "1200") or "1200"),
        "TRAINED_MODEL_READY_POLL_SECONDS": str(os.getenv("TRAINED_MODEL_READY_POLL_SECONDS", "15") or "15"),
        "TRAINED_MODEL_TIMEOUT_SECONDS": str(os.getenv("TRAINED_MODEL_TIMEOUT_SECONDS", "900") or "900"),
        "TRAINED_MODEL_MAX_RETRIES": str(os.getenv("TRAINED_MODEL_MAX_RETRIES", "0") or "0"),
        "TRAINED_MODEL_BACKOFF_SECONDS": str(os.getenv("TRAINED_MODEL_BACKOFF_SECONDS", "10") or "10"),
    }
    export_lines = [f"export {key}={shlex.quote(str(value))}" for key, value in exports.items() if value is not None]
    script_lines = [
        "set -euo pipefail",
        *export_lines,
        f"cd {shlex.quote(str(repo_dir))}",
        "if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi",
        "python wait_for_trained_model.py",
        "python warm_trained_model.py",
        "python run_ai_trading_smoke.py",
        "latest=$(ls -1t results/ai_smoke_*.json | head -n1)",
        f"echo {shlex.quote(RESULT_PATH_PREFIX)}\"$latest\"",
        f"echo {shlex.quote(RESULT_BEGIN)}",
        'cat "$latest"',
        f"echo {shlex.quote(RESULT_END)}",
    ]
    return f"bash -lc {shlex.quote(chr(10).join(script_lines))}"


def _extract_result(output: str) -> tuple[str | None, dict[str, Any]]:
    text = str(output or "")
    result_path = None
    for line in text.splitlines():
        if line.startswith(RESULT_PATH_PREFIX):
            result_path = line.split("=", 1)[1].strip()
            break
    if RESULT_BEGIN not in text or RESULT_END not in text:
        raise RuntimeError("Studio smoke output did not include result markers.")
    fragment = text.split(RESULT_BEGIN, 1)[1].split(RESULT_END, 1)[0].strip()
    payload = json.loads(fragment)
    if not isinstance(payload, dict):
        raise RuntimeError("Studio smoke output JSON was not an object.")
    return result_path, payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="quant_platform/configs/lightning_inference_studio.yaml")
    parser.add_argument("--status-in", default="lightning-inference-status.json")
    parser.add_argument("--result-out", default="results/ai_smoke_lightning_studio.json")
    args = parser.parse_args()

    auth_env = ensure_auth_env()
    os.environ.update(auth_env)
    set_process_env(auth_env)
    ensure_studio_auth_env()
    client, project = get_client_and_project(project_id=_resolved_project_id())
    config = load_studio_config(args.config)
    studio = resolve_studio(client, project.project_id, config)
    if studio is None:
        raise RuntimeError("No matching Lightning Studio was found for AI smoke.")
    studio_id = str(getattr(studio, "id", "") or "").strip()
    if not studio_id:
        raise RuntimeError("Lightning Studio did not expose an id.")
    instance = resolve_studio_instance(client, project.project_id, studio_id)
    if instance is None:
        raise RuntimeError("Lightning Studio does not have an active instance.")

    payload = _checked_execute(
        client,
        project.project_id,
        studio_id,
        command=_build_smoke_command(config, service_port=_service_port(config)),
        session_name=f"{config.studio_session_name}-ai-smoke",
    )
    result_path, result_payload = _extract_result(str(payload.get("output") or ""))

    out_path = Path(args.result_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result_payload, indent=2) + "\n")

    summary = {
        "ok": True,
        "project_id": project.project_id,
        "project_name": project.name,
        "studio_id": studio_id,
        "studio_name": str(getattr(studio, "name", "") or ""),
        "studio_result_path": result_path,
        "local_result_path": str(out_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
