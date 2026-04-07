from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sys
import time
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
    get_session_status,
    load_studio_config,
    resolve_studio,
    resolve_studio_instance,
)


RESULT_BEGIN = "__AI_SMOKE_RESULT_BEGIN__"
RESULT_END = "__AI_SMOKE_RESULT_END__"
RESULT_PATH_PREFIX = "__AI_SMOKE_RESULT_PATH__="
RESULT_CACHE_PATH = "results/ai_smoke_studio_current.json"


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


def _launch_detached_session(
    client,
    project_id: str,
    studio_id: str,
    *,
    command: str,
    session_name: str,
    max_attempts: int = 3,
    retry_sleep_seconds: int = 5,
) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            result = execute_studio_command(
                client,
                project_id,
                studio_id,
                command=command,
                session_name=session_name,
                detached=True,
            )
            return _command_payload(result)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= max_attempts:
                raise
            time.sleep(retry_sleep_seconds)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Failed to launch detached Studio smoke session.")


def _collect_command_output(client, project_id: str, studio_id: str, session_name: str, payload: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    output = str(payload.get("output") or "")
    if RESULT_BEGIN in output and RESULT_END in output:
        return output, None
    last_status = None
    for _ in range(12):
        status = get_session_status(client, project_id, studio_id, session_name)
        if status is not None:
            last_status = status
            candidate_output = str(status.get("output") or "")
            if candidate_output:
                output = candidate_output
            if RESULT_BEGIN in output and RESULT_END in output:
                return output, last_status
            if status.get("state") in {"completed", "failed"}:
                break
        time.sleep(5)
    return output, last_status


def _wait_for_detached_session_output(
    client,
    project_id: str,
    studio_id: str,
    session_name: str,
    *,
    timeout_seconds: int = 1800,
    poll_seconds: int = 15,
) -> tuple[str, dict[str, Any] | None]:
    deadline = time.time() + timeout_seconds
    output = ""
    last_status = None
    while time.time() < deadline:
        status = get_session_status(client, project_id, studio_id, session_name)
        if status is not None:
            last_status = status
            candidate_output = str(status.get("output") or "")
            if candidate_output:
                output = candidate_output
            if RESULT_BEGIN in output and RESULT_END in output:
                return output, last_status
            if status.get("state") in {"completed", "failed"}:
                break
        time.sleep(poll_seconds)
    return output, last_status


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
        f"rm -f {shlex.quote(RESULT_CACHE_PATH)}",
        "python wait_for_trained_model.py",
        "python run_ai_trading_smoke.py",
        "latest=$(ls -1t results/ai_smoke_*.json | head -n1)",
        f"cp \"$latest\" {shlex.quote(RESULT_CACHE_PATH)}",
        f"echo {shlex.quote(RESULT_PATH_PREFIX)}\"$latest\"",
        f"echo {shlex.quote(RESULT_BEGIN)}",
        'cat "$latest"',
        f"echo {shlex.quote(RESULT_END)}",
    ]
    return f"bash -lc {shlex.quote(chr(10).join(script_lines))}"


def _fetch_result_file(client, project_id: str, studio_id: str, *, config, session_name: str) -> tuple[str | None, dict[str, Any] | None]:
    repo_dir = Path(config.studio_root_dir.rstrip("/")) / config.studio_repo_dir
    command = "\n".join(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(str(repo_dir))}",
            f"if [ ! -f {shlex.quote(RESULT_CACHE_PATH)} ]; then exit 3; fi",
            f"echo {shlex.quote(RESULT_PATH_PREFIX)}{RESULT_CACHE_PATH}",
            f"echo {shlex.quote(RESULT_BEGIN)}",
            f"cat {shlex.quote(RESULT_CACHE_PATH)}",
            f"echo {shlex.quote(RESULT_END)}",
        ]
    )
    try:
        payload = _checked_execute(
            client,
            project_id,
            studio_id,
            command=f"bash -lc {shlex.quote(command)}",
            session_name=session_name,
        )
    except Exception:
        return None, None
    output = str(payload.get("output") or "")
    try:
        return _extract_result(output)
    except Exception:
        return None, None


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

    session_name = f"{config.studio_session_name}-ai-smoke-{int(time.time())}"
    payload = _launch_detached_session(
        client,
        project.project_id,
        studio_id,
        command=_build_smoke_command(config, service_port=_service_port(config)),
        session_name=session_name,
    )
    output_text, session_status = _wait_for_detached_session_output(
        client,
        project.project_id,
        studio_id,
        session_name,
    )
    try:
        result_path, result_payload = _extract_result(output_text)
    except Exception as exc:  # noqa: BLE001
        result_path, result_payload = _fetch_result_file(
            client,
            project.project_id,
            studio_id,
            config=config,
            session_name=f"{session_name}-result-fetch",
        )
        if result_payload is not None:
            out_path = Path(args.result_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result_payload, indent=2) + "\n")
            summary = {
                "ok": True,
                "project_id": project.project_id,
                "project_name": project.name,
                "studio_id": studio_id,
                "studio_name": str(getattr(studio, "name", "") or ""),
                "result_path": result_path,
                "result_out": str(out_path),
                "recovered_via_file_fetch": True,
            }
            print(json.dumps(summary, indent=2))
            return
        raise RuntimeError(
            json.dumps(
                {
                    "error": str(exc),
                    "payload": payload,
                    "session_status": session_status,
                    "output_tail": output_text[-4000:],
                },
                indent=2,
            )
        ) from exc

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
