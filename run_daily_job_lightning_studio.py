from __future__ import annotations

import argparse
import base64
import csv
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


RESULT_BEGIN = "__DAILY_STUDIO_RESULT_BEGIN__"
RESULT_END = "__DAILY_STUDIO_RESULT_END__"
RESULT_PATH_PREFIX = "__DAILY_STUDIO_RESULT_PATH__="
FILE_BEGIN = "__DAILY_STUDIO_FILE_BEGIN__"
FILE_END = "__DAILY_STUDIO_FILE_END__"
FILE_PATH_PREFIX = "__DAILY_STUDIO_FILE_PATH__="
FILE_MISSING_PREFIX = "__DAILY_STUDIO_FILE_MISSING__="
RESULT_CACHE_PATH = "results/daily_job_studio_current.json"
LOG_CACHE_PATH = "results/daily_job_studio_current.log"


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


def _metadata_only(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    compact = dict(payload)
    raw_output = compact.pop("output", None)
    if raw_output is not None:
        compact["has_output"] = bool(str(raw_output))
        compact["output_chars"] = len(str(raw_output))
    return compact


def _service_port(config) -> int:
    override = str(os.getenv("LIGHTNING_INFERENCE_PORT") or "").strip()
    if override:
        return int(override)
    command = str(config.run.command or "")
    parts = command.split("--port", 1)
    if len(parts) == 2:
        return int(parts[1].strip().split()[0])
    return 8000


def _build_daily_command(config, *, service_port: int, reset_ai_positions: bool, clear_email_markers: bool) -> str:
    repo_dir = Path(config.studio_root_dir.rstrip("/")) / config.studio_repo_dir
    exports = dict(config.run.app_env)
    passthrough_keys = [
        "SMTP_SERVER",
        "SMTP_PORT",
        "SENDER_EMAIL",
        "SENDER_PASSWORD",
        "RECIPIENT_EMAIL",
        "TWELVEDATA_API_KEYS",
        "ARM_LIVE_TRADING",
        "LOG_LEVEL",
        "TRAINED_MODEL_API_KEY",
        "AI_PROMPT_CANDIDATES_LIMIT",
        "TRAINED_MODEL_TIMEOUT_SECONDS",
        "TRAINED_MODEL_READY_TIMEOUT_SECONDS",
        "TRAINED_MODEL_READY_POLL_SECONDS",
        "TRAINED_MODEL_MAX_RETRIES",
        "TRAINED_MODEL_BACKOFF_SECONDS",
        "TRAINED_MODEL_BATCH_SIZE",
        "TRAINED_MODEL_CLASS_TOKEN_INFERENCE",
        "TRAINED_MODEL_WARMUP_TIMEOUT_SECONDS",
        "DISABLE_CORE_TRADING",
    ]
    for key in passthrough_keys:
        value = os.getenv(key)
        if value is not None:
            exports[key] = str(value)
    exports["TRAINED_MODEL_INFERENCE_URL"] = f"http://127.0.0.1:{service_port}"
    exports["PYTHONUNBUFFERED"] = "1"

    export_lines = [
        f"export {key}={shlex.quote(str(value))}"
        for key, value in exports.items()
        if value is not None and str(key).strip()
    ]

    prep_script = "\n".join(
        [
            "import glob",
            "import json",
            "import os",
            "import sqlite3",
            "from pathlib import Path",
            "import yaml",
            "",
            "repo = Path.cwd()",
            "results_dir = repo / 'results'",
            "results_dir.mkdir(parents=True, exist_ok=True)",
            "cfg = yaml.safe_load((repo / 'config.yaml').read_text()) or {}",
            "db_path = str(cfg.get('data', {}).get('cache_path') or 'data/trading_bot.db')",
            "db_file = Path(db_path) if os.path.isabs(db_path) else (repo / db_path)",
            f"clear_markers = {str(bool(clear_email_markers))}",
            f"reset_ai = {str(bool(reset_ai_positions))}",
            "if clear_markers:",
            "    for path in results_dir.glob('email_sent*.ok'):",
            "        try:",
            "            path.unlink()",
            "        except FileNotFoundError:",
            "            pass",
            "if reset_ai:",
            "    seed_path = repo / 'state' / 'positions_seed.json'",
            "    if seed_path.exists():",
            "        payload = json.loads(seed_path.read_text())",
            "        if isinstance(payload, dict):",
            "            payload['positions_ai'] = []",
            "            seed_path.write_text(json.dumps(payload, indent=2) + '\\n')",
            "    if db_file.exists():",
            "        conn = sqlite3.connect(str(db_file))",
            "        try:",
            "            conn.execute(\"DELETE FROM positions_ai\")",
            "            conn.commit()",
            "        except sqlite3.OperationalError:",
            "            pass",
            "        finally:",
            "            conn.close()",
            "    for pattern in ('daily_report_ai_*.csv', 'unrealized_ai_*.csv', 'trades_*.csv'):",
            "        for path in results_dir.glob(pattern):",
            "            try:",
            "                path.unlink()",
            "            except FileNotFoundError:",
            "                pass",
            "print(json.dumps({'reset_ai_positions': reset_ai, 'db_path': str(db_file)}, indent=2))",
        ]
    )

    post_script = "\n".join(
        [
            "import csv",
            "import glob",
            "import json",
            "import os",
            "from pathlib import Path",
            "",
            "repo = Path.cwd()",
            "results_dir = repo / 'results'",
            "log_path = results_dir / 'daily_job_studio_current.log'",
            "text = log_path.read_text(errors='replace') if log_path.exists() else ''",
            "email_sent_count = text.count('Email sent successfully')",
            "core_disabled = str(os.getenv('DISABLE_CORE_TRADING') or '').strip().lower() in {'1', 'true', 'yes', 'on'}",
            "required_email_count = 1 if core_disabled else 2",
            "trained_model_batch_responses = text.count('Trained model batch response')",
            "trained_model_batch_failures = text.count('Trained model batch inference failed')",
            "trained_model_unusable = 'No usable trained-model predictions' in text",
            "ai_reports = sorted(glob.glob(str(results_dir / 'daily_report_ai_*.csv')))",
            "core_reports = [",
            "    path",
            "    for path in sorted(glob.glob(str(results_dir / 'daily_report_*.csv')))",
            "    if not Path(path).name.startswith('daily_report_ai_')",
            "]",
            "ai_row = {}",
            "if ai_reports:",
            "    with open(ai_reports[-1], newline='') as handle:",
            "        rows = list(csv.DictReader(handle))",
            "        if rows:",
            "            ai_row = rows[-1]",
            "core_row = {}",
            "if core_reports:",
            "    with open(core_reports[-1], newline='') as handle:",
            "        rows = list(csv.DictReader(handle))",
            "        if rows:",
            "            core_row = rows[-1]",
            "ai_new_positions = int(float(ai_row.get('new_positions_opened', 0) or 0)) if ai_row else 0",
            "ai_open_positions = int(float(ai_row.get('open_positions', 0) or 0)) if ai_row else 0",
            "ai_skip_reason = str(ai_row.get('ai_llm_skipped_reason', '') or '').strip().lower() if ai_row else ''",
            "ai_skipped_legit = ai_skip_reason in {'no_capacity', 'no_slots', 'no_candidates', 'all_neutral', 'no_tradeable_signals'}",
            "ai_llm_ok = None",
            "if ai_row:",
            "    raw_ai_llm_ok = ai_row.get('ai_llm_ok')",
            "    if raw_ai_llm_ok is not None and str(raw_ai_llm_ok).strip() != '':",
            "        s = str(raw_ai_llm_ok).strip().lower()",
            "        if s in ('1', 'true', 'yes', 'y', 't'):",
            "            ai_llm_ok = True",
            "        elif s in ('0', 'false', 'no', 'n', 'f'):",
            "            ai_llm_ok = False",
            "model_call_observed = trained_model_batch_responses > 0",
            "ai_path_ok = bool(model_call_observed or ai_skipped_legit)",
            "run_rc = int(os.getenv('DAILY_RUN_RC', '1') or '1')",
            "result = {",
            "    'ok': bool(",
            "        run_rc == 0",
            "        and email_sent_count >= required_email_count",
            "        and trained_model_batch_failures == 0",
            "        and not trained_model_unusable",
            "        and ai_llm_ok is not False",
            "        and ai_path_ok",
            "    ),",
            "    'run_rc': run_rc,",
            "    'email_sent_count': email_sent_count,",
            "    'required_email_count': required_email_count,",
            "    'core_disabled': core_disabled,",
            "    'decision_engine': 'trained_model',",
            "    'backend': 'lightning-studio-local-http',",
            "    'trained_model_inference_url': os.getenv('TRAINED_MODEL_INFERENCE_URL', ''),",
            "    'trained_model_batch_responses': trained_model_batch_responses,",
            "    'trained_model_batch_failures': trained_model_batch_failures,",
            "    'trained_model_unusable': trained_model_unusable,",
            "    'model_call_observed': model_call_observed,",
            "    'ai_llm_ok': ai_llm_ok,",
            "    'ai_skipped_legit': ai_skipped_legit,",
            "    'ai_skip_reason': ai_skip_reason,",
            "    'ai_new_positions_opened': ai_new_positions,",
            "    'ai_open_positions': ai_open_positions,",
            "    'ai_report': ai_row,",
            "    'core_report': core_row,",
            "}",
            f"(results_dir / '{Path(RESULT_CACHE_PATH).name}').write_text(json.dumps(result, indent=2) + '\\n')",
            "print(json.dumps(result, indent=2))",
            "raise SystemExit(0 if result['ok'] else 1)",
        ]
    )

    warmup_script = "\n".join(
        [
            "import json",
            "import os",
            "import time",
            "import requests",
            "",
            "base_url = str(os.getenv('TRAINED_MODEL_INFERENCE_URL') or '').strip().rstrip('/')",
            "if not base_url:",
            "    print(json.dumps({'trained_model_warmup': 'skipped', 'reason': 'missing_inference_url'}))",
            "    raise SystemExit(0)",
            "warmup_url = base_url + '/warmup'",
            "api_key = str(os.getenv('TRAINED_MODEL_API_KEY') or '').strip()",
            "headers = {'Content-Type': 'application/json'}",
            "if api_key:",
            "    headers['Authorization'] = f'Bearer {api_key}'",
            "timeout_seconds = int(float(os.getenv('TRAINED_MODEL_WARMUP_TIMEOUT_SECONDS') or 1800))",
            "t0 = time.time()",
            "response = requests.post(warmup_url, headers=headers, timeout=timeout_seconds)",
            "response.raise_for_status()",
            "payload = response.json() if response.content else {}",
            "print(json.dumps({",
            "    'trained_model_warmup': 'ok',",
            "    'warmup_url': warmup_url,",
            "    'elapsed_seconds': round(time.time() - t0, 2),",
            "    'status_code': response.status_code,",
            "    'payload': payload,",
            "}, indent=2))",
        ]
    )

    script_lines = [
        "set -euo pipefail",
        *export_lines,
        f"cd {shlex.quote(str(repo_dir))}",
        "if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi",
        f"rm -f {shlex.quote(RESULT_CACHE_PATH)} {shlex.quote(LOG_CACHE_PATH)}",
        "python - <<'PY'",
        prep_script,
        "PY",
        "python - <<'PY'",
        warmup_script,
        "PY",
        "run_rc=0",
        f"python main.py daily_job 2>&1 | tee {shlex.quote(LOG_CACHE_PATH)} || run_rc=$?",
        "export DAILY_RUN_RC=\"$run_rc\"",
        "python - <<'PY'",
        post_script,
        "PY",
        f"echo {shlex.quote(RESULT_PATH_PREFIX)}{RESULT_CACHE_PATH}",
        f"echo {shlex.quote(RESULT_BEGIN)}",
        f"cat {shlex.quote(RESULT_CACHE_PATH)}",
        f"echo {shlex.quote(RESULT_END)}",
        "exit \"$run_rc\"",
    ]
    return f"bash -lc {shlex.quote(chr(10).join(script_lines))}"


def _launch_detached_session(client, project_id: str, studio_id: str, *, command: str, session_name: str) -> dict[str, Any]:
    result = execute_studio_command(
        client,
        project_id,
        studio_id,
        command=command,
        session_name=session_name,
        detached=True,
    )
    return _command_payload(result)


def _wait_for_detached_session_output(
    client,
    project_id: str,
    studio_id: str,
    session_name: str,
    *,
    timeout_seconds: int = 5400,
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
    result = execute_studio_command(
        client,
        project_id,
        studio_id,
        command=f"bash -lc {shlex.quote(command)}",
        session_name=session_name,
        detached=False,
    )
    output = str(_command_payload(result).get("output") or "")
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
        raise RuntimeError("Studio daily output did not include result markers.")
    fragment = text.split(RESULT_BEGIN, 1)[1].split(RESULT_END, 1)[0].strip()
    payload = json.loads(fragment)
    if not isinstance(payload, dict):
        raise RuntimeError("Studio daily output JSON was not an object.")
    return result_path, payload


def _fetch_remote_file(
    client,
    project_id: str,
    studio_id: str,
    *,
    config,
    remote_relative_path: str,
    session_name: str,
) -> tuple[str | None, bytes | None]:
    repo_dir = Path(config.studio_root_dir.rstrip("/")) / config.studio_repo_dir
    script = "\n".join(
        [
            "import base64",
            "from pathlib import Path",
            f"path = Path({remote_relative_path!r})",
            "if not path.exists() or not path.is_file():",
            f"    print({FILE_MISSING_PREFIX!r} + str(path))",
            "    raise SystemExit(0)",
            f"print({FILE_PATH_PREFIX!r} + str(path))",
            f"print({FILE_BEGIN!r})",
            "print(base64.b64encode(path.read_bytes()).decode('ascii'))",
            f"print({FILE_END!r})",
        ]
    )
    command = "\n".join(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(str(repo_dir))}",
            "python - <<'PY'",
            script,
            "PY",
        ]
    )
    result = execute_studio_command(
        client,
        project_id,
        studio_id,
        command=f"bash -lc {shlex.quote(command)}",
        session_name=session_name,
        detached=False,
    )
    output = str(_command_payload(result).get("output") or "")
    missing_path = None
    remote_path = None
    for line in output.splitlines():
        if line.startswith(FILE_MISSING_PREFIX):
            missing_path = line.split("=", 1)[1].strip() if "=" in line else line[len(FILE_MISSING_PREFIX):].strip()
            break
        if line.startswith(FILE_PATH_PREFIX):
            remote_path = line.split("=", 1)[1].strip() if "=" in line else line[len(FILE_PATH_PREFIX):].strip()
            break
    if missing_path is not None:
        return missing_path, None
    if FILE_BEGIN not in output or FILE_END not in output:
        raise RuntimeError(f"Studio file fetch output did not include file markers for {remote_relative_path}.")
    fragment = output.split(FILE_BEGIN, 1)[1].split(FILE_END, 1)[0].strip()
    return remote_path, base64.b64decode(fragment.encode("ascii"))


def _sync_remote_ai_artifacts(
    client,
    project_id: str,
    studio_id: str,
    *,
    config,
    payload: dict[str, Any] | None,
    session_prefix: str,
) -> list[str]:
    if not isinstance(payload, dict):
        return []
    ai_report = payload.get("result", {}).get("ai_report") if isinstance(payload.get("result"), dict) else None
    if not isinstance(ai_report, dict):
        return []
    date_text = str(ai_report.get("date") or "").strip()
    if not date_text:
        return []
    date_token = date_text.replace("-", "")
    targets = [
        f"results/daily_report_ai_{date_token}.csv",
        f"results/unrealized_ai_{date_token}.csv",
    ]
    local_results_dir = Path("results")
    local_results_dir.mkdir(parents=True, exist_ok=True)
    saved_files: list[str] = []
    for index, remote_path in enumerate(targets, start=1):
        fetched_path, blob = _fetch_remote_file(
            client,
            project_id,
            studio_id,
            config=config,
            remote_relative_path=remote_path,
            session_name=f"{session_prefix}-artifact-{index}",
        )
        if blob is None:
            continue
        filename = Path(fetched_path or remote_path).name
        local_path = local_results_dir / filename
        local_path.write_bytes(blob)
        saved_files.append(str(local_path))
    return saved_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="quant_platform/configs/lightning_inference_studio.yaml")
    parser.add_argument("--result-out", default="results/daily_job_lightning_studio.json")
    parser.add_argument("--session-name", default="trading-bot-daily-job")
    parser.add_argument("--timeout-seconds", type=int, default=10800)
    parser.add_argument("--reset-ai-positions", action="store_true")
    parser.add_argument("--keep-email-markers", action="store_true")
    args = parser.parse_args()

    auth_env = ensure_auth_env()
    os.environ.update(auth_env)
    set_process_env(auth_env)
    ensure_studio_auth_env()
    client, project = get_client_and_project(project_id=_resolved_project_id())
    config = load_studio_config(args.config)
    studio = resolve_studio(client, project.project_id, config)
    if studio is None:
        raise RuntimeError("No matching Lightning Studio was found for full daily run.")
    studio_id = str(getattr(studio, "id", "") or "").strip()
    if not studio_id:
        raise RuntimeError("Lightning Studio did not expose an id.")
    instance = resolve_studio_instance(client, project.project_id, studio_id)
    if instance is None:
        raise RuntimeError("Lightning Studio does not have an active instance.")

    command = _build_daily_command(
        config,
        service_port=_service_port(config),
        reset_ai_positions=bool(args.reset_ai_positions),
        clear_email_markers=not bool(args.keep_email_markers),
    )
    launch = _launch_detached_session(
        client,
        project.project_id,
        studio_id,
        command=command,
        session_name=args.session_name,
    )
    output, session_status = _wait_for_detached_session_output(
        client,
        project.project_id,
        studio_id,
        args.session_name,
        timeout_seconds=max(60, int(args.timeout_seconds or 5400)),
    )

    result_path = None
    payload = None
    if RESULT_BEGIN in output and RESULT_END in output:
        result_path, payload = _extract_result(output)
    else:
        result_path, payload = _fetch_result_file(
            client,
            project.project_id,
            studio_id,
            config=config,
            session_name=f"{args.session_name}-fetch",
        )

    synced_files = _sync_remote_ai_artifacts(
        client,
        project.project_id,
        studio_id,
        config=config,
        payload={"result": payload} if isinstance(payload, dict) else None,
        session_prefix=f"{args.session_name}-sync",
    )

    report = {
        "project_id": project.project_id,
        "project_name": project.name,
        "studio_id": studio_id,
        "session_name": args.session_name,
        # Persist metadata only; full command/session output can contain secrets.
        "launch": _metadata_only(launch),
        "session": _metadata_only(session_status),
        "result_path": result_path,
        "result": payload,
        "synced_files": synced_files,
    }
    Path(args.result_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.result_out).write_text(json.dumps(json_safe(report), indent=2) + "\n")
    print(json.dumps(json_safe(report), indent=2))

    if not isinstance(payload, dict) or not payload.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
