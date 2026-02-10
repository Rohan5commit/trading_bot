from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Optional

from zoneinfo import ZoneInfo

from src.engine.pipeline import run_signals_pipeline


@dataclass
class SchedulerState:
    last_sent_date: str | None


def _parse_time(value: str) -> dt_time:
    hour, minute = value.split(":")
    return dt_time(int(hour), int(minute))


def _load_state(path: Path) -> SchedulerState:
    if not path.exists():
        return SchedulerState(last_sent_date=None)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return SchedulerState(last_sent_date=payload.get("last_sent_date"))


def _save_state(path: Path, state: SchedulerState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"last_sent_date": state.last_sent_date}, indent=2), encoding="utf-8")


def _get_wifi_device() -> Optional[str]:
    try:
        result = subprocess.run(
            ["networksetup", "-listallhardwareports"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None

    lines = result.stdout.splitlines()
    for idx, line in enumerate(lines):
        if "Hardware Port: Wi-Fi" in line:
            for next_line in lines[idx + 1 : idx + 4]:
                if "Device:" in next_line:
                    return next_line.split(":", 1)[1].strip()
    return None


def _wifi_connected(device: Optional[str]) -> bool:
    if not device:
        return False
    try:
        result = subprocess.run(
            ["networksetup", "-getairportnetwork", device],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return False
    output = result.stdout.strip()
    if "Current Wi-Fi Network:" in output:
        return True
    if "You are not associated" in output:
        return False
    return False


def run_daily_signal_scheduler(config: dict) -> None:
    scheduler_cfg = config.get("scheduler", {})
    tz = ZoneInfo(scheduler_cfg.get("timezone", "Asia/Singapore"))
    run_time = _parse_time(scheduler_cfg.get("run_time_sgt", "08:00"))
    window_end = _parse_time(scheduler_cfg.get("window_end_sgt", "00:00"))
    check_minutes = int(scheduler_cfg.get("wifi_check_minutes", 5))
    state_path = Path(scheduler_cfg.get("state_path", "logs/scheduler_state.json"))
    env_path = Path(scheduler_cfg.get("env_path", ".env"))

    wifi_device = _get_wifi_device()
    _load_dotenv(env_path)

    while True:
        now = datetime.now(tz)
        state = _load_state(state_path)
        today = now.date().isoformat()

        start_dt = datetime.combine(now.date(), run_time, tzinfo=tz)
        end_dt = datetime.combine(now.date(), window_end, tzinfo=tz)
        if window_end <= run_time:
            end_dt += timedelta(days=1)

        if state.last_sent_date == today:
            next_start = start_dt + timedelta(days=1)
            sleep_seconds = max(60, int((next_start - now).total_seconds()))
            time.sleep(sleep_seconds)
            continue

        if now < start_dt:
            time.sleep(max(60, int((start_dt - now).total_seconds())))
            continue

        if now >= end_dt:
            next_start = start_dt + timedelta(days=1)
            time.sleep(max(60, int((next_start - now).total_seconds())))
            continue

        if _wifi_connected(wifi_device):
            try:
                sent = run_signals_pipeline(config)
            except Exception:
                sent = False
            if sent:
                state.last_sent_date = today
                _save_state(state_path, state)
                time.sleep(max(60, int((end_dt - now).total_seconds())))
            else:
                time.sleep(check_minutes * 60)
        else:
            time.sleep(check_minutes * 60)


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"").strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
