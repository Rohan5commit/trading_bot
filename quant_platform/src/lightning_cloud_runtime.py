from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import time

from lightning_app import BuildConfig, CloudCompute, LightningApp, LightningFlow, LightningWork
from lightning_app.storage import Drive

from src.lightning_checkpoint import run_checkpointed_command
from src.lightning_cloud_utils import LightningCloudConfig, ROOT_DIR, load_run_config


STATUS_FILENAME = ".lightning-cloud-status.json"
SHUTDOWN_RESTART_EXIT_CODE = 75


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _resolve_repo_relative(path_value: str) -> str:
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve().relative_to(ROOT_DIR.resolve()).as_posix()
    return Path(path_value).as_posix()


class CheckpointedCommandWork(LightningWork):
    def __init__(self, config: LightningCloudConfig) -> None:
        build_config = BuildConfig(requirements=[str((ROOT_DIR / config.work_requirements_file).resolve())])
        cloud_compute = CloudCompute(name=config.cloud_compute_name, disk_size=config.disk_size_gb)
        super().__init__(
            parallel=True,
            raise_exception=False,
            cloud_build_config=build_config,
            cloud_compute=cloud_compute,
        )
        self.app_name = config.app_name
        self.command = config.command
        self.checkpoint_dir = _resolve_repo_relative(config.checkpoint_dir)
        self.tracked_paths = list(config.tracked_paths)
        self.save_every_seconds = int(config.save_every_seconds)
        self.grace_period_seconds = int(config.grace_period_seconds)
        self.restart_exit_codes = list(config.restart_exit_codes)
        self.last_exit_code = 0
        self.last_error = ""
        self.last_snapshot_at = ""
        self.last_snapshot_reason = ""
        self.snapshot_count = 0
        self.restored_from_drive = False
        self.started_at = ""
        self.finished_at = ""
        self.current_run_token = ""
        self.state_file = STATUS_FILENAME
        self.checkpoint_drive = Drive(f"lit://{config.drive_id}", root_folder=str(ROOT_DIR))

    def _checkpoint_dir_path(self) -> Path:
        return (ROOT_DIR / self.checkpoint_dir).resolve()

    def _status_path(self) -> Path:
        return (ROOT_DIR / self.state_file).resolve()

    def _write_status_file(self, *, manifest: dict | None = None) -> None:
        payload = {
            "generated_at_utc": _utc_now(),
            "app_name": self.app_name,
            "work_name": self.name,
            "command": self.command,
            "current_run_token": self.current_run_token,
            "last_exit_code": self.last_exit_code,
            "last_error": self.last_error,
            "last_snapshot_at": self.last_snapshot_at,
            "last_snapshot_reason": self.last_snapshot_reason,
            "snapshot_count": self.snapshot_count,
            "restored_from_drive": self.restored_from_drive,
            "started_at_utc": self.started_at or None,
            "finished_at_utc": self.finished_at or None,
            "checkpoint_dir": self.checkpoint_dir,
            "tracked_paths": self.tracked_paths,
        }
        if manifest is not None:
            payload["manifest"] = manifest
        self._status_path().write_text(json.dumps(payload, indent=2))

    def _restore_checkpoint_dir(self) -> bool:
        existing = self.checkpoint_drive.list(self.checkpoint_dir, component_name=self.name)
        if not existing:
            return False
        self.checkpoint_drive.get(self.checkpoint_dir, component_name=self.name, overwrite=True)
        return True

    def _sync_snapshot(self, manifest: dict) -> None:
        self.snapshot_count += 1
        self.last_snapshot_at = manifest.get("generated_at_utc", "")
        self.last_snapshot_reason = manifest.get("reason", "")
        self._write_status_file(manifest=manifest)

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                self.checkpoint_drive.put(self.checkpoint_dir)
                self.checkpoint_drive.put(self.state_file)
                return
            except Exception as exc:  # pragma: no cover - exercised in cloud only
                last_error = exc
                if attempt == 2:
                    break
                time.sleep(5)

        assert last_error is not None
        raise RuntimeError(f"Failed to sync Lightning Drive snapshot: {last_error}") from last_error

    def run(self, run_token: str) -> None:
        self.current_run_token = str(run_token)
        self.started_at = self.started_at or _utc_now()
        self.finished_at = ""
        self.last_error = ""
        self.last_exit_code = 0
        self.restored_from_drive = self._restore_checkpoint_dir()
        self._write_status_file()

        exit_code = run_checkpointed_command(
            shell_command=self.command,
            workdir=ROOT_DIR,
            checkpoint_dir=self._checkpoint_dir_path(),
            tracked_paths=self.tracked_paths,
            save_every_seconds=self.save_every_seconds,
            grace_period_seconds=self.grace_period_seconds,
            shutdown_exit_code=SHUTDOWN_RESTART_EXIT_CODE,
            after_snapshot=self._sync_snapshot,
        )

        self.finished_at = _utc_now()
        self.last_exit_code = int(exit_code)
        self._write_status_file()
        if exit_code == 0:
            return

        self.last_error = f"Wrapped command exited with code {exit_code}."
        self._write_status_file()
        raise RuntimeError(self.last_error)


class AutoResumeFlow(LightningFlow):
    def __init__(self, config: LightningCloudConfig) -> None:
        super().__init__()
        self.runner = CheckpointedCommandWork(config)
        self.app_name = config.app_name
        self.max_restarts = int(config.max_restarts)
        self.restart_exit_codes = list(config.restart_exit_codes)
        self.restart_count = 0
        self.run_token = "0"
        self.completed = False
        self.failed = False
        self.last_action = "idle"
        self.runner_phase = ""
        self.runner_exit_code = 0
        self.last_snapshot_at = ""
        self.last_snapshot_reason = ""
        self.snapshot_count = 0

    def run(self) -> None:
        if not self.completed and not self.failed:
            self.runner.run(self.run_token)

        self.runner_phase = str(self.runner.status.stage)
        self.runner_exit_code = int(self.runner.last_exit_code)
        self.last_snapshot_at = self.runner.last_snapshot_at
        self.last_snapshot_reason = self.runner.last_snapshot_reason
        self.snapshot_count = int(self.runner.snapshot_count)

        if self.runner.has_succeeded:
            self.completed = True
            self.last_action = "completed"
            return

        if not self.runner.has_failed:
            return

        if self.runner_exit_code in self.restart_exit_codes and self.restart_count < self.max_restarts:
            self.restart_count += 1
            self.run_token = str(self.restart_count)
            self.last_action = f"restart-{self.restart_count}"
            return

        self.failed = True
        self.last_action = "failed"


def build_app(config_path: str | Path | None = None) -> LightningApp:
    selected_path = Path(config_path or ROOT_DIR / "configs" / "lightning_run.yaml")
    if not selected_path.is_absolute():
        selected_path = (ROOT_DIR / selected_path).resolve()
    config = load_run_config(selected_path)
    return LightningApp(AutoResumeFlow(config))
