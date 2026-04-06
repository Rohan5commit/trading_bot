from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import shutil
import signal
import subprocess
import time
from typing import Callable, Iterable


@dataclass(frozen=True)
class TrackedPath:
    source: Path
    relative_path: Path


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)


def _copy_path(source: Path, destination: Path) -> None:
    if source.is_symlink():
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(os.readlink(source), destination)
        return
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return
    if source.is_dir():
        shutil.copytree(source, destination)
        return
    raise FileNotFoundError(f"Cannot snapshot missing path: {source}")


def normalize_tracked_paths(tracked_paths: Iterable[str], workdir: Path, checkpoint_dir: Path) -> list[TrackedPath]:
    normalized: list[TrackedPath] = []
    seen: set[str] = set()
    workdir = workdir.resolve()
    checkpoint_dir = checkpoint_dir.resolve()

    for raw_path in tracked_paths:
        value = raw_path.strip()
        if not value:
            continue
        source = Path(value).expanduser()
        if not source.is_absolute():
            source = (workdir / source).resolve()
        else:
            source = source.resolve()

        if source == checkpoint_dir or _is_relative_to(source, checkpoint_dir):
            raise ValueError(f"Tracked path '{value}' cannot point to the checkpoint directory.")
        if _is_relative_to(checkpoint_dir, source):
            raise ValueError(
                f"Tracked path '{value}' contains the checkpoint directory. Track a narrower output directory instead."
            )

        try:
            relative_path = source.relative_to(workdir)
        except ValueError as exc:
            raise ValueError(f"Tracked path '{value}' must stay inside the working directory {workdir}.") from exc

        key = relative_path.as_posix()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(TrackedPath(source=source, relative_path=relative_path))

    if not normalized:
        raise ValueError("At least one tracked path is required.")
    return normalized


def _path_stats(path: Path) -> tuple[int, int, int]:
    if path.is_file() or path.is_symlink():
        return 1, 0, path.stat().st_size

    file_count = 0
    dir_count = 0
    total_bytes = 0
    for child in path.rglob("*"):
        if child.is_dir():
            dir_count += 1
            continue
        file_count += 1
        if child.exists():
            total_bytes += child.stat().st_size
    return file_count, dir_count, total_bytes


def create_snapshot(
    tracked_paths: list[TrackedPath],
    checkpoint_dir: Path,
    *,
    reason: str,
    command: str,
    child_exit_code: int | None = None,
    restored_from_checkpoint: bool = False,
) -> dict:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_dir = checkpoint_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    for tracked in tracked_paths:
        source = tracked.source
        destination = state_dir / tracked.relative_path
        if destination.exists() or destination.is_symlink():
            _remove_path(destination)

        entry = {
            "path": tracked.relative_path.as_posix(),
            "exists": source.exists(),
            "kind": "missing",
            "files": 0,
            "directories": 0,
            "bytes": 0,
            "modified_at_utc": None,
        }

        if source.exists():
            _copy_path(source, destination)
            files, directories, total_bytes = _path_stats(source)
            entry.update(
                {
                    "kind": "directory" if source.is_dir() else "file",
                    "files": files,
                    "directories": directories,
                    "bytes": total_bytes,
                    "modified_at_utc": datetime.fromtimestamp(source.stat().st_mtime, UTC).isoformat(),
                }
            )
        entries.append(entry)

    manifest = {
        "generated_at_utc": utc_now(),
        "reason": reason,
        "command": command,
        "checkpoint_dir": str(checkpoint_dir),
        "child_exit_code": child_exit_code,
        "restored_from_checkpoint": restored_from_checkpoint,
        "entries": entries,
    }
    (checkpoint_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def restore_snapshot(tracked_paths: list[TrackedPath], checkpoint_dir: Path) -> bool:
    state_dir = checkpoint_dir / "state"
    if not state_dir.exists():
        return False

    restored_any = False
    for tracked in tracked_paths:
        source = state_dir / tracked.relative_path
        target = tracked.source
        if not source.exists():
            continue
        if target.exists() or target.is_symlink():
            _remove_path(target)
        _copy_path(source, target)
        restored_any = True
    return restored_any


def run_checkpointed_command(
    *,
    shell_command: str,
    workdir: Path,
    checkpoint_dir: Path,
    tracked_paths: Iterable[str],
    save_every_seconds: int,
    grace_period_seconds: int,
    shutdown_exit_code: int = 0,
    after_snapshot: Callable[[dict], None] | None = None,
) -> int:
    workdir = workdir.resolve()
    checkpoint_dir = checkpoint_dir.resolve()
    tracked = normalize_tracked_paths(tracked_paths, workdir, checkpoint_dir)
    restored_from_checkpoint = restore_snapshot(tracked, checkpoint_dir)

    print(f"[lightning-checkpoint] workdir={workdir}")
    print(f"[lightning-checkpoint] checkpoint_dir={checkpoint_dir}")
    print(f"[lightning-checkpoint] restored_from_checkpoint={restored_from_checkpoint}")

    child = subprocess.Popen(
        shell_command,
        cwd=workdir,
        shell=True,
        executable="/bin/bash",
        start_new_session=True,
    )

    shutdown_requested = False
    shutdown_signal: int | None = None
    shutdown_deadline: float | None = None
    shutdown_snapshot_taken = False

    def _forward_signal(signum: int) -> None:
        if child.poll() is not None:
            return
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            pass

    def _handle_signal(signum: int, _frame) -> None:
        nonlocal shutdown_requested, shutdown_signal, shutdown_deadline
        signal_name = signal.Signals(signum).name
        print(f"[lightning-checkpoint] received {signal_name}; saving state and shutting down child process")
        shutdown_requested = True
        shutdown_signal = signum
        shutdown_deadline = time.monotonic() + grace_period_seconds
        _forward_signal(signum)

    previous_sigterm = signal.getsignal(signal.SIGTERM)
    previous_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    exit_code: int | None = None
    next_snapshot_at = time.monotonic() + max(save_every_seconds, 1)

    def _create_and_publish_snapshot(*, reason: str, child_exit_code: int | None = None) -> dict:
        manifest = create_snapshot(
            tracked,
            checkpoint_dir,
            reason=reason,
            command=shell_command,
            child_exit_code=child_exit_code,
            restored_from_checkpoint=restored_from_checkpoint,
        )
        if after_snapshot is not None:
            after_snapshot(manifest)
        return manifest

    try:
        while True:
            try:
                exit_code = child.wait(timeout=1.0)
                break
            except subprocess.TimeoutExpired:
                pass

            now = time.monotonic()
            if shutdown_requested and not shutdown_snapshot_taken:
                reason = f"signal_{signal.Signals(shutdown_signal).name.lower()}" if shutdown_signal else "signal"
                _create_and_publish_snapshot(reason=reason)
                shutdown_snapshot_taken = True
                continue

            if not shutdown_requested and now >= next_snapshot_at:
                _create_and_publish_snapshot(reason="interval")
                print(f"[lightning-checkpoint] saved interval snapshot at {utc_now()}")
                next_snapshot_at = now + max(save_every_seconds, 1)

            if shutdown_requested and shutdown_deadline and now >= shutdown_deadline and child.poll() is None:
                print("[lightning-checkpoint] grace period expired; forcing child process group to stop")
                _forward_signal(signal.SIGKILL)

        final_reason = "completed" if exit_code == 0 and not shutdown_requested else "shutdown" if shutdown_requested else "failed"
        _create_and_publish_snapshot(reason=final_reason, child_exit_code=exit_code)
        if shutdown_requested:
            return shutdown_exit_code
        return exit_code or 0
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)
