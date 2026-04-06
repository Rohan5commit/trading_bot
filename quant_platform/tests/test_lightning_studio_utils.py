from __future__ import annotations

from pathlib import Path

from src.lightning_studio_utils import (
    build_repo_sync_command,
    build_session_command,
    load_studio_config,
    studio_checkpoint_path,
)


def _write_config(tmp_path: Path, extra_lines: list[str] | None = None) -> Path:
    config_path = tmp_path / "lightning.yaml"
    lines = [
        "app_name: studio-build",
        "command: python train.py",
        "tracked_paths:",
        "  - artifacts",
        "studio_name: test-studio",
        "studio_repo_url: https://github.com/example/private-repo.git",
        "studio_repo_ref: main",
        "studio_repo_dir: studio-repo",
        "studio_session_name: studio-job",
    ]
    if extra_lines:
        lines.extend(extra_lines)
    config_path.write_text("\n".join(lines) + "\n")
    return config_path


def test_studio_checkpoint_path_defaults_outside_repo(tmp_path: Path) -> None:
    config = load_studio_config(_write_config(tmp_path))

    assert studio_checkpoint_path(config) == "/teamspace/studios/this_studio/.lightning-checkpoints/studio-job"


def test_build_repo_sync_command_uses_github_token_for_private_clone(tmp_path: Path, monkeypatch) -> None:
    config = load_studio_config(_write_config(tmp_path))
    monkeypatch.setenv("GITHUB_TOKEN", "ghs-test-token")

    command = build_repo_sync_command(config)

    assert "x-access-token:ghs-test-token@github.com/example/private-repo.git" in command
    assert "git clone --depth 1 --branch main" in command
    assert "git -C /teamspace/studios/this_studio/studio-repo reset --hard FETCH_HEAD" in command


def test_build_session_command_uses_studio_checkpoint_dir(tmp_path: Path) -> None:
    config = load_studio_config(
        _write_config(
            tmp_path,
            extra_lines=["studio_checkpoint_dir: /teamspace/studios/this_studio/checkpoints/custom"],
        )
    )

    command = build_session_command(config)

    assert "--checkpoint-dir /teamspace/studios/this_studio/checkpoints/custom" in command
    assert "if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi" in command
