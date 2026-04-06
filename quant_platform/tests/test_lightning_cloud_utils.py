from __future__ import annotations

from pathlib import Path
import sys

from src.lightning_cloud_utils import (
    DEFAULT_CLOUD_CLUSTER_ID,
    build_launch_command,
    ensure_auth_env,
    load_run_config,
    next_app_name,
    sanitize_drive_id,
)


def test_sanitize_drive_id_normalizes_name() -> None:
    assert sanitize_drive_id("Lightning CPU Build Data!!") == "lightning-cpu-build-data"


def test_next_app_name_keeps_stable_prefix() -> None:
    assert next_app_name("lightning-cpu-build-data").startswith("lightning-cpu-build-data-")


def test_load_run_config_accepts_legacy_run_name(tmp_path: Path) -> None:
    config_path = tmp_path / "lightning.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run_name: legacy-name",
                "command: python train.py",
                "tracked_paths:",
                "  - artifacts",
            ]
        )
    )

    config = load_run_config(config_path)

    assert config.app_name == "legacy-name"
    assert config.cloud_cluster_id == DEFAULT_CLOUD_CLUSTER_ID
    assert config.force_run_on_launch is True
    assert config.drive_id == "legacy-name-checkpoints"
    assert config.tracked_paths == ("artifacts",)


def test_build_launch_command_includes_flags_and_env(tmp_path: Path) -> None:
    config_path = tmp_path / "lightning.yaml"
    config_path.write_text(
        "\n".join(
            [
                "app_name: cpu-build",
                "command: python train.py",
                "tracked_paths:",
                "  - artifacts",
                "app_env:",
                "  FOO: bar",
            ]
        )
    )

    config = load_run_config(config_path)
    command = build_launch_command(config, config_path=config_path)

    assert command[0] == sys.executable
    assert Path(command[1]).name == "lightning_cloud_dispatch.py"
    assert "lightning_cloud_app.py" in command
    assert "--without-server" in command
    assert "--force-running" in command
    assert config.config_env_var in command
    assert str(config_path) in command
    assert "FOO=bar" in command


def test_ensure_auth_env_sets_cluster_and_compatibility_vars(monkeypatch) -> None:
    monkeypatch.setattr("src.lightning_cloud_utils.fetch_user_id", lambda **_: "user-123")

    env = ensure_auth_env(
        {
            "LIGHTNING_USERNAME": "rohansanthoshkumar",
            "LIGHTNING_API_KEY": "secret-token",
        },
        cluster_id="lightning-public-prod",
    )

    assert env["LIGHTNING_USER_ID"] == "user-123"
    assert env["LIGHTNING_CLUSTER_ID"] == "lightning-public-prod"
    assert env["GRID_CLUSTER_ID"] == "lightning-public-prod"
    assert env["LIGHTNING_CLOUD_URL"] == "https://lightning.ai"
    assert env["GRID_URL"] == "https://lightning.ai"
