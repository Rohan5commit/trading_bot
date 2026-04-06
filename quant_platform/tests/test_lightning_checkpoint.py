from pathlib import Path
import json

from lightning_checkpoint import create_snapshot, normalize_tracked_paths, restore_snapshot, run_checkpointed_command


def test_snapshot_round_trip(tmp_path):
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    checkpoint_dir = workdir / ".lightning-checkpoints"

    artifacts_dir = workdir / "artifacts"
    artifacts_dir.mkdir()
    report_file = workdir / "reports" / "summary.txt"
    report_file.parent.mkdir()

    (artifacts_dir / "model.bin").write_text("checkpoint-v1")
    report_file.write_text("initial-report")

    tracked = normalize_tracked_paths(["artifacts", "reports"], workdir, checkpoint_dir)
    manifest = create_snapshot(
        tracked,
        checkpoint_dir,
        reason="interval",
        command="python train.py",
        restored_from_checkpoint=False,
    )

    assert manifest["reason"] == "interval"
    assert (checkpoint_dir / "manifest.json").exists()

    (artifacts_dir / "model.bin").write_text("checkpoint-v2")
    report_file.write_text("mutated-report")

    restored = restore_snapshot(tracked, checkpoint_dir)
    assert restored is True
    assert (artifacts_dir / "model.bin").read_text() == "checkpoint-v1"
    assert report_file.read_text() == "initial-report"

    persisted_manifest = json.loads((checkpoint_dir / "manifest.json").read_text())
    assert persisted_manifest["entries"][0]["path"] == "artifacts"


def test_normalize_tracked_paths_rejects_checkpoint_overlap(tmp_path):
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    checkpoint_dir = workdir / ".lightning-checkpoints"
    checkpoint_dir.mkdir()

    try:
        normalize_tracked_paths(["."], workdir, checkpoint_dir)
    except ValueError as exc:
        assert "contains the checkpoint directory" in str(exc)
    else:
        raise AssertionError("Expected tracked path validation to fail.")


def test_run_checkpointed_command_calls_after_snapshot(tmp_path):
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    checkpoint_dir = workdir / ".lightning-checkpoints"
    artifacts_dir = workdir / "artifacts"
    artifacts_dir.mkdir()

    manifests = []
    exit_code = run_checkpointed_command(
        shell_command="printf 'model-v1' > artifacts/model.bin",
        workdir=workdir,
        checkpoint_dir=checkpoint_dir,
        tracked_paths=["artifacts"],
        save_every_seconds=3600,
        grace_period_seconds=5,
        after_snapshot=manifests.append,
    )

    assert exit_code == 0
    assert manifests
    assert manifests[-1]["reason"] == "completed"
    assert (checkpoint_dir / "manifest.json").exists()
