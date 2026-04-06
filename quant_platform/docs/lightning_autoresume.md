# Lightning Studio Auto-Resume

This repo now targets Lightning's free Studio control plane directly.

As of March 26, 2026, the live free Studio on this account is:

- Studio name: `rude-teal-op1l`
- cluster: `gcp-lightning-public-prod`
- compute: `cpu-4`
- free instance flag: `true`
- phase: `CLOUD_SPACE_INSTANCE_STATE_RUNNING`

## What The Repo Does

- Uses the Studio API instead of the deprecated Grid runs path.
- Reuses the existing shared checkpoint wrapper in [lightning_checkpoint.py](/Users/rohan/train-once-quant-platform/src/lightning_checkpoint.py).
- Clones or refreshes this private repo inside the Studio workspace using the GitHub Actions token.
- Starts the configured workload as a detached Studio session.
- Polls the session from GitHub Actions and restarts it when the Studio or detached command is no longer running.

## Files

- [lightning_studio_utils.py](/Users/rohan/train-once-quant-platform/src/lightning_studio_utils.py)
  Studio discovery, start, bootstrap, detached command launch, and session polling helpers.
- [lightning_studio_run.py](/Users/rohan/train-once-quant-platform/scripts/lightning_studio_run.py)
  Manual launcher that ensures the Studio is running and starts the detached workload session.
- [lightning_studio_snapshot.py](/Users/rohan/train-once-quant-platform/scripts/lightning_studio_snapshot.py)
  Scheduled health check and restart entrypoint.
- [lightning-autoresume-run.yml](/Users/rohan/train-once-quant-platform/.github/workflows/lightning-autoresume-run.yml)
  Manual Studio launch workflow.
- [lightning-progress-snapshot.yml](/Users/rohan/train-once-quant-platform/.github/workflows/lightning-progress-snapshot.yml)
  Scheduled Studio health check workflow.

## Shared Config

Edit [lightning_run.yaml](/Users/rohan/train-once-quant-platform/configs/lightning_run.yaml).

Important Studio fields:

- `studio_name`
- `studio_cluster_id`
- `studio_compute_name`
- `studio_disk_size_gb`
- `studio_ide`
- `studio_repo_url`
- `studio_repo_ref`
- `studio_repo_dir`
- `studio_session_name`
- `studio_checkpoint_dir`

Important workload fields:

- `command`
- `tracked_paths`
- `save_every_seconds`
- `grace_period_seconds`
- `work_requirements_file`

## Persistence

- The detached workload runs from `/teamspace/studios/this_studio/train-once-quant-platform-studio`.
- The repo directory is automation-managed by the workflows and refreshed from GitHub on restart.
- Checkpoints live outside the repo tree at `/teamspace/studios/this_studio/.lightning-checkpoints/train-once-build-data`, so a repo refresh does not erase saved state.
- The same checkpoint wrapper restores tracked paths on restart and snapshots them again every 4 hours and on shutdown.

## Workflows

- `Launch Lightning Auto-Resume Studio`
  Ensures the target Studio exists, ensures it is running, refreshes the repo clone, bootstraps dependencies in the built-in Studio environment, and launches the detached session.
- `Lightning Studio Snapshot`
  Runs every 5 minutes, captures the Studio/session state, and restarts the detached session if needed.

## Fallback

The GitHub-hosted CPU loop in [github-actions-autoresume-run.yml](/Users/rohan/train-once-quant-platform/.github/workflows/github-actions-autoresume-run.yml) remains in the repo as a non-Lightning fallback, but the primary path is now the free Studio workflow above.
