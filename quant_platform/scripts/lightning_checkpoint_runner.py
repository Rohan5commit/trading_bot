from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lightning_checkpoint import run_checkpointed_command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shell-command", required=True, help="Shell command to execute inside the Lightning run.")
    parser.add_argument("--workdir", default=".", help="Working directory for the wrapped command.")
    parser.add_argument("--checkpoint-dir", default=".lightning-checkpoints")
    parser.add_argument("--tracked-path", action="append", dest="tracked_paths", required=True)
    parser.add_argument("--save-every-seconds", type=int, default=4 * 60 * 60)
    parser.add_argument("--grace-period-seconds", type=int, default=300)
    parser.add_argument(
        "--shutdown-exit-code",
        type=int,
        default=0,
        help="Exit code to return when the wrapper is terminated after saving a shutdown snapshot.",
    )
    args = parser.parse_args()

    exit_code = run_checkpointed_command(
        shell_command=args.shell_command,
        workdir=(ROOT_DIR / args.workdir).resolve() if not Path(args.workdir).is_absolute() else Path(args.workdir),
        checkpoint_dir=(ROOT_DIR / args.checkpoint_dir).resolve()
        if not Path(args.checkpoint_dir).is_absolute()
        else Path(args.checkpoint_dir),
        tracked_paths=args.tracked_paths,
        save_every_seconds=args.save_every_seconds,
        grace_period_seconds=args.grace_period_seconds,
        shutdown_exit_code=args.shutdown_exit_code,
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
