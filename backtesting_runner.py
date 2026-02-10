import os
import subprocess
import sys


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    backtesting_dir = os.path.join(base_dir, "backtesting")
    if not os.path.isdir(backtesting_dir):
        print("Backtesting folder not found.")
        return 1

    args = sys.argv[1:] or ["backtest"]
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = backtesting_dir + (os.pathsep + pythonpath if pythonpath else "")

    return subprocess.call(
        [sys.executable, "-m", "src.main", *args],
        cwd=backtesting_dir,
        env=env
    )


if __name__ == "__main__":
    raise SystemExit(main())
