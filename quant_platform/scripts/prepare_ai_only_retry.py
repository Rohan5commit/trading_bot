from __future__ import annotations

import argparse
import json
from pathlib import Path


AI_RESULT_PATTERNS = (
    "daily_report_ai_*.csv",
    "unrealized_ai_*.csv",
    "trades_ai_*.csv",
)


def clear_ai_retry_state(results_dir: Path) -> dict:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    removed: list[str] = []
    kept: list[str] = []

    for path in results_dir.glob("email_sent*.ok"):
        name = path.name
        is_generic = name.startswith("email_sent_") and not name.startswith("email_sent_core_") and not name.startswith("email_sent_ai_")
        is_ai = name.startswith("email_sent_ai_")
        if is_generic or is_ai:
            try:
                path.unlink()
                removed.append(name)
            except FileNotFoundError:
                pass
        else:
            kept.append(name)

    for pattern in AI_RESULT_PATTERNS:
        for path in results_dir.glob(pattern):
            try:
                path.unlink()
                removed.append(path.name)
            except FileNotFoundError:
                pass

    return {
        "results_dir": str(results_dir),
        "removed": sorted(set(removed)),
        "kept_markers": sorted(set(kept)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    payload = clear_ai_retry_state(Path(args.results_dir))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
