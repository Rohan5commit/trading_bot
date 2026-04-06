from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from modal_build_corpus import build_app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--memory", type=int, default=1024)
    parser.add_argument("--gpu", default="")
    parser.add_argument("--summary", default="corpus_summary.json")
    parser.add_argument("--progress", default="progress.json")
    args = parser.parse_args()

    app = build_app(args.cpu, args.memory, args.gpu)
    with app.run():
        summary = app.stats.remote()
        progress = app.progress.remote()

    Path(args.summary).write_text(json.dumps(summary, indent=2))
    Path(args.progress).write_text(json.dumps(progress, indent=2))


if __name__ == "__main__":
    main()
