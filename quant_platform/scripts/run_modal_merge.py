from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.modal_merge_corpus import build_app

DEFAULT_CPU = int(os.getenv("MODAL_CPU", "16"))
DEFAULT_MEMORY_MB = int(os.getenv("MODAL_MEMORY_MB", "65536"))

app = build_app(DEFAULT_CPU, DEFAULT_MEMORY_MB)


@app.local_entrypoint()
def main(
    config: str = "configs/data_large.yaml",
    summary: str = "merge_summary.json",
) -> None:
    result = app.merge.remote(config)
    Path(summary).write_text(json.dumps(result, indent=2))
