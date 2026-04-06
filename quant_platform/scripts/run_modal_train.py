from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.modal_train import build_app

DEFAULT_GPU = os.getenv("MODAL_GPU", "L40S")

app = build_app(DEFAULT_GPU)


@app.local_entrypoint()
def main(
    config: str = "configs/training.yaml",
    out: str = "artifacts.tar.gz",
) -> None:
    app.train.remote(config)
    data = app.package.remote()
    Path(out).write_bytes(data)
