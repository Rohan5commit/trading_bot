from __future__ import annotations

import os

from src.lightning_cloud_runtime import build_app


app = build_app(os.environ.get("LIGHTNING_RUN_CONFIG"))
