from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class ProvenanceRecord:
    dataset: str
    source: str
    retrieved_at: str
    last_updated: str | None
    parameters: Dict[str, Any]
    output_path: str | None


class ProvenanceLogger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: ProvenanceRecord) -> None:
        payload = {
            "dataset": record.dataset,
            "source": record.source,
            "retrieved_at": record.retrieved_at,
            "last_updated": record.last_updated,
            "parameters": record.parameters,
            "output_path": record.output_path,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
