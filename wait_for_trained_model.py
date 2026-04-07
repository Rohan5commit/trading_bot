import json
import os

import yaml

from trained_model_client import TrainedModelTradeClient


def load_config(path="config.yaml"):
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def main():
    config = load_config()
    ai_cfg = config.get("ai_trading", {}) if isinstance(config, dict) else {}
    client = TrainedModelTradeClient(ai_cfg)
    if not client.is_ready():
        raise SystemExit(client.last_error or "trained model client is not configured")

    timeout_seconds = int(os.getenv("TRAINED_MODEL_READY_TIMEOUT_SECONDS", "720") or 720)
    poll_seconds = float(os.getenv("TRAINED_MODEL_READY_POLL_SECONDS", "15") or 15)
    payload = client.wait_until_ready(timeout_seconds=timeout_seconds, poll_seconds=poll_seconds)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
