from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import modal


for candidate in ("/root", str(Path(__file__).resolve().parents[1])):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)


def _ignore_local_path(path: Path) -> bool:
    ignored_parts = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    ignored_names = {".DS_Store", "merge_summary.json", "artifacts.tar.gz"}
    return any(part in ignored_parts for part in path.parts) or path.name in ignored_names


IMAGE = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "peft",
        "trl",
        "bitsandbytes",
        "datasets",
        "pyyaml",
        "xgboost",
        "pandas",
        "numpy",
    )
    .add_local_dir("src", "/root/src", ignore=_ignore_local_path)
    .add_local_dir("scripts", "/root/scripts", ignore=_ignore_local_path)
    .add_local_dir("configs", "/root/configs")
)

ARTIFACTS_VOLUME = "train-once-artifacts"
DATA_VOLUME = "train-once-data"
DEFAULT_GPU = os.getenv("MODAL_GPU", "L40S")
DEFAULT_TIMEOUT_SECONDS = int(os.getenv("MODAL_GPU_TIMEOUT_SECONDS", str(60 * 60 * 10)))


def _modal_secrets() -> list[modal.Secret]:
    keys = ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "WANDB_API_KEY", "NVIDIA_NIM_API_KEY"]
    env = {key: os.environ[key] for key in keys if os.environ.get(key)}
    if not env:
        return []
    return [modal.Secret.from_dict(env)]


app = modal.App("train-once-quant-platform")
artifacts = modal.Volume.from_name(ARTIFACTS_VOLUME, create_if_missing=True)
data = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)


@app.function(
    gpu=DEFAULT_GPU,
    timeout=DEFAULT_TIMEOUT_SECONDS,
    image=IMAGE,
    volumes={"/artifacts": artifacts, "/data": data},
    secrets=_modal_secrets(),
)
def train(config_path: str) -> dict:
    os.chdir("/root")
    os.environ["PYTHONPATH"] = "/root"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    subprocess.run(["python", "-m", "src.training.train_lora", "--config", config_path], check=True)

    Path("/artifacts").mkdir(parents=True, exist_ok=True)
    subprocess.run(["bash", "-lc", "cp -r artifacts/. /artifacts/"], check=True)

    artifacts.commit()
    metadata_path = Path("/artifacts/lora_adapter/training_metadata.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    return {"status": "ok", "metadata": metadata}


@app.function(image=IMAGE, volumes={"/artifacts": artifacts})
def package() -> bytes:
    subprocess.run(["tar", "-czf", "/tmp/artifacts.tar.gz", "-C", "/artifacts", "."], check=True)
    return Path("/tmp/artifacts.tar.gz").read_bytes()


@app.local_entrypoint()
def main(
    config: str = "configs/training.yaml",
    out: str = "",
) -> None:
    train.remote(config)
    if out:
        data_bytes = package.remote()
        Path(out).write_bytes(data_bytes)
