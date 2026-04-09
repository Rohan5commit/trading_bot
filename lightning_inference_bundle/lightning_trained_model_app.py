from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

try:
    from lightning.app import BuildConfig, CloudCompute, LightningApp, LightningFlow, LightningWork
except ModuleNotFoundError:
    from lightning_app import BuildConfig, CloudCompute, LightningApp, LightningFlow, LightningWork


DEFAULT_COMPUTE_NAME = os.getenv("LIGHTNING_INFERENCE_COMPUTE_NAME", "cpu-4")
DEFAULT_DISK_SIZE_GB = int(os.getenv("LIGHTNING_INFERENCE_DISK_GB", "80") or 80)
DEFAULT_PORT = int(os.getenv("LIGHTNING_INFERENCE_PORT", "8000") or 8000)
RUNTIME_REQUIREMENTS_FILE = Path(__file__).with_name("runtime-requirements.txt")
_BOOTSTRAP_SENTINEL = Path("/tmp/trading_bot_lightning_runtime_ready")
_REQUIRED_MODULES = (
    "fastapi",
    "uvicorn",
    "requests",
    "torch",
    "huggingface_hub",
    "tokenizers",
    "transformers",
    "peft",
    "accelerate",
    "safetensors",
    "sentencepiece",
)


def _missing_runtime_modules() -> list[str]:
    return [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]


def _ensure_runtime_dependencies() -> None:
    os.environ["PATH"] = f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}".strip(":")
    missing = _missing_runtime_modules()
    if not missing and _BOOTSTRAP_SENTINEL.exists():
        return
    if not RUNTIME_REQUIREMENTS_FILE.exists():
        raise FileNotFoundError(f"Missing runtime requirements file: {RUNTIME_REQUIREMENTS_FILE}")

    env = os.environ.copy()
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")

    def _run(cmd: list[str]) -> None:
        subprocess.check_call(cmd, env=env)

    try:
        _run([sys.executable, "-m", "pip", "--version"])
    except subprocess.CalledProcessError:
        _run([sys.executable, "-m", "ensurepip", "--upgrade"])

    install_cmd = [sys.executable, "-m", "pip", "install", "-r", str(RUNTIME_REQUIREMENTS_FILE)]
    try:
        _run(install_cmd)
    except subprocess.CalledProcessError:
        _run(install_cmd[:4] + ["--user"] + install_cmd[4:])

    remaining = _missing_runtime_modules()
    if remaining:
        raise RuntimeError(f"Missing runtime dependencies after bootstrap: {remaining}")
    _BOOTSTRAP_SENTINEL.write_text("ok\n")


class TrainedModelInferenceWork(LightningWork):
    def __init__(self) -> None:
        build_config = BuildConfig(requirements=[str(RUNTIME_REQUIREMENTS_FILE)])
        cloud_compute = CloudCompute(name=DEFAULT_COMPUTE_NAME, disk_size=DEFAULT_DISK_SIZE_GB)
        super().__init__(
            parallel=True,
            port=DEFAULT_PORT,
            raise_exception=False,
            cloud_build_config=build_config,
            cloud_compute=cloud_compute,
        )

    def run(self) -> None:
        _ensure_runtime_dependencies()
        import uvicorn
        from trained_model_service_runtime import app as service_app

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        uvicorn.run(
            service_app,
            host="0.0.0.0",
            port=self.port,
            log_level=os.getenv("TRAINED_MODEL_LOG_LEVEL", "info").lower(),
        )


class RootFlow(LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.inference = TrainedModelInferenceWork()

    def run(self) -> None:
        self.inference.run()


app = LightningApp(RootFlow())
