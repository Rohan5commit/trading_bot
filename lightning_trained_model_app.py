from __future__ import annotations

import os
from pathlib import Path

from lightning_app import BuildConfig, CloudCompute, LightningApp, LightningFlow, LightningWork


ROOT_DIR = Path(__file__).resolve().parent
REQUIREMENTS_FILE = ROOT_DIR / "requirements-lightning-inference.txt"
DEFAULT_COMPUTE_NAME = os.getenv("LIGHTNING_INFERENCE_COMPUTE_NAME", "cpu")
DEFAULT_DISK_SIZE_GB = int(os.getenv("LIGHTNING_INFERENCE_DISK_GB", "80") or 80)
DEFAULT_PORT = int(os.getenv("LIGHTNING_INFERENCE_PORT", "8000") or 8000)


class TrainedModelInferenceWork(LightningWork):
    def __init__(self) -> None:
        build_config = BuildConfig(requirements=[str(REQUIREMENTS_FILE.resolve())])
        cloud_compute = CloudCompute(name=DEFAULT_COMPUTE_NAME, disk_size=DEFAULT_DISK_SIZE_GB)
        super().__init__(
            parallel=True,
            port=DEFAULT_PORT,
            raise_exception=False,
            cloud_build_config=build_config,
            cloud_compute=cloud_compute,
        )

    def run(self) -> None:
        import uvicorn

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        uvicorn.run(
            "trained_model_service_runtime:app",
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

    def configure_layout(self):
        return [{"name": "trained-model-inference", "content": self.inference.url}]


app = LightningApp(RootFlow())
