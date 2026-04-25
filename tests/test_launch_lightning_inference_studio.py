import importlib
import os
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")
    requests_stub.get = None
    sys.modules["requests"] = requests_stub

if "lightning_cloud_utils" not in sys.modules:
    cloud_utils_stub = types.ModuleType("lightning_cloud_utils")
    cloud_utils_stub.ensure_auth_env = lambda: {}
    cloud_utils_stub.json_safe = lambda value: value
    cloud_utils_stub.set_process_env = lambda env: None
    sys.modules["lightning_cloud_utils"] = cloud_utils_stub

if "lightning_studio_utils" not in sys.modules:
    studio_utils_stub = types.ModuleType("lightning_studio_utils")
    studio_utils_stub.build_bootstrap_command = lambda config: ""
    studio_utils_stub.build_repo_sync_command = lambda config: ""
    studio_utils_stub.ensure_studio_auth_env = lambda: {}
    studio_utils_stub.ensure_studio_exists = lambda *args, **kwargs: None
    studio_utils_stub.ensure_studio_running = lambda *args, **kwargs: None
    studio_utils_stub.execute_studio_command = lambda *args, **kwargs: None
    studio_utils_stub.get_client_and_project = lambda *args, **kwargs: (None, None)
    studio_utils_stub.get_session_status = lambda *args, **kwargs: None
    studio_utils_stub.load_studio_config = lambda *args, **kwargs: None
    studio_utils_stub.resolve_studio_instance = lambda *args, **kwargs: None
    studio_utils_stub.wait_for_session_status = lambda *args, **kwargs: None
    sys.modules["lightning_studio_utils"] = studio_utils_stub


module = importlib.import_module("launch_lightning_inference_studio")
_build_stop_service_command = module._build_stop_service_command
_reuse_existing_service_allowed = module._reuse_existing_service_allowed


class LaunchLightningInferenceStudioTest(unittest.TestCase):
    def test_build_stop_service_command_has_pgrep_fallback(self) -> None:
        command = _build_stop_service_command(8000, "trading-bot-inference")
        self.assertIn("command -v lsof", command)
        self.assertIn("command -v pgrep", command)
        self.assertIn("command -v screen", command)
        self.assertIn("uvicorn trained_model_service_runtime:app.*--port 8000", command)
        self.assertIn(".session_trading-bot-inference", command)

    def test_reuse_existing_service_is_opt_in(self) -> None:
        old = os.environ.pop("LIGHTNING_INFERENCE_REUSE_EXISTING_SERVICE", None)
        try:
            self.assertFalse(_reuse_existing_service_allowed())
            os.environ["LIGHTNING_INFERENCE_REUSE_EXISTING_SERVICE"] = "1"
            self.assertTrue(_reuse_existing_service_allowed())
        finally:
            if old is None:
                os.environ.pop("LIGHTNING_INFERENCE_REUSE_EXISTING_SERVICE", None)
            else:
                os.environ["LIGHTNING_INFERENCE_REUSE_EXISTING_SERVICE"] = old


if __name__ == "__main__":
    unittest.main()
