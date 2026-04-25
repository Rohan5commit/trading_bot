from launch_lightning_inference_studio import _build_stop_service_command


def test_build_stop_service_command_has_pgrep_fallback() -> None:
    command = _build_stop_service_command(8000)
    assert "command -v lsof" in command
    assert "command -v pgrep" in command
    assert "uvicorn trained_model_service_runtime:app.*--port 8000" in command
