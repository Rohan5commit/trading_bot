from launch_lightning_inference_studio import _build_stop_service_command, _reuse_existing_service_allowed


def test_build_stop_service_command_has_pgrep_fallback() -> None:
    command = _build_stop_service_command(8000, "trading-bot-inference")
    assert "command -v lsof" in command
    assert "command -v pgrep" in command
    assert "command -v screen" in command
    assert "uvicorn trained_model_service_runtime:app.*--port 8000" in command
    assert ".session_trading-bot-inference" in command


def test_reuse_existing_service_is_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("LIGHTNING_INFERENCE_REUSE_EXISTING_SERVICE", raising=False)
    assert _reuse_existing_service_allowed() is False
    monkeypatch.setenv("LIGHTNING_INFERENCE_REUSE_EXISTING_SERVICE", "1")
    assert _reuse_existing_service_allowed() is True
