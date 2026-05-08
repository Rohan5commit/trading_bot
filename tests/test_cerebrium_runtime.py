from __future__ import annotations

from unittest.mock import Mock, patch

from trained_model_client import TrainedModelTradeClient
from trained_model_service_runtime import _candidate_prompt
from quant_platform.scripts.plan_ai_runtime import choose_runtime


def test_cerebrium_env_takes_primary_url_and_key(monkeypatch):
    monkeypatch.setenv("AI_PRIMARY_BACKEND", "cerebrium")
    monkeypatch.setenv("CEREBRIUM_INFERENCE_URL", "https://example.test/app")
    monkeypatch.setenv("CEREBRIUM_API_KEY", "secret-token")
    monkeypatch.setenv("TRAINED_MODEL_INFERENCE_URL", "https://rollback.test/app")

    client = TrainedModelTradeClient({"trained_model": {"inference_url_env": "TRAINED_MODEL_INFERENCE_URL"}})

    assert client.provider == "cerebrium"
    assert client.inference_url == "https://example.test/app"
    assert client._request_headers()["Authorization"] == "Bearer secret-token"


def test_manager_context_is_sent_to_http_payload(monkeypatch):
    monkeypatch.setenv("AI_PRIMARY_BACKEND", "cerebrium")
    monkeypatch.setenv("CEREBRIUM_INFERENCE_URL", "https://example.test/app")
    client = TrainedModelTradeClient({"trained_model": {"batch_size": 8}})
    response = Mock()
    response.status_code = 200
    response.headers = {"x-cerebrium-request-id": "req-1"}
    response.json.return_value = {"model": "quant-trained-trading-model", "signals": [{"label": "BUY", "confidence": 0.7}]}
    response.raise_for_status.return_value = None

    with patch("trained_model_client.requests.post", return_value=response) as post:
        result = client.predict_candidates(
            [{"symbol": "AAPL"}],
            manager_context={"last_backend": "distilled_local", "top_symbol_biases": []},
        )

    assert result[0]["label"] == "BUY"
    assert post.call_args.kwargs["json"]["manager_context"]["last_backend"] == "distilled_local"
    assert client.last_request_id == "req-1"


def test_service_prompt_includes_compact_symbol_memory_context():
    prompt = _candidate_prompt(
        {
            "symbol": "AAPL",
            "return_1d": 1,
            "return_5d": 2,
            "return_10d": 3,
            "manager_context": {
                "last_backend": "distilled_local",
                "top_symbol_biases": [{"symbol": "AAPL", "side": "LONG", "bias": 0.25, "confidence": 0.5}],
            },
        }
    )

    assert "LB=distilled_local" in prompt
    assert "MEM_LONG=0.250/0.50" in prompt


def test_plan_runtime_prefers_cerebrium_when_url_configured(monkeypatch):
    monkeypatch.setenv("CEREBRIUM_INFERENCE_URL", "https://example.test/app")

    plan = choose_runtime()

    assert plan["runtime_mode"] == "cerebrium_full"
    assert plan["selected_backend"] == "cerebrium_full"
