from model.inference_client import ModelInferenceClient


def test_technical_score():
    client = ModelInferenceClient()
    score = client._technical_score({"rsi_14": 20, "macd": 1, "ema_9": 5, "ema_21": 4})
    assert score > 0
