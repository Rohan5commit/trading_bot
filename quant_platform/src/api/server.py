from __future__ import annotations

from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from data.feature_store import FeatureStore
from model.inference_client import ModelInferenceClient, InferenceConfig

app = FastAPI()
client = ModelInferenceClient(InferenceConfig())
feature_store = FeatureStore()


class PredictRequest(BaseModel):
    ticker: str
    date: str


class BacktestRangeRequest(BaseModel):
    ticker: str
    start: str
    end: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {"backend": client.config.backend, "artifacts": str(client.artifacts_path)}


@app.post("/predict")
def predict(req: PredictRequest):
    return client.predict(req.ticker, req.date).model_dump()


@app.post("/backtest-range")
def backtest_range(req: BacktestRangeRequest):
    frame = feature_store.build_feature_frame([req.ticker], req.start, req.end, normalize=True)
    rows = []
    for date in sorted(frame["date"].unique()):
        result = client.predict(req.ticker, str(date))
        rows.append(result.model_dump())
    return rows
