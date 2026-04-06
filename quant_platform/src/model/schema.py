from __future__ import annotations

from pydantic import BaseModel


class SupportingFactors(BaseModel):
    tabular_score: float
    llm_score: float
    technical_score: float


class SignalResult(BaseModel):
    ticker: str
    date: str
    score: float
    direction: str
    confidence: float
    factors: SupportingFactors
